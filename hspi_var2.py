import torch
from torch_geometric.data import Data
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from tqdm import tqdm
import os
import numpy as np
import random
import argparse
#from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.utils import negative_sampling
from conformal import helper
from mapie.metrics import regression_ssc,regression_ssc_score
#from conformal.icp3 import IcpRegressor, FeatRegressorNc, RegressorNc
from conformal.iicp import IcpRegressor, FeatRegressorNc, RegressorNc
from torch_geometric.data import DataLoader
from conformal.utils import compute_coverage
from numpy import save
from conformal.utils import compute_coverage, default_loss
#np.warnings.filterwarnings('ignore')

def makedirs(path):
    if not os.path.exists(path):
        print('creating dir: {}'.format(path))
        os.makedirs(path)
    else:
        print(path, "already exists!")

def seed_torch(seed, verbose=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if verbose:
        print("==> Set seed to {:}".format(seed))

def main (emb, device, edge_index, pos_rt_edge, pos_cal_edge, y_train,y_cal,y_test, pos_test_edge, args,seed):
    dir = f"ckpt/{'hspi'}"
    mod='model'+str(seed)+'.pt'
    #emb=emb.to(device)
    print(os.path.exists(os.path.join(dir, mod)))
    if os.path.exists(os.path.join(dir, mod)) and not args.no_resume:
        model = helper.mse_model(in_shape=128, out_shape=1, hidden_size=args.hidden_size,f_layers=args.GraphSAGE_convs,g_layers=args.FFN_layers,
                                 dropout=args.dropout,pos_train=pos_rt_edge, pos_cal=pos_cal_edge, pos_test=pos_test_edge,train_y=y_train,
                                 cal_y= y_cal,test_y= y_test,edge_index=edge_index,batch_size=args.batch_size,emb=emb)
        print(f"==> Load model from {dir}")
        model.load_state_dict(torch.load(os.path.join(dir, mod), map_location=device))
        print('?????????????????????????')
    else:
        model = None
    #print(model)
    mean_estimator = helper.MSENet_RegressorAdapter(model=model, device=device,args=args, fit_params=None,
                                                    in_shape=128, out_shape=1,hidden_size=args.hidden_size, 
                                                    learn_func=nn_learn_func, epochs=args.epochs,
                                                    batch_size=args.batch_size, dropout=args.dropout, lr=args.lr, wd=args.wd,
                                                    test_ratio=cv_test_ratio, random_state=cv_random_state, pos_rt_edge=pos_rt_edge,
                                                    pos_cal_edge=pos_cal_edge, pos_test_edge=pos_test_edge,train_y=y_train,cal_y=y_cal,
                                                    test_y=y_test,edge_index=edge_index,emb=emb)
    #print(mean_estimator.model)
    if float(args.feat_norm) <= 0 or args.feat_norm == "inf":
        args.feat_norm = "inf"
        print("Use inf as feature norm")
    else:
        args.feat_norm = float(args.feat_norm)
    nc = FeatRegressorNc(mean_estimator, inv_lr=args.feat_lr, inv_step=args.feat_step,
                         feat_norm=args.feat_norm, certification_method=args.cert_method)
    print(nc.err_func)
    icp = IcpRegressor(nc)

    if os.path.exists(os.path.join(dir, mod)) and not args.no_resume:
        pass
    else:
        icp.fit(pos_rt_edge, y_train, pos_test_edge, y_test)
        makedirs(dir)
        print(f"==> Saving model at {dir}/"+mod)
        file='model'+str(seed)+'.pt'
        torch.save(mean_estimator.model.state_dict(), os.path.join(dir, file))
    #
    
    
    #icp.calibrate_batch(emb,edge_index,pos_cal_edge,y_cal)
    icp.calibrate_batch(emb,edge_index,pos_cal_edge,y_cal)
    predictions=icp.predict(emb,edge_index,pos_test_edge,significance=alpha)
    y_lower, y_upper = predictions[..., 0], predictions[..., 1]
    test_coverage, test_length = compute_coverage(y_test.detach().cpu().numpy(), y_lower, y_upper, alpha,name="FeatRegressorNc", verbose=True)
    print('AVG_Length=')
    print(test_length)
    print('AVG_Coverage=')
    print(test_coverage)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "--d", default=0, type=int)
    parser.add_argument('--seed', type=int, nargs='+', default=[0])
    #parser.add_argument("--data", type=str, default="community", help="meps20 fb1 fb2 blog")

    parser.add_argument("--alpha", type=float, default=0.1, help="miscoverage error")
    parser.add_argument("--GraphSAGE_convs", "--fc", type=int, default=2)
    parser.add_argument("--FFN_layers", "--gc", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=50) #20
    parser.add_argument("--batch_size", "--bs", type=int, default=(64*164))
    parser.add_argument("--hidden_size", "--hs", type=int, default=64)
    parser.add_argument("--dropout", "--do", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=1e-6)
    parser.add_argument("--no-resume", action="store_true", default=True)

    parser.add_argument("--feat_opt", "--fo", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--feat_lr", "--fl", type=float, default=5e-5) # 1e-5
    parser.add_argument("--feat_step", "--fs", type=int, default=None) #None
    parser.add_argument("--feat_norm", "--fn", default=-1)
    parser.add_argument("--cert_method", "--cm", type=int, default=0, choices=[0, 1, 2, 3])
    args = parser.parse_args()
    fcp_coverage_list, fcp_length_list, cp_coverage_list, cp_length_list = [], [], [], []
    for seed in tqdm(args.seed):
        seed_torch(seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = "{:}".format(args.device)
        device = torch.device("cpu") if args.device < 0 else torch.device("cuda")
        print(device)
        nn_learn_func = torch.optim.Adam

        # ratio of held-out data, used in cross-validation
        cv_test_ratio = 0.05
        alpha = args.alpha
        cv_random_state = 1

        # Load data from TSV file
        #import pandas as pd

        file_path = 'dataset/HS-PI.tsv'
        df = pd.read_csv(file_path, sep='\t', header=None, names=['source', 'target', 'weight'])

        # Create a mapping between original indices and new indices starting from 0
        unique_nodes = pd.concat([df['source'], df['target']]).unique()
        index_mapping = {original_index: new_index for new_index, original_index in enumerate(unique_nodes)}

        # Apply the mapping to 'source' and 'target' columns
        df['source'] = df['source'].map(index_mapping)
        df['target'] = df['target'].map(index_mapping)

        # Save original indices for tracking line numbers
        df['original_index'] = df.index

        # Split the data into train, validation, and test sets
        train_ratio, val_ratio, test_ratio = 0.8, 0.2, 0.2
        train_df, test_df = train_test_split(df, test_size=test_ratio, random_state=42)

        # Convert dataframes to PyTorch tensors
        def df_to_tensor(df):
            indices = torch.tensor(df[['source', 'target']].values, dtype=torch.long).t().contiguous()
            weights = torch.tensor(df['weight'].values, dtype=torch.float).view(-1, 1)
            return indices, weights, df['original_index'].values

        pos_train_edge, pos_y, train_indices = df_to_tensor(train_df)
        #val_indices, val_weights, val_orig_indices = df_to_tensor(val_df)
        pos_test_edge, y_test, test_indices = df_to_tensor(test_df)

        # Save test indices
        np.save('tensors/pos_test_indices.npy', test_indices)

        # Create PyG graph
        edge_index = torch.cat([pos_train_edge, pos_test_edge], dim=1)
        edge_attr = torch.cat([pos_y, y_test], dim=0)
        graph = Data(edge_index=edge_index, edge_attr=edge_attr)

        pos_test_edge = pos_test_edge.T
        pos_train_edge = pos_train_edge.T
        pos_y = pos_y.squeeze()
        y_test = y_test.squeeze()

        fn = 'tensors/hspi_pos_test_edge_min1_cat' + str(seed) + '.pt'
        torch.save(pos_test_edge, fn)

        edge_index = graph.edge_index.to(device)
        pos_train_edge = pos_train_edge.to(device)
        pos_test_edge = pos_test_edge.to(device)
        pos_y = pos_y.to(device)
        y_test = y_test.to(device)

        n_train = len(pos_train_edge)
        fn = 'tensors/hspi_min1_y_test' + str(seed) + '.pt'
        torch.save(y_test, fn)
        idx = np.random.permutation(n_train)
        n_half = int(np.floor(n_train / 4))
        idx_train1, idx_cal1 = idx[:n_half], idx[n_half:2 * n_half]

        pos_rt_edge = pos_train_edge[idx_train1]
        pos_cal_edge = pos_train_edge[idx_cal1]
        y_train = pos_y[idx_train1]
        y_cal = pos_y[idx_cal1]

        # Save the indices (line numbers) for pos_rt and pos_cal
        train_indices = train_indices[idx]
        train_rt_indices = train_indices[:n_half]
        train_cal_indices = train_indices[n_half:2 * n_half]

        np.save('tensors/pos_rt_indices.npy', train_rt_indices)
        np.save('tensors/pos_cal_indices.npy', train_cal_indices)

        emb = torch.nn.Embedding(graph.num_nodes, 128).to(device) # each node has an embedding that has to be learnt # hard coded for now
        
        emb.requires_grad_(True)
        main(emb.weight,device, edge_index, pos_rt_edge, pos_cal_edge,y_train,y_cal,y_test, pos_test_edge, args,seed)
        
