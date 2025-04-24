import sys
import copy
import torch
import numpy as np
import torch.nn as nn
import abc
#import ipdb
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from functools import partial
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.data import DataLoader
class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()
        self.dropout = dropout
        # Create linear layers
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(2*in_channels, hidden_channels))
        #self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.lins.append(nn.ReLU())
        #self.lins.append(nn.Dropout(self.dropout))
        for _ in range(num_layers):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(nn.ReLU())
            #self.lins.append(nn.Dropout(self.dropout))
        self.lins.append(nn.Linear(hidden_channels, out_channels))
        #self.lins.append(nn.ReLU())

        

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        # x_i and x_j are both of shape (E, D)
        #x = x_i * x_j
        x=torch.cat((x_i,x_j),1)
        
        #2import pdb;pdb.set_trace()
        for lin in self.lins:
            x = lin(x)
            #x = F.relu(x)
            #x = F.dropout(x, p=self.dropout, training=self.training)
        #x = self.lins[-1](x)
        return x

class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, emb=False):
        super(GNNStack, self).__init__()
        conv_model = pyg.nn.GCNConv

        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))
        self.dropout = dropout
        self.num_layers = num_layers
        self.emb = emb

        # Create num_layers GraphSAGE convs
        assert (self.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(self.num_layers - 1):
            self.convs.append(conv_model(hidden_dim, hidden_dim))

        # post-message-passing processing 
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, output_dim),nn.ReLU())

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.post_mp(x)

        # Return final layer of embeddings if specified
        if self.emb:
            return x
        return x
        # Else return class probabilities
        #return F.log_softmax(x, dim=1)

    #def loss(self, pred, label):
    #    return F.nll_loss(pred, label)



class BaseModelAdapter(BaseEstimator):
    __metaclass__ = abc.ABCMeta

    def __init__(self, model, fit_params=None):
        super(BaseModelAdapter, self).__init__()

        self.model = model
        self.last_x, self.last_y = None, None
        self.clean = False
        self.fit_params = {} if fit_params is None else fit_params

    def fit(self, x, y):
        self.model.fit(x, y, **self.fit_params)
        self.clean = False

    def predict(self, x):
        if (not self.clean or
            self.last_x is None or
            self.last_y is None or
            not np.array_equal(self.last_x, x)
        ):
            self.last_x = x
            self.last_y = self._underlying_predict(x)
            self.clean = True

        return self.last_y.copy()

    @abc.abstractmethod
    def _underlying_predict(self, x):
        pass


class RegressorAdapter(BaseModelAdapter):
    def __init__(self, model, fit_params=None):
        super(RegressorAdapter, self).__init__(model, fit_params)

    def _underlying_predict(self, x):
        return self.model.predict(x)
def epoch_internal_train(model, loss_func, pos_train, train_y, batch_size, emb, edge_index, optimizer, cnt=0, best_cnt=np.Inf,device='cuda'):
    model.encoder=model.encoder.to(device)
    model.g=model.g.to(device)
    model.encoder.train()
    model.g.train()
    #shuffle_idx = np.arange(x_train.shape[0])
    #np.random.shuffle(shuffle_idx)
    #x_train = x_train[shuffle_idx]
    #y_train = y_train[shuffle_idx]
    epoch_losses = []
    emb=emb.to(device)
    edge_index=edge_index.to(device)
    #node_emb=model.encoder(emb, edge_index)
    #print(pos_train.shape[0])
    for edge_id,b_y in zip(DataLoader(range(pos_train.shape[0]), batch_size, shuffle=False),DataLoader(range(len(train_y)), batch_size, shuffle=False)):
        #print(2)
        #print(edge_id)
        #cnt = cnt + 1
        #print(i)
        node_emb=model.encoder(emb, edge_index)
        optimizer.zero_grad()
        pos_edge = pos_train[edge_id].T  # (2, B)
        #import pdb;pdb.set_trace()
        pos_pred = model.g(node_emb[pos_edge[0]], node_emb[pos_edge[1]])  # (B, )
        #preds = model(batch_x)
        batch_y=train_y[b_y].to(device)
        #print(pos_pred.flatten())
        #print(batch_y)
        #import pdb;pdb.set_trace()
        loss = loss_func(pos_pred.flatten(), batch_y)
        #print(loss)

        #print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_losses.append(loss.cpu().detach().numpy())
        #i+=1
        #if cnt >= best_cnt:
            #break
    
    #for edge_id in DataLoader(range(neg_train.shape[0]), batch_size, shuffle=True):
        #cnt = cnt + 1
        #optimizer.zero_grad()
        #node_emb=model.encoder(emb, edge_index)
        #neg_edge = neg_train[edge_id].T  # (2, B)
        #neg_pred = model.g(node_emb[neg_edge[0]], node_emb[neg_edge[1]])  # (B, )
        #preds = model(batch_x)
        #batch_y=torch.zeros(len(neg_pred))
        #loss = loss_func(neg_pred, batch_y)
        #print(loss)
        #loss.backward()
        #optimizer.step()
        #epoch_losses.append(loss.cpu().detach().numpy())
        #optimizer.zero_grad()
        #if cnt >= best_cnt:
            #break
    
    torch.cuda.empty_cache()
    epoch_loss = np.mean(epoch_losses)

    return epoch_loss,emb,edge_index


class LearnerOptimized:
    def __init__(self, model, optimizer_class, loss_func, emb, edge_index, device='cpu', test_ratio=0.2, random_state=0):
        self.model=model
        #self.model_f = model.encoder.to(device)
        #self.model_g = model.g.to(device)
        self.optimizer_class = torch.optim.Adam
        self.optimizer = 'Adam'
        self.loss_func = loss_func.to(device)
        self.device = device
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.emb=emb.to(device)
        self. edge_index= edge_index
        self.loss_history = []
        self.test_loss_history = []
        self.full_loss_history = []
        #testing_losses=[]

    def fit(self, pos_train, train_y, pos_test,test_y, epochs, batch_size, verbose=False):
        sys.stdout.flush()
        model = copy.deepcopy(self.model)
        model = model.to(self.device)
        optimizer = self.optimizer_class(list(model.encoder.parameters())+list(model.g.parameters()))
        best_epoch = epochs

        #x_train, xx, y_train, yy = train_test_split(x, y, test_size=self.test_ratio, random_state=self.random_state) # Need to work on this

        #x_train = torch.from_numpy(x_train).float().to(self.device).requires_grad_(False)
        #xx = torch.from_numpy(xx).float().to(self.device).requires_grad_(False)
        #y_train = torch.from_numpy(y_train).float().to(self.device).requires_grad_(False)
        #yy = torch.from_numpy(yy).float().to(self.device).requires_grad_(False)

        best_cnt = 1e10
        best_test_epoch_loss = 1e10
        test_y=test_y.cpu()
        cnt = 0
        for e in range(epochs):
            #import pdb;pdb.set_trace()
            epoch_loss, emb,edge_index = epoch_internal_train(model, self.loss_func, pos_train, train_y, batch_size, self.emb,
                                                   self.edge_index, optimizer, cnt,self.device)
            #print('Epoch')
            #print(e)
            #print(epoch_loss)
            self.loss_history.append(epoch_loss)
            #pos_test_preds=[]
            # test
            testing_losses=[]
            model.encoder.eval()
            model.g.eval()

            node_emb=model.encoder(emb,edge_index)
            px=0
            loss2=[]
            for perm,b_y1 in zip(DataLoader(range(pos_test.size(0)), batch_size),DataLoader(range(len(test_y)),batch_size)):
                edge = pos_test[perm].t()
                pos_test_preds= model.g(node_emb[edge[0]], node_emb[edge[1]]).squeeze().cpu()
                batch_y1=test_y[b_y1]
                #batch_y1=batch_y1.to(self.device)
            #print(pos_test_preds)
            #px=torch.tensor(pos_test_preds)
            #print(batch_size.size())
            #print(px.size())
                #import pdb; pdb.set_trace()
                loss = self.loss_func(pos_test_preds, batch_y1)
                loss2.append(loss.cpu().detach().numpy())
            testing_losses.append(np.mean(loss2))
            #loss2=[]
            #for perm in DataLoader(range(neg_test.size(0)), batch_size):
                #edge = neg_test[perm].t()
                #neg_test_preds= model.g(node_emb[edge[0]], node_emb[edge[1]]).squeeze().cpu()
                #batch_y2=torch.zeros(len(neg_test_preds))
                #loss = self.loss_func(neg_test_preds, batch_y2)
                #loss2.append(loss.cpu().detach().numpy())
            #testing_losses.append(np.mean(loss2))
            test_epoch_loss = np.mean(testing_losses)
            self.test_loss_history.append(test_epoch_loss)

            #test_preds = preds.cpu().detach().numpy()
            #test_preds = np.squeeze(test_preds)

            if (test_epoch_loss <= best_test_epoch_loss):
                best_test_epoch_loss = test_epoch_loss
                best_epoch = e
                self.model = copy.deepcopy(model)
                #best_cnt = cnt

            if (e + 1) % 1 == 0 and verbose:
                print("CV: Epoch {}: Train {}, Test {}, Best epoch {}, Best loss {}".format(e + 1, epoch_loss,
                                            test_epoch_loss, best_epoch, best_test_epoch_loss))
                sys.stdout.flush()
        #print('my')
        #for param in self.model.encoder.parameters():
        #    print(param.data)
        # use all the data to train the model, for best_cnt steps
        #x = torch.from_numpy(x).float().to(self.device).requires_grad_(False)
        #y = torch.from_numpy(y).float().to(self.device).requires_grad_(False)

        best_cnt = 20
        return self.model # Harcoded
        #for e in range(best_epoch + 1):
            #if cnt > best_cnt:
                #break

            #epoch_loss, cnt = epoch_internal_train(self.model, self.loss_func, x, y, batch_size,
            #                                       self.optimizer, cnt, best_cnt)
            #self.full_loss_history.append(epoch_loss)

            #if (e + 1) % 100 == 0 and verbose:
                #print("Full: Epoch {}: {}, cnt {}".format(e + 1, epoch_loss, cnt))
                #sys.stdout.flush()
        

    def predict(self,emb,edge_index,pos_test_edge):
        #self.model.g=self.model.g.cpu()
        #self.model.encoder=self.model.encoder.cpu()
        self.model.encoder.eval()
        self.model.g.eval()
        #emb=emb.cpu()
        #edge_index=edge_index.cpu()
        node_emb=self.model.encoder(emb,edge_index)
        all_preds = []
        for perm in DataLoader(range(pos_test_edge.size(0)),(1024)):
            edge = pos_test_edge[perm].t()
            #import pdb;pdb.set_trace()
            pos_test_preds= self.model.g(node_emb[edge[0]], node_emb[edge[1]]).squeeze().cpu()
            all_preds.append(pos_test_preds.detach().numpy())
        ret_val=np.concatenate(all_preds, axis=0)
        #ret_val = self.model(x).cpu().detach().numpy()
        return ret_val

class mse_model(nn.Module):
    def __init__(self, in_shape=128, out_shape=1, hidden_size=64,f_layers=2,g_layers=2, dropout=0.5,
                 pos_train=None,pos_cal=None,pos_test=None,train_y=None,cal_y=None,test_y=None,
                 edge_index=None,batch_size=(64*1024),emb=None):

        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.f_layers=f_layers
        self.g_layers=g_layers
        self.build_model()
        #self.init_weights()
        self.pos_train=pos_train
        self.pos_cal=pos_cal
        self.pos_test=pos_test
        selftrain_y=train_y
        self.cal_y=cal_y
        self.test_y=test_y
        self.edge_index=edge_index
        self.batch_size=batch_size
        self.emb=emb
    def build_model(self):
        self.encoder = GNNStack(self.in_shape, self.hidden_size, self.hidden_size, self.f_layers, self.dropout, emb=False)
        self.g = LinkPredictor(self.hidden_size, self.hidden_size, 1, self.g_layers, self.dropout)

        #self.base_model = nn.Sequential(self.encoder, self.g)

    def forward(self,edge_id,ids): #pos_train or whatever
          
        #X1=self.encoder(self.emb,self.edge_index)
        #pos_edge=ids[edge_id].T
        #X2=self.g(X1[pos_edge[0]],X1[pos_edge[1]])
        return edge_id

        #pos_pred = self.g(X1[pos_edge[0]], X1[pos_edge[1]])  # (B, )

        # Sample negative edges (same number as number of positive edges) and predict class probabilities 
        #neg_edge = negative_sampling(edge_index, num_nodes=emb.shape[0],
        #                             num_neg_samples=edge_id.shape[0], method='dense')  # (Ne,2)

"""     def init_weights(self):
        for m in self.base_model:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0) """

    
        

class MSENet_RegressorAdapter(RegressorAdapter):
    def __init__(self, model, device, args, fit_params=None, in_shape=1, out_shape=1, hidden_size=1, 
                 learn_func=torch.optim.Adam, epochs=1000, batch_size=10,
                 dropout=0.1, lr=0.01, wd=1e-6, test_ratio=0.2, random_state=0,num_layers=2,pos_rt_edge=None,pos_cal_edge=None,
                 pos_test_edge=None,train_y=None,cal_y=None, test_y=None,edge_index=None,emb=None):
        super(MSENet_RegressorAdapter, self).__init__(model, fit_params)
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.lr = lr
        self.wd = wd
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.device = device
        self.num_layers=num_layers
        #self.emb=emb
        #self.edge_index=edge_index
        if model is None:
            self.model = mse_model(in_shape=in_shape, out_shape=out_shape, hidden_size=hidden_size,f_layers=args.GraphSAGE_convs,g_layers=args.FFN_layers,
                                 dropout=args.dropout,pos_train=pos_rt_edge, pos_cal=pos_cal_edge, pos_test=pos_test_edge,
                                 train_y=train_y,cal_y=cal_y,test_y=test_y,edge_index=edge_index,batch_size=args.batch_size,emb=emb)
        self.loss_func = torch.nn.MSELoss()
        self.learner = LearnerOptimized(self.model, partial(learn_func, lr=lr, weight_decay=wd),
                                        self.loss_func, device=device, test_ratio=self.test_ratio, random_state=self.random_state,emb=emb,edge_index=edge_index)

        self.hidden_size = hidden_size
        self.in_shape = in_shape
        self.learn_func = learn_func

    def fit(self, pos_train, train_y, pos_test,test_y):
        #import pdb;pdb.set_trace()
        model=self.learner.fit(pos_train, train_y, pos_test,test_y, self.epochs, batch_size=self.batch_size, verbose=False)
        self.model = copy.deepcopy(model)
    def predict(self,emb,edge_index,pos_test_edge):
        return self.learner.predict(emb,edge_index,pos_test_edge)