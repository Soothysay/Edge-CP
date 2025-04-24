from __future__ import division
from collections import defaultdict
from functools import partial
import pdb
import abc
from tqdm import tqdm
import numpy as np
import sklearn.base
from sklearn.base import BaseEstimator
import torch
from .utils import compute_coverage, default_loss
from torch_geometric.data import DataLoader
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
class RegressionErrFunc(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(RegressionErrFunc, self).__init__()

    @abc.abstractmethod
    def apply(self, prediction, y):
        pass

    @abc.abstractmethod
    def apply_inverse(self, nc, significance):
        pass
    
class FeatErrorErrFunc(RegressionErrFunc):
    def __init__(self, feat_norm):
        super(FeatErrorErrFunc, self).__init__()
        self.feat_norm = feat_norm

    def apply(self, prediction, z):
        ret = (prediction - z).norm(p=self.feat_norm, dim=1)
        return ret

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc)[::-1]
        border = int(np.floor(significance * (nc.size + 1))) - 1
        border = min(max(border, 0), nc.size - 1)

        return np.vstack([nc[border], nc[border]])

class AbsErrorErrFunc(RegressionErrFunc):
    def __init__(self):
        super(AbsErrorErrFunc, self).__init__()

    def apply(self, prediction, y):
        err = np.abs(prediction - y)
        if err.ndim > 1:
            err = np.linalg.norm(err, ord=np.inf, axis=1)
        return err

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc)[::-1]
        border = int(np.floor(significance * (nc.size + 1))) - 1
        border = min(max(border, 0), nc.size - 1)
        return np.vstack([nc[border], nc[border]])

class BaseScorer(sklearn.base.BaseEstimator):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(BaseScorer, self).__init__()

    @abc.abstractmethod
    def fit(self, x, y):
        pass

    @abc.abstractmethod
    def score(self, x, y=None):
        pass

    @abc.abstractmethod
    def score_batch(self, dataloader):
        pass

class BaseModelNc(BaseScorer):
    def __init__(self, model, err_func, normalizer=None, beta=1e-6):
        super(BaseModelNc, self).__init__()
        self.err_func = err_func
        self.model = model
        self.normalizer = normalizer
        self.beta = beta

        if (self.normalizer is not None and hasattr(self.normalizer, 'base_model')):
            self.normalizer.base_model = self.model

        self.last_x, self.last_y = None, None
        self.last_prediction = None
        self.clean = False

    def fit(self, pos_train, train_y, pos_test,test_y):
        
        self.model.fit(pos_train, train_y, pos_test,test_y)
        
        self.clean = False

    def score(self, x, y=None):
        n_test = x.shape[0]
        prediction = self.model.predict(x)
        if self.normalizer is not None:
            norm = self.normalizer.score(x) + self.beta
        else:
            norm = np.ones(n_test)

        if prediction.ndim > 1:
            ret_val = self.err_func.apply(prediction, y)
        else:
            ret_val = self.err_func.apply(prediction, y) / norm

        return ret_val

    def score_batch(self,emb,edge_index,pos_cal_edge,y_cal):
        ret_val = []
        #for x, _, y in dataloader:
        prediction = self.model.predict(emb,edge_index,pos_cal_edge)
        if self.normalizer is not None:
            norm = self.normalizer.score(x) + self.beta
        else:
            norm = np.ones(len(prediction))

        if prediction.ndim > 1:
            ret_val = self.err_func.apply(prediction, y.detach().cpu().numpy())
        else:
            ret_val = self.err_func.apply(prediction, y_cal.detach().cpu().numpy()) / norm
            
        return ret_val

class RegressorNc(BaseModelNc):
    def __init__(self, model, err_func=AbsErrorErrFunc(), normalizer=None, beta=1e-6):
        super(RegressorNc, self).__init__(model, err_func, normalizer, beta)

    def predict(self,emb,edge_index,pos_test_edge,nc,n_samples,seed,significance=None):
        n_test = n_samples
        prediction = self.model.predict(emb,edge_index,pos_test_edge)
        #print(prediction)
        norm = np.ones(n_test)

        if significance:
            intervals = np.zeros((n_test, self.model.model.out_shape, 2))
            err_dist = self.err_func.apply_inverse(nc, significance)  # (2, y_dim)
            err_dist = np.stack([err_dist] * n_test)  # (B, 2, y_dim)
            if prediction.ndim > 1:  # CQR
                intervals[..., 0] = prediction - err_dist[:, 0]
                intervals[..., 1] = prediction + err_dist[:, 1]
            else:  # regular conformal prediction
                err_dist *= norm[:, None, None]
                intervals[..., 0] = prediction[:, None] - err_dist[:, 0]
                intervals[..., 1] = prediction[:, None] + err_dist[:, 1]
            #import pdb; pdb.set_trace()
            return intervals
        else:  # Not tested for CQR
            significance = np.arange(0.01, 1.0, 0.01)
            intervals = np.zeros((x.shape[0], 2, significance.size))

            for i, s in enumerate(significance):
                err_dist = self.err_func.apply_inverse(nc, s)
                err_dist = np.hstack([err_dist] * n_test)
                err_dist *= norm

                intervals[:, 0, i] = prediction - err_dist[0, :]
                intervals[:, 1, i] = prediction + err_dist[0, :]

            return intervals


class FeatRegressorNc(BaseModelNc):
    def __init__(self, model,
                 # err_func=FeatErrorErrFunc(),
                 inv_lr, inv_step, criterion=default_loss, feat_norm=np.inf, certification_method=0, cert_optimizer='sgd',
                 normalizer=None, beta=1e-6, g_out_process=None):
        if feat_norm in ["inf", np.inf, float('inf')]:
            self.feat_norm = np.inf
        elif (type(feat_norm) == int or float):
            self.feat_norm = feat_norm
        else:
            raise NotImplementedError
        err_func = FeatErrorErrFunc(feat_norm=self.feat_norm)

        super(FeatRegressorNc, self).__init__(model, err_func, normalizer, beta)
        self.criterion = criterion
        self.inv_lr = inv_lr
        self.inv_step = inv_step
        self.certification_method = certification_method
        self.cmethod = ['IBP', 'IBP+backward', 'backward', 'CROWN-Optimized'][self.certification_method]
        print(f"Use {self.cmethod} method for certification")

        self.cert_optimizer = cert_optimizer
        # the function to post process the output of g, because FCN needs interpolate and reshape
        self.g_out_process = g_out_process

    def inv_g(self,z0, y, step=None, record_each_step=False):
        #
        z = z0.detach().clone()
        z = z.detach()
        #import pdb;pdb.set_trace()
        z.requires_grad_()
        if self.cert_optimizer == "sgd":
            optimizer = torch.optim.SGD([z], lr=self.inv_lr)
        elif self.cert_optimizer == "adam":
            optimizer = torch.optim.Adam([z], lr=self.inv_lr)
        #import pdb;pdb.set_trace()
        self.model.model.g.eval()
        each_step_z = []
        for _ in range(step):
            for perm,b_y in zip(DataLoader(range(z.shape[0]),1024, shuffle=False),DataLoader(range(len(y)), (1024), shuffle=False)):
                batch_y=y[b_y].to(self.model.device)
        
                batch_y=batch_y.to(self.model.device)
                pred = self.model.model.g(z[perm][:,0,:],z[perm][:,1,:])
                
                if self.g_out_process is not None:
                    pred = self.g_out_process(pred)
                # Runoutofmemory
                loss = self.criterion(pred,batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if record_each_step:
                each_step_z.append(z.detach().cpu().clone())

        if record_each_step:
            return each_step_z
        else:
            return z.detach().cpu()

    
    def get_each_step_err_dist(self, x, y, z_pred, steps):
        
        each_step_z_true = self.inv_g(z_pred, y, step=steps, record_each_step=True)

        if self.normalizer is not None:
            raise NotImplementedError
        
        
        err_dist_list = []
        for i, step_z_true in enumerate(each_step_z_true):
            px = self.err_func.apply(z_pred[:,0,:].detach().cpu(), step_z_true[:,0,:].detach().cpu()).numpy() 
            py=self.err_func.apply(z_pred[:,1,:].detach().cpu(), step_z_true[:,1,:].detach().cpu()).numpy()
            #pred=torch.cat([z_pred[:,0,:].detach().cpu(),z_pred[:,1,:].detach().cpu()],dim=1)
            #true=torch.cat([step_z_true[:,0,:].detach().cpu(),step_z_true[:,1,:].detach().cpu()],dim=1) 
            err_dist=np.minimum(px,py)
            #err_dist=self.err_func.apply(pred,true).numpy()
            err_dist_list.append(err_dist)
        
        return err_dist_list


    def coverage_tight(self,pos_edge,batch_y,z_pred,steps,val_significance,pos_cal):
        z_pred_detach = z_pred.detach().clone()

        idx = torch.randperm(len(pos_edge))
        n_val = int(np.floor(len(pos_edge) /3)) #int(np.floor(len(pos_edge) / 3))
        val_idx, cal_idx = idx[:n_val], idx[n_val:]

        cal_x, val_x = pos_edge[cal_idx], pos_edge[val_idx]
        
        cal_y,val_y= batch_y[cal_idx],batch_y[val_idx]
        
        
        cal_z_pred, val_z_pred = z_pred_detach[cal_x],z_pred_detach[val_x]
        # Runtime error
        #import pdb; pdb.set_trace()
        cal_score_list = self.get_each_step_err_dist(cal_x,cal_y, cal_z_pred, steps=steps)
        #val_score_list = self.get_each_step_err_dist(val_x, val_y, val_z_pred, steps=steps)
        #import pdb; pdb.set_trace()
        val_coverage_list = []
        for i, step_cal_score in enumerate(cal_score_list):
            val_predictions=self.predict(z=z_pred.detach(),pos_test_edge=val_x.detach(),nc=step_cal_score,significance=val_significance)
            val_y_lower, val_y_upper = val_predictions[..., 0], val_predictions[..., 1]
            #import pdb;pdb.set_trace()
            val_coverage, _ = compute_coverage(val_y.detach().cpu().numpy(), val_y_lower, val_y_upper, val_significance,
                                               name="{}-th step's validation".format(i), verbose=False)
            val_coverage_list.append(val_coverage)
            #err_dist_threshold = self.err_func.apply_inverse(nc=cal_score, significance=val_significance)[0][0]
            #val_coverage = np.sum(val_score < err_dist_threshold) * 100 / len(val_score)
            #val_coverage_list.append(val_coverage)
        
        return val_coverage_list, len(val_x)


    def find_best_step_num_batch(self,z_pred,pos_cal,y_cal):
        max_inv_steps = 200
        val_significance = 0.1

        accumulate_val_coverage = np.zeros(max_inv_steps)
        accumulate_val_num = 0
        print("begin to find the best step number")
        
        for perm,b_y in zip(DataLoader(range(pos_cal.shape[0]), (64*1024), shuffle=False),DataLoader(range(len(y_cal)), (64*1024), shuffle=False)):
            #print(perm)
            #print(perm.size())
            pos_edge = pos_cal[perm]
            #import pdb; pdb.set_trace()
            batch_y=y_cal[b_y].to(self.model.device)
            
            batch_each_step_val_coverage, val_num = self.coverage_tight(pos_edge,batch_y, z_pred, steps=max_inv_steps, val_significance=val_significance,pos_cal=pos_cal)  # length: max_inv_steps
            #import pdb;pdb.set_trace()
            accumulate_val_coverage += np.array(batch_each_step_val_coverage) * val_num
            accumulate_val_num += val_num

        each_step_val_coverage = accumulate_val_coverage / accumulate_val_num
        #import pdb;pdb.set_trace()

        tolerance = 3
        # better way to do this
        print(each_step_val_coverage)
        #indices = np.where((each_step_val_coverage >= ((1 - val_significance) * 100)) & (each_step_val_coverage <= (((1 - val_significance) * 100) +1)))[0]
        indices = np.where((each_step_val_coverage > ((1 - val_significance) * 100)))[0]
        indices=indices[indices>tolerance]
        if indices.size > 0:
            best_step=indices[0]+1
            val_coverage= each_step_val_coverage[indices[0]]
            print("{}-th step's validation tight coverage is {}".format(best_step, val_coverage))
        else:
            raise ValueError(
                "does not find a good step to make the coverage higher than {}".format(1 - val_significance))
    

        #import pdb; pdb.set_trace()
        return best_step

    def score_batch(self, emb,edge_index,pos_cal,y_cal):
        self.model.model.encoder=self.model.model.encoder.to(self.model.device)
        self.model.model.g=self.model.model.g.to(self.model.device)
        self.model.model.encoder.eval()
        self.model.model.g.eval()
        emb=emb.to(self.model.device)
        edge_index=edge_index.to(self.model.device)
        z_pred = self.model.model.encoder(emb,edge_index)
        #import pdb;pdb.set_trace()
        if self.inv_step is None:
            self.inv_step = self.find_best_step_num_batch(z_pred,pos_cal,y_cal)

        print('calculating score:')
        ret_val = []
        #import pdb;pdb.set_trace()
        z1=z_pred[pos_cal[:,0]]
        z2=z_pred[pos_cal[:,1]]
        for perm1,perm2,b_y in zip(DataLoader(range(z1.shape[0]), (64*1024), shuffle=False),DataLoader(range(z2.shape[0]), (64*1024), shuffle=False),DataLoader(range(len(y_cal)), (64*1024), shuffle=False)):
            
            #pos_edge = pos_cal[perm]
            #
            batch_y=y_cal[b_y].to(self.model.device)
            z11=z1[perm1]
            z12=z2[perm2]
            #z_pred = self.model.model.encoder(emb,edge_index)
            #import pdb; pdb.set_trace()
            z_pred=torch.cat((z11.unsqueeze(1), z12.unsqueeze(1)), dim=1)
            z_true = self.inv_g(z_pred, batch_y, step=self.inv_step)
            px = self.err_func.apply(z11.detach().cpu(), z_true[:,0,:].detach().cpu().numpy())
            py = self.err_func.apply(z12.detach().cpu(), z_true[:,1,:].detach().cpu().numpy())
            batch_ret_val =np.minimum(px,py)
            #pred=torch.cat([z11.detach().cpu(),z12.detach().cpu()],dim=1)
            #true=torch.cat([z_true[:,0,:].detach().cpu(),z_true[:,1,:].detach().cpu()],dim=1) 
            #batch_ret_val=self.err_func.apply(pred,true).numpy()
            ret_val.append(batch_ret_val)
        ret_val = np.concatenate(ret_val, axis=0)
        #import pdb;pdb.set_trace()
        return ret_val
    

    def predict(self,z,pos_test_edge,nc,significance=None):
        self.model.model.eval()
        certification_method = 1
        intervals = np.zeros((pos_test_edge.shape[0],1,2))
        cmethod = ['IBP', 'IBP+backward', 'backward', 'CROWN-Optimized'][certification_method]
        cmethod='CROWN'
        feat_err_dist = self.err_func.apply_inverse(nc, significance)
        lirpa_model = BoundedModule(self.model.model.g, (torch.empty((len(pos_test_edge),64)),torch.empty((len(pos_test_edge),64))))
        #lirpa_model=lirpa_model.to('cuda:1')
        ptb = PerturbationLpNorm(norm=np.inf, eps=feat_err_dist[0][0])#0.00052861124 0.007694608 feat_err_dist=[[0.122, 0.122]] (math.sqrt(0.7905606)) 0.034484193
        my_input1 = BoundedTensor(z[pos_test_edge.t()[0]], ptb).to('cuda:0')
        my_input2 = BoundedTensor(z[pos_test_edge.t()[1]], ptb).to('cuda:0')
        #print(lirpa_model._modules)
        lirpa_model.eval()
        #import pdb;pdb.set_trace()
        lb, ub = lirpa_model.compute_bounds(x=(my_input1,my_input2), method=cmethod)
        lb, ub = lb.detach().cpu().numpy(), ub.detach().cpu().numpy()
        intervals[..., 0] = lb
        intervals[..., 1] = ub
        #import pdb;pdb.set_trace()
        return intervals

class BaseIcp(BaseEstimator):
    def __init__(self, nc_function, condition=None):
        self.cal_x, self.cal_y = None, None
        self.nc_function = nc_function

        default_condition = lambda x: 0
        is_default = (callable(condition) and
                      (condition.__code__.co_code ==
                       default_condition.__code__.co_code))

        if is_default:
            self.condition = condition
            self.conditional = False
        elif callable(condition):
            self.condition = condition
            self.conditional = True
        else:
            self.condition = lambda x: 0
            self.conditional = False

    @classmethod
    def get_problem_type(cls):
        return 'regression'

    def fit(self, pos_train, train_y, pos_test,test_y):
        self.nc_function.fit(pos_train, train_y, pos_test,test_y)

    def calibrate(self, x, y,emb,edge_index, increment=False):
        # Change here
        self._calibrate_hook(x, y, increment)
        self._update_calibration_set(x, y, increment)

        if self.conditional:
            
            category_map = np.array([self.condition((x[i, :], y[i])) for i in range(y.size)])
            self.categories = np.unique(category_map)
            self.cal_scores = defaultdict(partial(np.ndarray, 0))

            for cond in self.categories:
                idx = category_map == cond
                cal_scores = self.nc_function.score(self.cal_x[idx, :], self.cal_y[idx])
                self.cal_scores[cond] = np.sort(cal_scores, 0)[::-1]
        else:
            
            self.categories = np.array([0])
            cal_scores = self.nc_function.score(self.cal_pos, self.cal_neg,emb,edge_index)
            self.cal_scores = {0: np.sort(cal_scores, 0)[::-1]}

    def calibrate_batch(self,emb,edge_index,pos_cal_edge,y_cal):
        if self.conditional:
            raise NotImplementedError

        else:
            self.categories = np.array([0])
            cal_scores = self.nc_function.score_batch(emb,edge_index,pos_cal_edge,y_cal)
            self.cal_scores = {0: np.sort(cal_scores, 0)[::-1]}
            #import pdb; pdb.set_trace()

    def _calibrate_hook(self, x, y, increment):
        pass

    def _update_calibration_set(self, x, y, increment):
        if increment and self.cal_x is not None and self.cal_y is not None:
            self.cal_x = np.vstack([self.cal_x, x])
            self.cal_y = np.hstack([self.cal_y, y])
        else:
            self.cal_pos, self.cal_neg = x, y

class IcpRegressor(BaseIcp):
    def __init__(self, nc_function, condition=None):
        super(IcpRegressor, self).__init__(nc_function, condition)

    def predict(self,emb,edge_index,pos_test_edge,significance=None):

        n_significance = (99 if significance is None
                          else np.array(significance).size)

        #import pdb;pdb.set_trace()
        z=self.nc_function.model.model.encoder(emb,edge_index)
        #import pdb;pdb.set_trace()
        qhat=np.quantile(self.cal_scores[0],0.9)
        diff=np.absolute(qhat-self.cal_scores[0])
        print('QHAT')
        print(np.mean(diff))
        prediction=self.nc_function.predict(z,pos_test_edge,self.cal_scores[0],significance) #prediction=
                

        return prediction

    def if_in_coverage(self, x, y, significance):
        self.nc_function.model.model.eval()
        condition_map = np.array([self.condition((x[i, :], None))
                                  for i in range(x.shape[0])])
        result_array = np.zeros(len(x)).astype(bool)
        for condition in self.categories:
            idx = condition_map == condition
            if np.sum(idx) > 0:
                err_dist = self.nc_function.score(x[idx, :], y[idx])
                err_dist_threshold = self.nc_function.err_func.apply_inverse(self.cal_scores[condition], significance)[0][0]
                result_array[idx] = (err_dist < err_dist_threshold)
        return result_array

    def if_in_coverage_batch(self, dataloader, significance):
        self.nc_function.model.model.eval()
        err_dist = self.nc_function.score_batch(dataloader)
        err_dist_threshold = self.nc_function.err_func.apply_inverse(self.cal_scores[0], significance)[0][0]
        result_array = (err_dist < err_dist_threshold)
        return result_array
