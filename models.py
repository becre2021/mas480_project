import torch 
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


from attrdict import AttrDict
import numpy as np
import random

import matplotlib.pyplot as plt



class MLP_Layer(nn.Module):
    #def __init__(self,in_dim=1,out_dim=1,hdim = 50 ):
    def __init__(self,in_dim=1,out_dim=1,hdim = 20 ):
        
        super(MLP_Layer,self).__init__()

        #self.in_dim = in_dim + 1
        self.in_dim = in_dim + 1        
        self.out_dim = out_dim
        self.hdim = hdim

        linear1 = nn.Linear(self.in_dim,self.hdim )
        linear2 = nn.Linear(self.hdim,self.hdim)
        linear3 = nn.Linear(self.hdim,self.hdim)
        linear4 = nn.Linear(self.hdim,self.out_dim)

        activation =  nn.Tanh()
        linear = [linear1,
                  activation,
                  linear2,
                  activation,
                  linear3,
                  activation,
                  linear4]
        self.mapping = nn.Sequential(*linear)
        
        def init_weights(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform(module.weight)
                module.bias.data.fill_(0.0)
                
        self.mapping.apply(init_weights)
        
                
    def forward(self,x):        
        return self.mapping(x)
    
    
    
    


class PINN_Basic(nn.Module):
    def __init__(self,nn_model, pdetype = 'v1' ):
        super(PINN_Basic,self).__init__()

        self.pde_type = pdetype
        self.nn_model = nn_model
        self.in_dim =  nn_model.in_dim         
        self.out_dim = nn_model.out_dim
        self.reg_lambda = 1
        
        #self.mse_loss = nn.MSELoss(reduce=False)
        self.mse_loss = nn.MSELoss(reduce=True,reduction = 'mean')
            
    def forward(self,xt):        
        u_xt = self.nn_model(xt)                
        return u_xt
    
    
    def compute_loss(self,xt_rd,y_rd,xt_init,u_init,xt_bd,u_bd):
        loss_rd = self.compute_pdeloss_rd(xt_rd,y_rd)
        loss_init = self.compute_pdeloss_bdinit(xt_init,u_init)
        loss_bd = self.compute_pdeloss_bdinit(xt_bd,u_bd)        
        #loss = loss_rd + self.reg_lambda*(0.5*loss_init + 0.5*loss_bd)
        #return loss_rd + self.reg_lambda*(0.5*loss_init + 0.5*loss_bd)
        return loss_rd,loss_init,loss_bd


    
    def compute_pdeloss_rd(self,xt,y):        
        xt_grad = xt.clone()
        xt_grad.requires_grad = True

        u = self.nn_model(xt_grad)                
        du_dxt= autograd.grad(u,xt_grad,torch.ones_like(u).to(u.device),retain_graph=True,create_graph=True)[0]
        du_dx_dxt = autograd.grad(du_dxt[:,:,:-1],xt_grad,torch.ones_like(du_dxt[:,:,:-1]).to(xt_grad.device),retain_graph=True,create_graph=True)[0]
        du_dx = du_dxt[:,:,:-1]
        du_dxx = du_dx_dxt[:,:,:-1]

        # compute loss depending on pde
        pde_loss = self.mse_loss(0.01*du_dxx,y)
        
        return pde_loss
    
    
    def compute_pdeloss_bdinit(self,xt,y):        
        u = self.nn_model(xt)                
        # compute loss depending on pde
        pde_loss = self.mse_loss(u,y)
        
        return pde_loss
        
    
    
#--------------------------------------------------------
# SWAG
# reference:
# https://github.com/wjmaddox/swa_gaussian/blob/ed5fd56e34083b42630239e59076952dee44daf4/swag/posteriors/swag.py#L237
# https://github.com/wjmaddox/swa_gaussian/blob/master/experiments/train/train.py
#--------------------------------------------------------
    
    
def swag_parameters(module, params, no_cov_mat=True):
    for name in list(module._parameters.keys()):
        #print(name)
        if module._parameters[name] is None:
            continue
        data = module._parameters[name].data
        # turn off pop, which makes error
        #module._parameters.pop(name)
        module.register_buffer("%s_mean" % name, data.new(data.size()).zero_())
        module.register_buffer("%s_sq_mean" % name, data.new(data.size()).zero_())

        if no_cov_mat is False:
            module.register_buffer(
                "%s_cov_mat_sqrt" % name, data.new_empty((0, data.numel())).zero_()
            )

        params.append((module, name))
    return


class PINN_SWAG(PINN_Basic):
    
    def __init__(self,nn_model=MLP_Layer(in_dim = 1, out_dim=1) , pdetype = 'v1', max_num_models = 10 ,update_period = 50, eps=1e-16):
        super(PINN_SWAG,self).__init__(nn_model, pdetype)
        self.name = 'swag'

        # ----------------
        # swag
        # ----------------        
        self.register_buffer("n_models", torch.zeros([1], dtype=torch.long))     
        
        self.var_clamp = eps
        self.max_num_models = max_num_models
        self.no_cov_mat = False        
        self.params = list()
        self.nn_model.apply( lambda module: swag_parameters(module=module, params=self.params, no_cov_mat=self.no_cov_mat )  )        
            
            
    # %s_mean" % name
    # %s_sq_mean" % name
    # %s_cov_mat_sqrt_mean" % name
    
    def compute_preddist(self,xt,num_sample=10):        
        pred_list = []
        for _ in range(num_sample):
            self.sample()
            pred_list.append(self.forward(xt).unsqueeze(dim=-1))
        pred_list = torch.cat(pred_list,dim=-1)    
        emp_pmu = pred_list.mean(dim=-1) 
        emp_pstd = pred_list.std(dim=-1) 
        
        #return emp_pmu,emp_pstd,pred_list
        return emp_pmu,emp_pstd

    
    
    #def collect_model(self, base_model):
    def collect_model(self):        
        #for (module, name), base_param in zip(self.params, base_model.parameters()):
        for (module, name), base_param in zip(self.params, self.nn_model.parameters()):            
            mean = module.__getattr__("%s_mean" % name)
            sq_mean = module.__getattr__("%s_sq_mean" % name)

            # first moment
            mean = mean * self.n_models.item() / (
                self.n_models.item() + 1.0
            ) + base_param.data / (self.n_models.item() + 1.0)

            # second moment
            sq_mean = sq_mean * self.n_models.item() / (
                self.n_models.item() + 1.0
            ) + base_param.data ** 2 / (self.n_models.item() + 1.0)

            # square root of covariance matrix
            if self.no_cov_mat is False:
                cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)

                # block covariance matrices, store deviation from current mean
                dev = (base_param.data - mean).view(-1, 1)
                cov_mat_sqrt = torch.cat((cov_mat_sqrt, dev.view(-1, 1).t()), dim=0)

                # remove first column if we have stored too many models
                if (self.n_models.item() + 1) > self.max_num_models:
                    cov_mat_sqrt = cov_mat_sqrt[1:, :]
                module.__setattr__("%s_cov_mat_sqrt" % name, cov_mat_sqrt)

            module.__setattr__("%s_mean" % name, mean)
            module.__setattr__("%s_sq_mean" % name, sq_mean)
        self.n_models.add_(1)    
        return 
    
    
    def sample(self, scale=1.0, cov=False, seed=None, block=False, fullrank=True):
#         if seed is not None:
#             torch.manual_seed(seed)

#         if not block:
#             self.sample_fullrank(scale, cov, fullrank)
#         else:
#             self.sample_blockwise(scale, cov, fullrank)

        self.sample_blockwise(scale, cov, fullrank)
        return


    def sample_blockwise(self, scale, cov, fullrank):
        for module, name in self.params:
            mean = module.__getattr__("%s_mean" % name)

            sq_mean = module.__getattr__("%s_sq_mean" % name)
            eps = torch.randn_like(mean)

            var = torch.clamp(sq_mean - mean ** 2, self.var_clamp)

            scaled_diag_sample = scale * torch.sqrt(var) * eps

            if cov is True:
                cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)
                eps = cov_mat_sqrt.new_empty((cov_mat_sqrt.size(0), 1)).normal_()
                cov_sample = (
                    scale / ((self.max_num_models - 1) ** 0.5)
                ) * cov_mat_sqrt.t().matmul(eps).view_as(mean)

                if fullrank:
                    w = mean + scaled_diag_sample + cov_sample
                else:
                    w = mean + scaled_diag_sample

            else:
                w = mean + scaled_diag_sample

    
            #---------------------------------------
            # debuggin checked, 
            # when updating the samples of parameters, it update copied module and original module (nn base model) automatically
            # we do not need to copy the item manually
            # we sample -> predict where the sampled parameters are automatically reflected on the original module (nn base model)             
            #---------------------------------------        
            
            #module.__setattr__(name, w)            
            module.__getattr__(name).data = w 
        
        return     
    
    
    
    
    
    
    
    
    
#----------------------------------------------------------------------------
# Bayes by backpror with local repamerization trick
# reference:
# https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/src/Bayes_By_Backprop_Local_Reparametrization/model.py
#----------------------------------------------------------------------------


def KLD_cost(mu_p, sig_p, mu_q, sig_q):
    KLD = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    # https://arxiv.org/abs/1312.6114 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return KLD

class BayesLinear_localreparam(nn.Module):
    """Linear Layer where activations are sampled from a fully factorised normal which is given by aggregating
     the moments of each weight's normal distribution. The KL divergence is obtained in closed form. Only works
      with gaussian priors.
    """
    def __init__(self, n_in, n_out, prior_sig=0.1**2):
        super(BayesLinear_localreparam, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.prior_sig = prior_sig

        # Learnable parameters
        self.W_mu = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-0.1, 0.1))        
        torch.nn.init.xavier_uniform(self.W_mu.data)
        #self.W_mu = nn.Parameter(torch.Tensor(self.n_in, self.n_out).xavier_uniform_())        
        self.W_p = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-6, -5))

        self.b_mu = nn.Parameter(torch.Tensor(self.n_out).uniform_(-1e-8, 1e-8))
        self.b_p = nn.Parameter(torch.Tensor(self.n_out).uniform_(-6, -5))
        self.bias = True

    def forward(self, X, sample=False):
        #         print(self.training)

        if not self.training and not sample:  # This is just a placeholder function
            #print('check weight sampling x')            
            output = torch.mm(X, self.W_mu) + self.b_mu.expand(X.size()[0], self.n_out)
            return output, 0, 0

        else:
            #print('check weight sampling o')
            # calculate std
            std_w = 1e-6 + F.softplus(self.W_p, beta=1, threshold=20)
            std_b = 1e-6 + F.softplus(self.b_p, beta=1, threshold=20)

            act_W_mu = torch.mm(X, self.W_mu)  # self.W_mu + std_w * eps_W
            act_W_std = torch.sqrt(torch.mm(X.pow(2), std_w.pow(2)))

            # Tensor.new()  Constructs a new tensor of the same data type as self tensor.
            # the same random sample is used for every element in the minibatch output
            eps_W = Variable(self.W_mu.data.new(act_W_std.size()).normal_(mean=0, std=1))
            eps_b = Variable(self.b_mu.data.new(std_b.size()).normal_(mean=0, std=1))

            act_W_out = act_W_mu + act_W_std * eps_W  # (batch_size, n_output)
            act_b_out = self.b_mu + std_b * eps_b

            output = act_W_out + act_b_out.unsqueeze(0).expand(X.shape[0], -1)

            kld = KLD_cost(mu_p=0, sig_p=self.prior_sig, mu_q=self.W_mu, sig_q=std_w)  
            kld = kld + KLD_cost(mu_p=0, sig_p=0.1, mu_q=self.b_mu,sig_q=std_b)
            return output, kld, 0

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.n_in, self.n_out, self.bias is not None
        )        
    

class BayesMLP_Layer(nn.Module):
    #def __init__(self,in_dim=1,out_dim=1,hdim = 50 ):
    def __init__(self,in_dim=1,out_dim=1,hdim = 20 ):
        
        super(BayesMLP_Layer,self).__init__()

        #self.in_dim = in_dim + 1
        self.in_dim = in_dim + 1        
        self.out_dim = out_dim
        self.hdim = hdim

        self.act = nn.Tanh()
        self.b_linear1 = BayesLinear_localreparam( self.in_dim,self.hdim )
        self.b_linear2 = BayesLinear_localreparam( self.hdim,self.hdim )
        self.b_linear3 = BayesLinear_localreparam( self.hdim,self.hdim )
        self.b_linear4 = BayesLinear_localreparam( self.hdim,self.out_dim )
        
                
    def forward(self, x ,use_sample = True):                
        need_reshape = False
        tlqw ,tlpw = 0,0
        
        if x.dim() == 3:
            nb,ndata,ndim = x.shape
            x = x.reshape(-1,ndim)
            need_reshape=True
            
        #--------------------------
        #layer 1
        #--------------------------        
        x, lqw, lpw = self.b_linear1(x,use_sample)
        tlqw = tlqw + lqw
        tlpw = tlpw + lpw
        x = self.act(x)        
        #--------------------------
        #layer 2
        #--------------------------        
        x, lqw, lpw = self.b_linear2(x,use_sample)
        tlqw = tlqw + lqw
        tlpw = tlpw + lpw
        x = self.act(x)        
        #--------------------------
        #layer 3
        #--------------------------        
        x, lqw, lpw = self.b_linear3(x,use_sample)
        tlqw = tlqw + lqw
        tlpw = tlpw + lpw
        x = self.act(x)
        #--------------------------
        #layer 4
        #--------------------------        
        y, lqw, lpw = self.b_linear4(x,use_sample)
        tlqw = tlqw + lqw
        tlpw = tlpw + lpw
        
        if need_reshape:
            y = y.reshape(nb,ndata,self.out_dim)                
            
        return y, tlqw, tlpw
    
    


class PINN_BBB(PINN_Basic):    
    def __init__(self,nn_model=BayesMLP_Layer(in_dim = 1, out_dim=1),
                     pdetype = 'v1',
                     num_sample_intrain=5,
                     num_sample_intest=10,
                     eps=1e-16):
    #def __init__(self,in_dim = 1, out_dim = 1, pdetype = 'v1',num_sample_intrain=5, num_sample_intest=10, eps=1e-16):
        
        super(PINN_BBB,self).__init__(nn_model, pdetype)
        self.name = 'vi'

        #self.nn_model =  BayesMLP_Layer(in_dim = in_dim,out_dim = out_dim)
        self.num_sample_intrain = num_sample_intrain
        self.num_sample_intest = num_sample_intest

    def compute_preddist(self,xt,num_sample=10):        
        pred_list = []
        for _ in range(num_sample):
            pred_list.append(self.forward(xt)[0].unsqueeze(dim=-1))
        pred_list = torch.cat(pred_list,dim=-1)    
        emp_pmu = pred_list.mean(dim=-1) 
        emp_pstd = pred_list.std(dim=-1) 
        
        #return emp_pmu,emp_pstd,pred_list
        return emp_pmu,emp_pstd

    

    def forward(self,xt,use_sample=True):        
        u_xt, tlqw, tlpw = self.nn_model(xt,use_sample)                            
        return u_xt, tlqw, tlpw
    

    def compute_loss(self,xt_rd,y_rd,xt_init,u_init,xt_bd,u_bd):
        loss_rd,reg_rd = 0,0
        loss_init,reg_init = 0,0
        loss_bd,reg_bd = 0,0
        
        for i in range(self.num_sample_intrain):        
            i_loss_rd,i_reg_rd      = self.compute_pdeloss_rd(xt_rd,y_rd)            
            loss_rd = loss_rd + i_loss_rd
            reg_rd = reg_rd + i_reg_rd
            
            i_loss_init,i_reg_init  = self.compute_pdeloss_bdinit(xt_init,u_init)
            loss_init = loss_init + i_loss_init
            reg_init = reg_init + i_reg_init

            i_loss_bd,i_reg_bd      = self.compute_pdeloss_bdinit(xt_bd,u_bd)        
            loss_bd = loss_bd + i_loss_bd
            reg_bd = reg_bd + i_reg_bd
            
        loss_rd, reg_rd = loss_rd/self.num_sample_intrain, reg_rd/self.num_sample_intrain    
        loss_init, reg_init = loss_init/self.num_sample_intrain, reg_init/self.num_sample_intrain    
        loss_bd, reg_bd = loss_bd/self.num_sample_intrain, reg_bd/self.num_sample_intrain    
        
        return (loss_rd,reg_rd),(loss_init,reg_init),(loss_bd,reg_bd)

    
    def compute_pdeloss_rd(self,xt,y):        
        xt_grad = xt.clone()
        xt_grad.requires_grad = True

        u, tlqw, tlpw = self.nn_model(xt_grad)                
        # ------------------------------        
        # compute regloss by prior
        # ------------------------------        
        nbatch = np.prod(xt.shape[:2])
        reg_loss = (tlqw - tlpw)/nbatch

        # ------------------------------
        # compute pde bd loss
        # ------------------------------        
        du_dxt= autograd.grad(u,xt_grad,torch.ones_like(u).to(u.device),retain_graph=True,create_graph=True)[0]
        du_dx_dxt = autograd.grad(du_dxt[:,:,:-1],xt_grad,torch.ones_like(du_dxt[:,:,:-1]).to(xt_grad.device),retain_graph=True,create_graph=True)[0]
        du_dx = du_dxt[:,:,:-1]
        du_dxx = du_dx_dxt[:,:,:-1]

        # compute loss depending on pde
        pde_loss = self.mse_loss(0.01*du_dxx,y)
            
        return pde_loss,reg_loss
    
    
    def compute_pdeloss_bdinit(self,xt,y):        
        u, tlqw, tlpw = self.nn_model(xt)
        # ------------------------------
        # compute regloss by prior        
        # ------------------------------        
        nbatch = np.prod(xt.shape[:2])
        reg_loss = (tlqw - tlpw)/nbatch
        
        # ------------------------------
        # compute pde init loss
        # ------------------------------                
        # compute loss depending on pde
        pde_loss = self.mse_loss(u,y)
        
        return pde_loss,reg_loss    

    
    
    
    
    
    
    
#----------------------------------------------------------------------------
# MC-dropout 
# reference:
# https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/src/Bayes_By_Backprop_Local_Reparametrization/model.py
#----------------   
    
    
    
def MC_dropout(act_vec, p=0.5, mask=True):
    return F.dropout(act_vec, p=p, training=mask, inplace=True)


class MLP_Layer_MCdrop(nn.Module):
    def __init__(self,in_dim=1,out_dim=1,hdim = 20 ,p =0.01 ):
        super(MLP_Layer_MCdrop,self).__init__()

        #self.in_dim = in_dim + 1
        self.in_dim = in_dim + 1        
        self.out_dim = out_dim
        self.hdim = hdim

        self.p = p
        linear1 = nn.Linear(self.in_dim,self.hdim)
        dropout1 = nn.Dropout(self.p)
        linear2 = nn.Linear(self.hdim,self.hdim)
        dropout2 = nn.Dropout(self.p)        
        linear3 = nn.Linear(self.hdim,self.hdim)
        dropout3 = nn.Dropout(self.p)        
        linear4 = nn.Linear(self.hdim,self.out_dim)

        activation =  nn.Tanh()
        linear = [linear1,
                  activation,
                  dropout1,                  
                  linear2,
                  activation,
                  dropout2,                                    
                  linear3,
                  activation,
                  dropout3,                                                      
                  linear4]
        self.linear = nn.Sequential(*linear)
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.0)
                
        self.linear.apply(init_weights)
        
                
    def forward(self,x):        
        return self.linear(x)
    
    
    
    
#--------------------------------------------------------
# reference:
# https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/src/MC_dropout/model.py
# https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/notebooks/regression/mc_dropout_homo.ipynb
#--------------------------------------------------------
class PINN_MCdrop(PINN_Basic):
    
    def __init__(self,nn_model= MLP_Layer_MCdrop(in_dim=1,out_dim=1,p=0.01), 
                      pdetype = 'v1',
                      num_sample_intrain=5, 
                      num_sample_intest=10,
                      eps=1e-16):
    #def __init__(self,in_dim = 1, out_dim = 1, pdetype = 'v1',num_sample_intrain=5, num_sample_intest=10, eps=1e-16):
        
        
        super(PINN_MCdrop,self).__init__(nn_model, pdetype)
        self.name = 'mcdrop'

        #self.nn_model =  BayesMLP_Layer(in_dim = in_dim,out_dim = out_dim)
        self.num_sample_intrain = num_sample_intrain
        self.num_sample_intest = num_sample_intest

    def compute_preddist(self,xt,num_sample=10):        
        pred_list = []
        for _ in range(num_sample):
            pred_list.append(self.forward(xt)[0].unsqueeze(dim=-1))
        pred_list = torch.cat(pred_list,dim=-1)    
        emp_pmu = pred_list.mean(dim=-1) 
        emp_pstd = pred_list.std(dim=-1) 
        
        #return emp_pmu,emp_pstd,pred_list
        return emp_pmu,emp_pstd

    

    def forward(self,xt,use_sample=True):        
        if use_sample:
            self.nn_model.train()
        else:
            self.nn_model.eval()            
        u_xt = self.nn_model(xt)                            
        return u_xt, 0.0
    

    def compute_loss(self,xt_rd,y_rd,xt_init,u_init,xt_bd,u_bd):
        loss_rd,reg_rd = 0,0
        loss_init,reg_init = 0,0
        loss_bd,reg_bd = 0,0
        
        for i in range(self.num_sample_intrain):        
            i_loss_rd,i_reg_rd      = self.compute_pdeloss_rd(xt_rd,y_rd)            
            loss_rd = loss_rd + i_loss_rd
            reg_rd = reg_rd + i_reg_rd
            
            i_loss_init,i_reg_init  = self.compute_pdeloss_bdinit(xt_init,u_init)
            loss_init = loss_init + i_loss_init
            reg_init = reg_init + i_reg_init

            i_loss_bd,i_reg_bd      = self.compute_pdeloss_bdinit(xt_bd,u_bd)        
            loss_bd = loss_bd + i_loss_bd
            reg_bd = reg_bd + i_reg_bd
            
        loss_rd, reg_rd = loss_rd/self.num_sample_intrain, reg_rd/self.num_sample_intrain    
        loss_init, reg_init = loss_init/self.num_sample_intrain, reg_init/self.num_sample_intrain    
        loss_bd, reg_bd = loss_bd/self.num_sample_intrain, reg_bd/self.num_sample_intrain    
        
        return (loss_rd,reg_rd),(loss_init,reg_init),(loss_bd,reg_bd)

    
    def compute_pdeloss_rd(self,xt,y):        
        xt_grad = xt.clone()
        xt_grad.requires_grad = True

#         u, tlqw, tlpw = self.nn_model(xt_grad)                
#         # ------------------------------        
#         # compute regloss by prior
#         # ------------------------------        
#         nbatch = np.prod(xt_bd.shape[:2])
#         reg_loss = (tlqw - tlpw)/nbatch

        u = self.nn_model(xt_grad)                

        # ------------------------------
        # compute pde bd loss
        # ------------------------------        
        du_dxt= autograd.grad(u,xt_grad,torch.ones_like(u).to(u.device),retain_graph=True,create_graph=True)[0]
        du_dx_dxt = autograd.grad(du_dxt[:,:,:-1],xt_grad,torch.ones_like(du_dxt[:,:,:-1]).to(xt_grad.device),retain_graph=True,create_graph=True)[0]
        du_dx = du_dxt[:,:,:-1]
        du_dxx = du_dx_dxt[:,:,:-1]

        # compute loss depending on pde
        pde_loss = self.mse_loss(0.01*du_dxx,y)
            
        return pde_loss,0.0
    
    
    def compute_pdeloss_bdinit(self,xt,y):        
#         u, tlqw, tlpw = self.nn_model(xt)
#         # ------------------------------
#         # compute regloss by prior        
#         # ------------------------------        
#         nbatch = np.prod(xt_bd.shape[:2])
#         reg_loss = (tlqw - tlpw)/nbatch

        u = self.nn_model(xt)

        # ------------------------------
        # compute pde init loss
        # ------------------------------                
        # compute loss depending on pde
        pde_loss = self.mse_loss(u,y)
        
        return pde_loss,0.0    
    