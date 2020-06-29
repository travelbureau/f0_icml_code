import torch
import torch.nn as nn
import torch.distributions as D
from maf import create_masks,FlowSequential

class GroupMaskedLinear(nn.Module):
 def __init__(self,masks,weights,biases,cond_weights=None):
  super().__init__()
  self.weights=weights*masks 
  if cond_weights is not None:
   self.cond_weights=cond_weights 
  self.biases=biases 

 def forward(self,gx,gy=None):
  out= torch.einsum('gbi,gji->gbj',gx,self.weights)+self.biases
  if gy is not None:
   out+=torch.einsum('gbi,gji->gbj',gy,self.cond_weights)
  return out

class GroupMADE(nn.Module):
 def __init__(self,DEVICE,made_models,input_size,hidden_size,n_hidden,activation='relu',input_order='sequential',input_degrees=None):
  super().__init__()
  self.register_buffer('base_dist_mean',torch.zeros(input_size))
  self.register_buffer('base_dist_var',torch.ones(input_size))
  masks,self.input_degrees=create_masks(input_size,hidden_size,n_hidden,input_order,input_degrees)
  if activation=='relu':
   activation_fn=nn.ReLU()
  else:
   raise ValueError('Check activation function.')
  num_guys=len(made_models)
  weights=torch.cat([made.net_input.weight[None,:,:]for made in made_models],dim=0).to(DEVICE)
  biases=torch.cat([made.net_input.bias[None,None,:]for made in made_models],dim=0).to(DEVICE)
  cond_weights=torch.cat([made.net_input.cond_weight[None,:,:]for made in made_models],dim=0).to(DEVICE)
  self.net_input=GroupMaskedLinear((masks[0][None,:,:]).repeat(num_guys,1,1).to(DEVICE),weights,biases,cond_weights)
  self.net=[]
  for i in range(1,len(masks)-1):
   m=masks[i]
   weights=torch.cat([made.net[i*2-1].weight[None,:,:]for made in made_models],dim=0).to(DEVICE)
   biases=torch.cat([made.net[i*2-1].bias[None,None,:]for made in made_models],dim=0).to(DEVICE)
   self.net+=[activation_fn,GroupMaskedLinear((m[None,:,:]).repeat(num_guys,1,1).to(DEVICE),weights,biases)]
  m=masks[-1].repeat(2,1).to(DEVICE)
  weights=torch.cat([made.net[-1].weight[None,:,:]for made in made_models],dim=0).to(DEVICE)
  biases=torch.cat([made.net[-1].bias[None,None,:]for made in made_models],dim=0).to(DEVICE)
  self.net+=[activation_fn,GroupMaskedLinear((m[None,:,:]).repeat(num_guys,1,1),weights,biases)]
  self.net=nn.Sequential(*self.net)

 @property
 def base_dist(self):
  return D.Normal(self.base_dist_mean,self.base_dist_var)

 def forward(self,gx,gy=None):
  m,loga=self.net(self.net_input(gx,gy)).chunk(chunks=2,dim=2)
  u=(gx-m)*torch.exp(-loga)
  log_abs_det_jacobian=-loga
  return u,log_abs_det_jacobian

 def inverse(self,gu,gy=None,sum_log_abs_det_jacobians=None):
  gx=torch.zeros_like(gu)
  for i in self.input_degrees:
   m,loga=self.net(self.net_input(gx,gy)).chunk(chunks=2,dim=2)
   gx[:,:,i]=gu[:,:,i]*torch.exp(loga[:,:,i])+m[:,:,i]
  log_abs_det_jacobian=loga
  return gx,log_abs_det_jacobian

 def log_prob(self,gx,gy=None):
  gu,log_abs_det_jacobian=self.forward(gx,gy)
  return torch.sum(self.base_dist.log_prob(gu)+log_abs_det_jacobian,dim=2)

class GroupMAF(nn.Module):
 def __init__(self,device,maf_models,input_size,hidden_size,n_hidden,activation='relu',input_order='sequential'):
  super().__init__()
  self.register_buffer('base_dist_mean',torch.zeros(input_size))
  self.register_buffer('base_dist_var',torch.ones(input_size))
  self.group_length=len(maf_models)
  modules=[]
  self.input_degrees=None
  for i in range(len(maf_models[0].net)):
   made_models=[maf.net[i]for maf in maf_models]
   modules+=[GroupMADE(device,made_models,input_size,hidden_size,n_hidden,activation,input_order,self.input_degrees)]
   self.input_degrees=modules[-1].input_degrees.flip(0)
  self.net=FlowSequential(*modules)

 @property
 def base_dist(self):
  return D.Normal(self.base_dist_mean,self.base_dist_var)

 def forward(self,gx,gy=None):
  return self.net(gx,gy)

 def inverse(self,gu,gy=None):
  return self.net.inverse(gu,gy)

 def log_prob(self,gx,gy=None):
  gu,sum_log_abs_det_jacobians=self.forward(gx,gy)
  return torch.sum(self.base_dist.log_prob(gu)+sum_log_abs_det_jacobians,dim=2)

class GroupIAF(nn.Module):
 def __init__(self,device,iaf_models,input_size,hidden_size,n_hidden,activation='relu',input_order='sequential'):
  super().__init__()
  self.register_buffer('base_dist_mean',torch.zeros(input_size))
  self.register_buffer('base_dist_var',torch.ones(input_size))
  self.group_length=len(iaf_models)
  modules=[]
  self.input_degrees=None
  for i in range(len(iaf_models[0].net)):
   made_models=[iaf.net[i]for iaf in iaf_models]
   modules+=[GroupMADE(device,made_models,input_size,hidden_size,n_hidden,activation,input_order,self.input_degrees)]
   self.input_degrees=modules[-1].input_degrees.flip(0)
  self.net=FlowSequential(*modules)

 @property
 def base_dist(self):
  return D.Normal(self.base_dist_mean,self.base_dist_var)

 def forward(self,gx,gy=None):
  return self.net(gx,gy)

 def sample(self,gu,gy=None):
  return self.forward(gu,gy)
  
 def log_prob(self,gx,gy=None):
  gu,sum_log_abs_det_jacobians=self.net.inverse(gx,gy)
  return torch.sum(self.base_dist.log_prob(gu)+sum_log_abs_det_jacobians,dim=2)
# Created by pyminifier (https://github.com/liftoff/pyminifier)
