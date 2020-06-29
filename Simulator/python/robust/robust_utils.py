import numpy as np
import numba
@numba.njit(cache=True)
def sum_square(x):
 return np.sum(np.square(x))

@numba.njit(cache=True)
def project_onto_simplex(v,B):
 u=v[np.argsort(-v)]
 sv=np.cumsum(u)
 for i in range(v.shape[0]):
  cond=u[i]>(sv[i]-B)/(i+1)
  if cond:
   rho=i+1
 theta=(sv[rho-1]-B)/rho
 x=v-theta
 for i in range(x.shape[0]):
  if x[i]<0:
   x[i]=0
 return x

@numba.njit(cache=True)
def linear_chi_square(v,u,rho,epsilon=1e-8):
 lam_l=0
 lam_u=np.inf
 B=1.
 x=project_onto_simplex(u,1.)
 assert(sum_square(x-u)<=rho),'Infeasible'
 lam_s=1.
 while lam_u==np.inf:
  x=project_onto_simplex(u-v/lam_s,B)
  grad=0.5*sum_square(x-u)-rho/2.
  if grad<0:
   lam_u=lam_s
  else:
   lam_s*=2
 while(lam_u-lam_l)>epsilon*lam_s:
  lam=(lam_u+lam_l)/2.
  x=project_onto_simplex(u-v/lam,B)
  grad=0.5*sum_square(x-u)-rho/2.
  if grad<0:
   lam_u=lam
  else:
   lam_l=lam
 lam=(lam_u+lam_l)/2.
 return project_onto_simplex(u-v/lam,B)
 
if __name__=='__main__':
 B=1.
 temp=np.array([0.3,0.7,0.1,0.2,0.6])
 rho=0.1
 v=np.array([0.1,0.1,0.1,0.1,0.1])
 print(project_onto_simplex(v,B))
 print(linear_chi_square(temp,v/np.sum(v),rho))
 print('')
 v=np.array([0.1,0.1,0.1,0.1,10])
 print(project_onto_simplex(v,B))
 print(linear_chi_square(temp,v/np.sum(v),rho))
 print('')
 v=np.array([0.1,0.1,0.1,0.1,0.2])
 print(project_onto_simplex(v,B))
 print(linear_chi_square(temp,v/np.sum(v),rho))
 print('')
 v=np.array([0.6,0.1,0.1,0.1,0.1])
 print(project_onto_simplex(v,B))
 print(linear_chi_square(temp,v/np.sum(v),rho))
 print('')
 v=np.array([0.3,0.7,0.1,0.2,0.6])
 print(project_onto_simplex(v,B))
 print(linear_chi_square(temp,v/np.sum(v),rho))
 print('')
 v=np.array([0.06,0.1,0.1,0.1,0.06])
 print(project_onto_simplex(v,B))
 print(linear_chi_square(temp,v/np.sum(v),rho))
 print('')
 temp=np.random.rand(100,)
 v=np.random.rand(100,)
 v/=np.sum(v)
 import time
 start=time.time()
 NUM=100
 for _ in range(NUM):
  out=linear_chi_square(temp,v,rho)
 print((time.time()-start)/NUM)
# Created by pyminifier (https://github.com/liftoff/pyminifier)
