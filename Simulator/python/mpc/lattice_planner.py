from mpc import trajectory_generator_utils,pure_pursuit_utils
import numpy as np
from PIL import Image
from numpy import linalg as LA
import math
import os,sys
import yaml
import msgpack
from numba import njit
from numba.typed import List,Dict
import time

from mpc import trajectory_generator
BRAKE_STEPS=60
assert(BRAKE_STEPS<=trajectory_generator.NUM_STEPS)

@njit(fastmath=False,cache=True)
def corner(current_s,WINDOW_SIZE,waypoints):
 picked_s=current_s+WINDOW_SIZE
 current_s-=WINDOW_SIZE
 current_idx=np.searchsorted(waypoints[:,4],current_s,side='right')-1
 picked_idx=np.searchsorted(waypoints[:,4],picked_s,side='right')-1
 if current_idx<0:
  current_idx=0
 if picked_idx<0:
  picked_idx=0
 if picked_idx<current_idx:
  curvature_sum=np.sum(waypoints[current_idx:,5])+np.sum(waypoints[0:picked_idx+1,5])
 else:
  curvature_sum=np.sum(waypoints[current_idx:picked_idx+1,5])
 curvature_sum=np.abs(curvature_sum)
 return curvature_sum

@njit(fastmath=False,cache=True)
def flow_to_xytheta_static(st,pose,wpts,lut_resolution,speed_lut,window_size,curvature_thresh,corner_on_b4):
 if np.any(np.isnan(st)):
  return None,None,None,corner_on_b4
 pose_query=(int(np.round(pose[0]/lut_resolution)),int(np.round(pose[1]/lut_resolution)))
 if pose_query in speed_lut:
  pose_val=speed_lut[pose_query]
 else:
  return None,None,None,corner_on_b4
 current_s=pose_val[4]
 corner_sum=corner(current_s,window_size,wpts)
 if corner_sum>=curvature_thresh:
  corner_on=True
  temp_s=st[:,0]*curvature_thresh/corner_sum
 else:
  corner_on=False
  temp_s=st[:,0]
 new_s=current_s+temp_s
 new_s[new_s>=wpts[-1,4]]-=wpts[-1,4]
 waypoint_idx=np.searchsorted(wpts[:,4],new_s,side='right')-1
 waypoint_idx[waypoint_idx<0]=0
 waypoint_idx[waypoint_idx>=wpts.shape[0]-1]=wpts.shape[0]-2
 l=(new_s-wpts[waypoint_idx,4])/(wpts[waypoint_idx+1,4]-wpts[waypoint_idx,4])
 new_x=wpts[waypoint_idx,0]+l*(wpts[waypoint_idx+1,0]-wpts[waypoint_idx,0])
 new_y=wpts[waypoint_idx,1]+l*(wpts[waypoint_idx+1,1]-wpts[waypoint_idx,1])
 pt_xy=np.stack((new_x,new_y),axis=1)
 angle=wpts[waypoint_idx,3]+np.pi/2
 angle_arr=np.stack((np.cos(angle),np.sin(angle)),axis=1)
 xy=np.expand_dims(st[:,1],axis=1)*angle_arr+pt_xy
 xy-=np.expand_dims(pose[:2],axis=0)
 rot=np.array([[np.cos(-pose[2]),np.sin(-pose[2])],[-np.sin(-pose[2]),np.cos(-pose[2])]])
 xy=np.dot(xy,rot)
 theta=wpts[waypoint_idx,3]+st[:,2]-pose[2]
 theta=np.mod(theta,2*np.pi)
 theta[theta>np.pi]-=(2*np.pi)
 temp=np.concatenate((xy,np.expand_dims(theta,axis=1)),axis=1)
 return temp,current_s,new_s,corner_on

class LatticePlanner(object):
 OPP_SPEED_SCALE=1.00 
 def __init__(self,map_path,cost_weights,waypoints,directory,is_ego):
  self.limp_s=0.
  self.is_limping=False
  self.is_ego=is_ego
  self.prev_traj=None
  self.prev_param=None
  self.prev_steer=0.
  self.cost_weights=cost_weights
  self.waypoints=waypoints
  self.wheelbase=0.3302
  self.max_reacquire=10
  self.safe_speed=2.5
  self.CORNER_ON=False
  self.track_lad=1.0
  self.STEER_LP=0.99
  self.CURVATURE_THRESH=20.
  self.WINDOW_SIZE=3.
  self.TOP_POP_NUM=3
  lut_all=np.load(directory+'mpc/lut_inuse.npz')
  self.lut_x=lut_all['x']
  self.lut_y=lut_all['y']
  self.lut_theta=lut_all['theta']
  self.lut_kappa=lut_all['kappa']
  self.lut=lut_all['lut']
  step_sizes=[]
  step_sizes.append(self.lut_x[1]-self.lut_x[0])
  step_sizes.append(self.lut_y[1]-self.lut_y[0])
  step_sizes.append(self.lut_theta[1]-self.lut_theta[0])
  step_sizes.append(self.lut_kappa[1]-self.lut_kappa[0])
  self.lut_stepsizes=np.array(step_sizes)
  with open(directory+'config.yaml','r')as yaml_stream:
   try:
    config=yaml.safe_load(yaml_stream)
    speed_lut_name=config['speed_lut_name']
    range_lut_name=config['range_lut_name']
   except yaml.YAMLError as ex:
    print(ex)
  speed_lut_temp=msgpack.unpack(open(directory+speed_lut_name,'rb'),use_list=False)
  self.speed_lut_numba=Dict()
  for key,val in speed_lut_temp.items():
   if key==b'resolution':
    continue
   self.speed_lut_numba[key]=val
  range_lut_temp=msgpack.unpack(open(directory+range_lut_name,'rb'),use_list=False)
  self.range_lut_numba=Dict()
  for key,val in range_lut_temp.items():
   if key==b'resolution':
    continue
   self.range_lut_numba[key]=val
  self.lut_resolution=float(speed_lut_temp[b'resolution'][0])

 def update_cost(self,cost_weights):
  self.prev_traj=None
  self.prev_param=None
  self.prev_steer=0
  self.cost_weights=cost_weights

 def _rotation_matrix(self,angle,direction,point=None):
  sina=math.sin(angle)
  cosa=math.cos(angle)
  direction=self._unit_vector(direction[:3])
  R=np.array(((cosa,0.0,0.0),(0.0,cosa,0.0),(0.0,0.0,cosa)),dtype=np.float64)
  R+=np.outer(direction,direction)*(1.0-cosa)
  direction*=sina
  R+=np.array(((0.0,-direction[2],direction[1]),(direction[2],0.0,-direction[0]),(-direction[1],direction[0],0.0)),dtype=np.float64)
  M=np.identity(4)
  M[:3,:3]=R
  if point is not None:
   point=np.array(point[:3],dtype=np.float64,copy=False)
   M[:3,3]=point-np.dot(R,point)
  return M

 def _unit_vector(self,data,axis=None,out=None):
  if out is None:
   data=np.array(data,dtype=np.float64,copy=True)
   if data.ndim==1:
    data/=math.sqrt(np.dot(data,data))
    return data
  else:
   if out is not data:
    out[:]=np.array(data,copy=False)
   data=out
  length=np.atleast_1d(np.sum(data*data,axis))
  np.sqrt(length,length)
  if axis is not None:
   length=np.expand_dims(length,axis)
  data/=length
  if out is None:
   return data

 def _get_current_waypoint(self,waypoints,lookahead_distance,position,theta):
  wpts=waypoints[:,0:2]
  nearest_point,nearest_dist,t,i=pure_pursuit_utils.nearest_point_on_trajectory_py2(position,wpts)
  if nearest_dist<lookahead_distance:
   lookahead_point,i2,t2=pure_pursuit_utils.first_point_on_trajectory_intersecting_circle(position,lookahead_distance,wpts,i+t,wrap=True)
   if i2==None:
    return None
   current_waypoint=np.empty(waypoints[i2,:].shape)
   current_waypoint[0:2]=waypoints[i2,0:2]
   current_waypoint[3]=waypoints[i2,3]
   current_waypoint[2]=waypoints[i2,2]
   return current_waypoint
  elif nearest_dist<self.max_reacquire:
   return waypoints[i,:]
  else:
   return None

 def _get_current_speed(self,position,theta):
  wpts=self.waypoints[:,0:2]
  position=position[0:2]
  nearest_point,nearest_dist,t,i=pure_pursuit_utils.nearest_point_on_trajectory_py2(position,wpts)
  speed=self.waypoints[i,2]
  return speed,nearest_dist

 def _pure_pursuit(self,pose_x,pose_y,pose_theta,trajectory,lookahead_distance):
  if trajectory is None:
   return self.safe_speed,0.0
  position=np.array([pose_x,pose_y])
  lookahead_point=self._get_current_waypoint(trajectory,lookahead_distance,position,pose_theta)
  if lookahead_point is None:
   return self.safe_speed,0.0
  speed,steering_angle=pure_pursuit_utils.get_actuation(pose_theta,lookahead_point,position,lookahead_distance,self.wheelbase)
  return speed,steering_angle

 def plan(self,pose,opp_pose,sampled_flow,other_prev_traj,other_prev_param,opp_collision,ds,current_vel):
  current_vel=max(0.01,current_vel)
  pose_x,pose_y,pose_theta=pose
  if self.is_ego:
   other_prev_traj[:,4]*=LatticePlanner.OPP_SPEED_SCALE
  lookup_grid,current_s,new_s,corner_on=flow_to_xytheta_static(sampled_flow[:,0:3],np.array(pose),self.waypoints,self.lut_resolution,self.speed_lut_numba,self.WINDOW_SIZE,self.CURVATURE_THRESH,self.CORNER_ON)
  self.CORNER_ON=corner_on
  if lookup_grid is None:
   safety_flag=False
   states_list_plot=None
   pp_traj,prev_traj_plot=self.getLimpParameters(pose_x,pose_y,current_vel)
   self.prev_flow=None
   return pp_traj,safety_flag,self.prev_flow,states_list_plot,prev_traj_plot,lookup_grid
  kappa0=trajectory_generator.get_curvature_command(np.roll(self.prev_param,-1),ds+self.limp_s)if self.prev_param is not None else 0.0
  rot=self._rotation_matrix(pose_theta,(0,0,1))
  states_list_local,parameters_list,filtered_flow,filtered_grid,filtered_new_s=trajectory_generator_utils.grid_lookup(lookup_grid,self.lut_x,self.lut_y,self.lut_theta,self.lut_kappa,self.lut,sampled_flow,new_s,kappa0,self.lut_stepsizes)
  num_traj=parameters_list.shape[0]
  if num_traj==0:
   safety_flag=False
   states_list_plot=None
   pp_traj,prev_traj_plot=self.getLimpParameters(pose_x,pose_y,current_vel)
   self.prev_flow=None
   return pp_traj,safety_flag,self.prev_flow,states_list_plot,prev_traj_plot,lookup_grid
  states_list=trajectory_generator_utils.trans_traj_list(states_list_local,np.array([[pose_x],[pose_y],[0.0]]),rot)
  dspeed=filtered_flow[:,3:]
  new_states_list=trajectory_generator_utils.get_velocity_profile(states_list,self.waypoints,dspeed,num_traj,current_vel)
  if other_prev_traj is None or other_prev_param is None:
   other_prev_param=np.array([5.,0,0,0,0])
   other_prev_traj=np.zeros((trajectory_generator.NUM_STEPS,5))
   temp=np.linspace(0.,5.,trajectory_generator.NUM_STEPS)
   other_prev_traj[:,0]=opp_pose[0]+temp*np.cos(opp_pose[2])
   other_prev_traj[:,1]=opp_pose[1]+temp*np.sin(opp_pose[2])
   other_prev_traj[:,4]=trajectory_generator_utils.WAYPOINT_SPEED
   self.prev_flow=None
  if self.prev_traj is None or self.prev_param is None:
   self.prev_param=np.array([5.,0,0,0,0])
   self.prev_traj=np.zeros((trajectory_generator.NUM_STEPS,5))
   temp=np.linspace(0.,5.,trajectory_generator.NUM_STEPS)
   self.prev_traj[:,0]=pose[0]+temp*np.cos(pose[2])
   self.prev_traj[:,1]=pose[1]+temp*np.sin(pose[2])
   self.prev_traj[:,2]=pose[2]
   self.prev_traj[:,4]+=trajectory_generator_utils.WAYPOINT_SPEED
   self.prev_flow=None
   self.prev_steer=0.0 
   prev_traj_plot=self.prev_traj
   pp_traj=np.empty((self.prev_traj.shape[0],4))
   pp_traj[:,0:2]=self.prev_traj[:,0:2]
   pp_traj[:,2]=self.prev_traj[:,4]
   pp_traj[:,3]=self.prev_traj[:,2]
   return pp_traj,False,self.prev_flow,None,prev_traj_plot,lookup_grid
  opp_relative_weights=np.array([1.])
  traj_costs,end_xy=trajectory_generator_utils.get_traj_list_cost(states_list,new_states_list,self.cost_weights,self.waypoints,self.prev_traj,parameters_list,other_prev_traj,np.array([other_prev_param]),opp_relative_weights,opp_collision)
  traj_costs[4,:]=trajectory_generator_utils.get_s_cost_wlut(states_list,num_traj,self.waypoints,self.speed_lut_numba,self.lut_resolution)
  traj_costs[9,:]=trajectory_generator_utils.get_range_costs(states_list,num_traj,self.range_lut_numba,self.lut_resolution)
  traj_costs[12,:]=trajectory_generator_utils.get_progress_costs(end_xy,opp_relative_weights,num_traj,self.speed_lut_numba,self.lut_resolution)
  traj_costs_final=trajectory_generator_utils.sum_cost(traj_costs,self.cost_weights)
  empty_cost_flag=False
  is_inf_flag=False
  safety_flag=False
  try:
   non_inf_idx=np.where(np.isfinite(traj_costs_final))[0]
   non_inf_costs=traj_costs_final[non_inf_idx]
   k=min(self.TOP_POP_NUM-1,non_inf_costs.shape[0])
   lowest_cost_idx_top=np.argpartition(non_inf_costs,k)[:k+1]
   lowest_cost_idx=np.argmin(non_inf_costs[lowest_cost_idx_top])
   lowest_cost_idx=lowest_cost_idx_top[lowest_cost_idx]
   lowest_cost_idx=non_inf_idx[lowest_cost_idx]
   best_traj=new_states_list[lowest_cost_idx*trajectory_generator.NUM_STEPS:(lowest_cost_idx+1)*trajectory_generator.NUM_STEPS,:]
   self.prev_traj=best_traj
   pp_traj=np.empty((best_traj.shape[0],4))
   pp_traj[:,0:2]=best_traj[:,0:2]
   pp_traj[:,2]=best_traj[:,4]
   pp_traj[:,3]=best_traj[:,2]
   self.prev_flow=(filtered_flow[non_inf_idx,:])[lowest_cost_idx_top,:]
   self.prev_param=parameters_list[lowest_cost_idx,:]
   self.is_limping=False
   self.limp_s=0.
  except ValueError:
   empty_cost_flag=True
   is_inf_flag=True
   self.prev_flow=None
  if empty_cost_flag or is_inf_flag:
   safety_flag=False
   states_list_plot=None
   pp_traj,prev_traj_plot=self.getLimpParameters(pose_x,pose_y,current_vel)
  else:
   states_list_plot=states_list
   prev_traj_plot=self.prev_traj
  return pp_traj,safety_flag,self.prev_flow,states_list_plot,prev_traj_plot,lookup_grid

 def getLimpParameters(self,pose_x,pose_y,current_vel):
  s_tot=np.linspace(0,self.prev_param[0],trajectory_generator.NUM_STEPS)
  s_tot-=s_tot[0]
  _,_,_,idx=pure_pursuit_utils.nearest_point_on_trajectory_py2(np.array([pose_x,pose_y]),self.prev_traj[:,:2])
  idx=min(idx,self.prev_traj.shape[0]-3)
  s_idx=s_tot[idx:]
  s_idx-=s_idx[0]
  temp_traj=np.empty((trajectory_generator.NUM_STEPS,5))
  temp_traj[:,:4]=trajectory_generator_utils.multiInterp2(np.linspace(0,s_idx[-1],trajectory_generator.NUM_STEPS),s_idx,self.prev_traj[idx:,:4])
  temp_traj[:,4]=np.concatenate([np.geomspace(current_vel,0.4,BRAKE_STEPS),0.4*np.ones((temp_traj.shape[0]-BRAKE_STEPS,))])
  self.prev_traj=temp_traj
  self.limp_s+=s_tot[idx]
  self.prev_param[0]=s_idx[-1]
  self.is_limping=True
  prev_traj_plot=self.prev_traj
  pp_traj=np.empty((self.prev_traj.shape[0],4))
  pp_traj[:,0:2]=self.prev_traj[:,0:2]
  pp_traj[:,2]=self.prev_traj[:,4]
  pp_traj[:,3]=self.prev_traj[:,2]
  return pp_traj,prev_traj_plot

 def compute_action(self,pp_traj,safety_flag,pose,off_policy=False):
  pose_x,pose_y,pose_theta=pose
  if safety_flag:
   next_speed,next_steer=self._pure_pursuit(pose_x,pose_y,pose_theta,pp_traj,self.track_lad)
   next_steer=self.STEER_LP*next_steer+(1-self.STEER_LP)*self.prev_steer
  elif off_policy:
   next_speed,next_steer=self._pure_pursuit(pose_x,pose_y,pose_theta,self.waypoints[:,0:4],self.track_lad)
   next_steer=self.STEER_LP*next_steer+(1-self.STEER_LP)*self.prev_steer
  else:
   next_speed,next_steer=self._pure_pursuit(pose_x,pose_y,pose_theta,pp_traj,self.track_lad)
  self.prev_steer=next_steer
  return next_speed,next_steer

def getLimpParametersStatic(pose_x,pose_y,current_vel,prev_traj_self,prev_param_self):
 prev_traj=np.copy(prev_traj_self)
 prev_param=np.copy(prev_param_self)
 s_tot=np.linspace(0,prev_param[0],trajectory_generator.NUM_STEPS)
 s_tot-=s_tot[0]
 _,_,_,idx=pure_pursuit_utils.nearest_point_on_trajectory_py2(np.array([pose_x,pose_y]),prev_traj[:,:2])
 idx=min(idx,prev_traj.shape[0]-3)
 s_idx=s_tot[idx:]
 s_idx-=s_idx[0]
 temp_traj=np.empty((trajectory_generator.NUM_STEPS,5))
 temp_traj[:,:4]=trajectory_generator_utils.multiInterp2(np.linspace(0,s_idx[-1],trajectory_generator.NUM_STEPS),s_idx,prev_traj[idx:,:4])
 temp_traj[:,4]=np.concatenate([np.geomspace(current_vel,0.4,BRAKE_STEPS),0.4*np.ones((temp_traj.shape[0]-BRAKE_STEPS,))])
 prev_traj=temp_traj
 extra_limp_s=s_tot[idx]
 prev_param[0]=s_idx[-1]
 pp_traj=np.empty((prev_traj.shape[0],4))
 pp_traj[:,0:2]=prev_traj[:,0:2]
 pp_traj[:,2]=prev_traj[:,4]
 pp_traj[:,3]=prev_traj[:,2]
 return pp_traj,prev_traj,prev_param,extra_limp_s

class RobustLatticePlanner(LatticePlanner):
 def __init__(self,map_path,waypoints,directory,cost_weights=None,is_ego=False):
  super().__init__(map_path,cost_weights,waypoints,directory,is_ego)
 
 def plan_multiple(self,pose,opp_pose,sampled_flow_list,other_prev_traj,other_prev_param,ds,current_vel,picked_cost_weights,picked_belief):
  current_vel=max(0.01,current_vel)
  pose_x,pose_y,pose_theta=pose
  num_guys=picked_belief.shape[0]
  output_traj_list=[None]*num_guys
  output_param_list=[None]*num_guys
  if other_prev_traj is None or other_prev_param is None:
   other_prev_param=np.array([5.,0,0,0,0])
   other_prev_traj=np.zeros((trajectory_generator.NUM_STEPS,5))
   temp=np.linspace(0.,5.,trajectory_generator.NUM_STEPS)
   other_prev_traj[:,0]=opp_pose[0]+temp*np.cos(opp_pose[2])
   other_prev_traj[:,1]=opp_pose[1]+temp*np.sin(opp_pose[2])
   other_prev_traj[:,4]=trajectory_generator_utils.WAYPOINT_SPEED
  if self.prev_traj is None or self.prev_param is None:
   self.prev_param=np.array([5.,0,0,0,0])
   self.prev_traj=np.zeros((trajectory_generator.NUM_STEPS,5))
   temp=np.linspace(0.,5.,trajectory_generator.NUM_STEPS)
   self.prev_traj[:,0]=pose[0]+temp*np.cos(pose[2])
   self.prev_traj[:,1]=pose[1]+temp*np.sin(pose[2])
   self.prev_traj[:,2]=pose[2]
   self.prev_traj[:,4]+=trajectory_generator_utils.WAYPOINT_SPEED
   self.prev_steer=0.0
   return[self.prev_traj]*num_guys,[self.prev_param]*num_guys
  sampled_flow_stacked=sampled_flow_list.reshape((sampled_flow_list.shape[0]*sampled_flow_list.shape[1],sampled_flow_list.shape[2]))
  lookup_grid,current_s,new_s,corner_on=flow_to_xytheta_static(sampled_flow_stacked[:,0:3],np.array(pose),self.waypoints,self.lut_resolution,self.speed_lut_numba,self.WINDOW_SIZE,self.CURVATURE_THRESH,self.CORNER_ON)
  self.CORNER_ON=corner_on
  if lookup_grid is None:
   pp_traj,prev_traj,prev_param,extra_limp_s=getLimpParametersStatic(pose_x,pose_y,current_vel,self.prev_traj,self.prev_param)
   self.prev_traj=prev_traj
   self.prev_param=prev_param
   self.limp_s+=extra_limp_s
   return[prev_traj]*num_guys,[self.prev_param]*num_guys
  kappa0=trajectory_generator.get_curvature_command(np.roll(self.prev_param,-1),ds+self.limp_s)if self.prev_param is not None else 0.0
  rot=self._rotation_matrix(pose_theta,(0,0,1))
  states_list_local,parameters_list,filtered_flow_list,num_traj_list=trajectory_generator_utils.grid_lookup_parallel(lookup_grid,self.lut_x,self.lut_y,self.lut_theta,self.lut_kappa,self.lut,kappa0,self.lut_stepsizes,num_guys,sampled_flow_list)
  for i in range(len(num_traj_list)):
   if num_traj_list[i]==0:
    pp_traj,prev_traj,prev_param,_=getLimpParametersStatic(pose_x,pose_y,current_vel,self.prev_traj,self.prev_param)
    output_traj_list[i]=prev_traj
    output_param_list[i]=prev_param
  states_list=trajectory_generator_utils.trans_traj_list_multiple(states_list_local,np.array([[pose_x],[pose_y],[0.0]]),rot)
  for i in range(len(num_traj_list)):
   if num_traj_list[i]==0:
    states_list[i]=output_traj_list[i]
    parameters_list[i]=output_param_list[i][None,:]
  dspeed_list=List()
  for filtered_flow in filtered_flow_list:
   dspeed_list.append(np.ascontiguousarray(filtered_flow[:,3:]))
  new_states_list=trajectory_generator_utils.get_velocity_profile_multiple(states_list,self.waypoints,dspeed_list,num_traj_list,current_vel)
  for i in range(len(num_traj_list)):
   if num_traj_list[i]==0:
    num_traj_list[i]=1
  opp_relative_weights=np.array([1.])
  traj_costs_list,end_xy_list=trajectory_generator_utils.get_traj_list_cost_multiple(states_list,new_states_list,picked_cost_weights,self.waypoints,self.prev_traj,parameters_list,other_prev_traj,np.array([other_prev_param]),opp_relative_weights)
  trajectory_generator_utils.get_s_cost_wlut_multiple(traj_costs_list,states_list,num_traj_list,self.waypoints,self.speed_lut_numba,self.lut_resolution)
  trajectory_generator_utils.get_range_costs_multiple(traj_costs_list,states_list,num_traj_list,self.range_lut_numba,self.lut_resolution)
  trajectory_generator_utils.get_progress_costs_multiple(traj_costs_list,end_xy_list,opp_relative_weights,num_traj_list,self.speed_lut_numba,self.lut_resolution)
  traj_costs_final_list=trajectory_generator_utils.sum_cost_multiple(traj_costs_list,picked_cost_weights)
  empty_cost_flag=False
  is_inf_flag=False
  safety_flag=False
  lowest_cost_idx_list=[np.argmin(traj_costs_final_list[i])for i in range(num_guys)]
  picked_traj_list=[new_states_list[i][lowest_cost_idx_list[i]*trajectory_generator.NUM_STEPS:(lowest_cost_idx_list[i]+1)*trajectory_generator.NUM_STEPS,:]for i in range(num_guys)]
  picked_params_list=[parameters_list[i][lowest_cost_idx_list[i],:]for i in range(num_guys)]
  max_belief_idx=np.argmax(picked_belief)
  for i in range(num_guys):
   if np.isinf(traj_costs_final_list[i][lowest_cost_idx_list[i]]):
    pp_traj,prev_traj,prev_param,extra_limp_s=getLimpParametersStatic(pose_x,pose_y,current_vel,self.prev_traj,self.prev_param)
    if max_belief_idx==i:
     self.limp_s+=extra_limp_s
    picked_params_list[i]=prev_param
    picked_traj_list[i]=prev_traj
   elif max_belief_idx==i:
    self.limp_s=0.
  self.prev_param=picked_params_list[max_belief_idx]
  self.prev_traj=picked_traj_list[max_belief_idx]
  return picked_traj_list,picked_params_list
 
 def plan_robust(self,pose,opp_pose,sampled_flow,other_prev_traj,other_prev_param,ds,current_vel,picked_idx_count,ballsize):
  current_vel=max(0.01,current_vel)
  pose_x,pose_y,pose_theta=pose
  lookup_grid,current_s,new_s,corner_on=flow_to_xytheta_static(sampled_flow[:,0:3],np.array(pose),self.waypoints,self.lut_resolution,self.speed_lut_numba,self.WINDOW_SIZE,self.CURVATURE_THRESH,self.CORNER_ON)
  self.CORNER_ON=corner_on
  if lookup_grid is None:
   safety_flag=False
   states_list_plot=None
   pp_traj,prev_traj_plot=self.getLimpParameters(pose_x,pose_y,current_vel)
   self.prev_flow=None
   return pp_traj,safety_flag,self.prev_flow,states_list_plot,prev_traj_plot,lookup_grid
  kappa0=trajectory_generator.get_curvature_command(np.roll(self.prev_param,-1),ds+self.limp_s)if self.prev_param is not None else 0.0
  emp_w=1.*picked_idx_count/np.sum(picked_idx_count)
  rot=self._rotation_matrix(pose_theta,(0,0,1))
  states_list_local,parameters_list,filtered_flow,filtered_grid,filtered_new_s=trajectory_generator_utils.grid_lookup(lookup_grid,self.lut_x,self.lut_y,self.lut_theta,self.lut_kappa,self.lut,sampled_flow,new_s,kappa0,self.lut_stepsizes)
  num_traj=parameters_list.shape[0]
  if num_traj==0:
   safety_flag=False
   states_list_plot=None
   pp_traj,prev_traj_plot=self.getLimpParameters(pose_x,pose_y,current_vel)
   self.prev_flow=None
   return pp_traj,safety_flag,self.prev_flow,states_list_plot,prev_traj_plot,lookup_grid
  states_list=trajectory_generator_utils.trans_traj_list(states_list_local,np.array([[pose_x],[pose_y],[0.0]]),rot)
  dspeed=filtered_flow[:,3:]
  new_states_list=trajectory_generator_utils.get_velocity_profile(states_list,self.waypoints,dspeed,num_traj,current_vel)
  if self.prev_traj is None or self.prev_param is None:
   self.prev_param=np.array([5.,0,0,0,0])
   self.prev_traj=np.zeros((trajectory_generator.NUM_STEPS,5))
   temp=np.linspace(0.,5.,trajectory_generator.NUM_STEPS)
   self.prev_traj[:,0]=pose[0]+temp*np.cos(pose[2])
   self.prev_traj[:,1]=pose[1]+temp*np.sin(pose[2])
   self.prev_traj[:,2]=pose[2]
   self.prev_traj[:,4]+=trajectory_generator_utils.WAYPOINT_SPEED
   self.prev_flow=None
   self.prev_steer=0.0 
   prev_traj_plot=self.prev_traj
   pp_traj=np.empty((self.prev_traj.shape[0],4))
   pp_traj[:,0:2]=self.prev_traj[:,0:2]
   pp_traj[:,2]=self.prev_traj[:,4]
   pp_traj[:,3]=self.prev_traj[:,2]
   return pp_traj,False,self.prev_flow,None,prev_traj_plot,lookup_grid
  traj_costs,end_xy,long_cost=trajectory_generator_utils.get_traj_list_cost_robust(states_list,new_states_list,self.cost_weights,self.waypoints,self.prev_traj,parameters_list,other_prev_traj,other_prev_param,emp_w)
  traj_costs[4,:]=trajectory_generator_utils.get_s_cost_wlut(states_list,num_traj,self.waypoints,self.speed_lut_numba,self.lut_resolution)
  traj_costs[9,:]=trajectory_generator_utils.get_range_costs(states_list,num_traj,self.range_lut_numba,self.lut_resolution)
  progress_cost=trajectory_generator_utils.get_progress_costs_robust(end_xy,long_cost.shape[0],num_traj,self.speed_lut_numba,self.lut_resolution)
  combined_robust_cost=trajectory_generator_utils.get_robust_cost(long_cost,self.cost_weights[11],progress_cost,self.cost_weights[12],picked_idx_count,ballsize)
  traj_costs_final=trajectory_generator_utils.sum_cost(traj_costs,self.cost_weights[:-2])+combined_robust_cost
  empty_cost_flag=False
  is_inf_flag=False
  safety_flag=False
  try:
   non_inf_idx=np.where(np.isfinite(traj_costs_final))[0]
   non_inf_costs=traj_costs_final[non_inf_idx]
   k=min(self.TOP_POP_NUM-1,non_inf_costs.shape[0])
   lowest_cost_idx_top=np.argpartition(non_inf_costs,k)[:k+1]
   lowest_cost_idx=np.argmin(non_inf_costs[lowest_cost_idx_top])
   lowest_cost_idx=lowest_cost_idx_top[lowest_cost_idx]
   lowest_cost_idx=non_inf_idx[lowest_cost_idx]
   best_traj=new_states_list[lowest_cost_idx*trajectory_generator.NUM_STEPS:(lowest_cost_idx+1)*trajectory_generator.NUM_STEPS,:]
   self.prev_traj=best_traj
   pp_traj=np.empty((best_traj.shape[0],4))
   pp_traj[:,0:2]=best_traj[:,0:2]
   pp_traj[:,2]=best_traj[:,4]
   pp_traj[:,3]=best_traj[:,2]
   self.prev_flow=(filtered_flow[non_inf_idx,:])[lowest_cost_idx_top,:]
   self.prev_param=parameters_list[lowest_cost_idx,:]
   self.is_limping=False
   self.limp_s=0.
  except ValueError:
   empty_cost_flag=True
   is_inf_flag=True
   self.prev_flow=None
  if empty_cost_flag or is_inf_flag:
   safety_flag=False
   states_list_plot=None
   pp_traj,prev_traj_plot=self.getLimpParameters(pose_x,pose_y,current_vel)
  else:
   states_list_plot=states_list
   prev_traj_plot=self.prev_traj
  return pp_traj,safety_flag,self.prev_flow,states_list_plot,prev_traj_plot,lookup_grid
# Created by pyminifier (https://github.com/liftoff/pyminifier)
