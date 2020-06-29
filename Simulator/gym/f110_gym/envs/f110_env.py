import gym
from gym import error,spaces,utils
from gym.utils import seeding
import zmq
import sim_requests_pb2
import numpy as np
from PIL import Image
import sys
import os
import signal
import subprocess
import math
import yaml
import csv
class F110Env(gym.Env,utils.EzPickle):
 metadata={'render.modes':[]}
 def __init__(self):
  self.params_set=False
  self.map_inited=False
  self.params=[]
  self.num_agents=2
  self.timestep=0.01
  self.map_path=None
  self.map_img=None
  self.ego_idx=0
  self.timeout=120.0
  self.start_thresh=0.5 
  self.x=None
  self.y=None
  self.theta=None
  self.in_collision=False
  self.collision_angle=None
  self.near_start=True
  self.num_toggles=0
  self.map_height=0.0
  self.map_width=0.0
  self.map_resolution=0.0
  self.free_thresh=0.0
  self.origin=[]
  tries=0
  max_tries=100
  min_port=6666
  self.port=min_port
  self.context=zmq.Context()
  self.socket=self.context.socket(zmq.PAIR)

  while tries<max_tries:
   try:
    self.socket.bind('tcp://*:%s'%str(min_port+tries))
    self.port=min_port+tries
    break
   except:
    tries=tries+1

  print('Gym env - Connected env to port: '+str(self.port))
  self.sim_p=None

 def __del__(self):
  if self.sim_p is None:
   pass
  else:
   os.kill(self.sim_p.pid,signal.SIGTERM)

 def _start_executable(self,path):
  mu=self.params[0]
  h_cg=self.params[1]
  l_r=self.params[2]
  cs_f=self.params[3]
  cs_r=self.params[4]
  I_z=self.params[5]
  mass=self.params[6]
  args=[path+'sim_server',str(self.timestep),str(self.num_agents),str(self.port),str(mu),str(h_cg),str(l_r),str(cs_f),str(cs_r),str(I_z),str(mass)]
  self.sim_p=subprocess.Popen(args)

 def _set_map(self):
  if not self.map_inited:
   print('Gym env - Sim map not initialized, call env.init_map() to init map.')

  map_request_proto=sim_requests_pb2.SimRequest()
  map_request_proto.type=1
  map_request_proto.map_request.map.extend((1.-self.map_img/255.).flatten().tolist())
  map_request_proto.map_request.origin_x=self.origin[0]
  map_request_proto.map_request.origin_y=self.origin[1]
  map_request_proto.map_request.map_resolution=self.map_resolution
  map_request_proto.map_request.free_threshold=self.free_thresh
  map_request_proto.map_request.map_height=self.map_height
  map_request_proto.map_request.map_width=self.map_width
  map_request_string=map_request_proto.SerializeToString()
  self.socket.send(map_request_string)
  sim_response_string=self.socket.recv()
  sim_response_proto=sim_requests_pb2.SimResponse()
  sim_response_proto.ParseFromString(sim_response_string)
  set_map_result=sim_response_proto.map_result.result
  if set_map_result==1:
   print('Gym env - Set map failed, exiting...')
   sys.exit()

 def _check_done(self):
  left_t=2.5
  right_t=5
  timeout=self.current_time>=self.timeout
  if self.double_finish:
   poses_x=np.array(self.all_x)-self.start_xs
   poses_y=np.array(self.all_y)-self.start_ys
   delta_pt=np.dot(self.start_rot,np.stack((poses_x,poses_y),axis=0))
   temp_y=delta_pt[1,:]
   idx1=temp_y>left_t
   idx2=temp_y<-right_t
   temp_y[idx1]-=left_t
   temp_y[idx2]=-right_t-temp_y[idx2]
   temp_y[np.invert(np.logical_or(idx1,idx2))]=0
   dist2=delta_pt[0,:]**2+temp_y**2
   closes=dist2<=0.1

   for i in range(self.num_agents):
    if closes[i]and not self.near_starts[i]:
     self.near_starts[i]=True
     self.toggle_list[i]+=1
    elif not closes[i]and self.near_starts[i]:
     self.near_starts[i]=False
     self.toggle_list[i]+=1
   done=(self.in_collision|(timeout)|np.all(self.toggle_list>=4))
   return done,self.toggle_list>=4

  delta_pt=np.dot(self.start_rot,np.array([self.x-self.start_x,self.y-self.start_y]))
  if delta_pt[1]>left_t:
   temp_y=delta_pt[1]-left_t
  elif delta_pt[1]<-right_t:
   temp_y=-right_t-delta_pt[1]
  else:
   temp_y=0
  dist2=delta_pt[0]**2+temp_y**2
  close=dist2<=0.1
  if close and not self.near_start:
   self.near_start=True
   self.num_toggles+=1
  elif not close and self.near_start:
   self.near_start=False
   self.num_toggles+=1
  done=(self.in_collision|(timeout)|(self.num_toggles>=4))
  return done

 def _check_passed(self):
  return 0

 def _update_state(self,obs_dict):
  self.x=obs_dict['poses_x'][obs_dict['ego_idx']]
  self.y=obs_dict['poses_y'][obs_dict['ego_idx']]
  if self.double_finish:
   self.all_x=obs_dict['poses_x']
   self.all_y=obs_dict['poses_y']
  self.theta=obs_dict['poses_theta'][obs_dict['ego_idx']]
  self.in_collision=obs_dict['collisions'][obs_dict['ego_idx']]
  self.collision_angle=obs_dict['collision_angles'][obs_dict['ego_idx']]

 def _raycast_opponents(self,obs_dict):
  new_obs={}
  return new_obs

 def step(self,action):
  if not self.params_set:
   print('ERROR - Gym Env - Params not set, call update params before stepping.')
   sys.exit()

  step_request_proto=sim_requests_pb2.SimRequest()
  step_request_proto.type=0
  step_request_proto.step_request.ego_idx=action['ego_idx']
  step_request_proto.step_request.requested_vel.extend(action['speed'])
  step_request_proto.step_request.requested_ang.extend(action['steer'])
  step_request_string=step_request_proto.SerializeToString()
  self.socket.send(step_request_string)
  sim_response_string=self.socket.recv()
  sim_response_proto=sim_requests_pb2.SimResponse()
  sim_response_proto.ParseFromString(sim_response_string)
  response_type=sim_response_proto.type
  if not response_type==0:
   print('Gym env - Wrong response type for stepping, exiting...')
   sys.exit()

  observations_proto=sim_response_proto.sim_obs
  if not observations_proto.ego_idx==action['ego_idx']:
   print('Gym env - Ego index mismatch, exiting...')
   sys.exit()

  carobs_list=observations_proto.observations
  obs={'ego_idx':observations_proto.ego_idx,'scans':[],'poses_x':[],'poses_y':[],'poses_theta':[],'linear_vels_x':[],'linear_vels_y':[],'ang_vels_z':[],'collisions':[],'collision_angles':[],'min_dists':[]}
  for car_obs in carobs_list:
   obs['scans'].append(car_obs.scan)
   obs['poses_x'].append(car_obs.pose_x)
   obs['poses_y'].append(car_obs.pose_y)
   if abs(car_obs.theta)<np.pi:
    obs['poses_theta'].append(car_obs.theta)
   else:
    obs['poses_theta'].append(-((2*np.pi)-car_obs.theta))
   obs['linear_vels_x'].append(car_obs.linear_vel_x)
   obs['linear_vels_y'].append(car_obs.linear_vel_y)
   obs['ang_vels_z'].append(car_obs.ang_vel_z)
   obs['collisions'].append(car_obs.collision)
   obs['collision_angles'].append(car_obs.collision_angle)
  reward=self.timestep
  self.current_time=self.current_time+self.timestep
  self._update_state(obs)
  if self.double_finish:
   done,temp=self._check_done()
   info={'checkpoint_done':temp}
  else:
   done=self._check_done()
   info={}
  return obs,reward,done,info

 def reset(self,poses=None):
  self.current_time=0.0
  self.in_collision=False
  self.collision_angles=None
  self.num_toggles=0
  self.near_start=True
  self.near_starts=np.array([True]*self.num_agents)
  self.toggle_list=np.zeros((self.num_agents,))
  if poses:
   pose_x=poses['x']
   pose_y=poses['y']
   pose_theta=poses['theta']
   self.start_x=pose_x[0]
   self.start_y=pose_y[0]
   self.start_theta=pose_theta[0]
   self.start_xs=np.array(pose_x)
   self.start_ys=np.array(pose_y)
   self.start_thetas=np.array(pose_theta)
   self.start_rot=np.array([[np.cos(-self.start_theta),-np.sin(-self.start_theta)],[np.sin(-self.start_theta),np.cos(-self.start_theta)]])
   reset_request_proto=sim_requests_pb2.SimRequest()
   reset_request_proto.type=4
   reset_request_proto.reset_bypose_request.num_cars=self.num_agents
   reset_request_proto.reset_bypose_request.ego_idx=0
   reset_request_proto.reset_bypose_request.car_x.extend(pose_x)
   reset_request_proto.reset_bypose_request.car_y.extend(pose_y)
   reset_request_proto.reset_bypose_request.car_theta.extend(pose_theta)
   reset_request_string=reset_request_proto.SerializeToString()
   self.socket.send(reset_request_string)
  else:
   self.start_x=0.0
   self.start_y=0.0
   self.start_theta=0.0
   self.start_rot=np.array([[np.cos(-self.start_theta),-np.sin(-self.start_theta)],[np.sin(-self.start_theta),np.cos(-self.start_theta)]])
   reset_request_proto=sim_requests_pb2.SimRequest()
   reset_request_proto.type=2
   reset_request_proto.reset_request.num_cars=self.num_agents
   reset_request_proto.reset_request.ego_idx=0
   reset_request_string=reset_request_proto.SerializeToString()
   self.socket.send(reset_request_string)
  reset_response_string=self.socket.recv()
  reset_response_proto=sim_requests_pb2.SimResponse()
  reset_response_proto.ParseFromString(reset_response_string)
  if reset_response_proto.reset_resp.result:
   print('Gym env - Reset failed')
   return None

  vels=[0.0]*self.num_agents
  angs=[0.0]*self.num_agents
  action={'ego_idx':self.ego_idx,'speed':vels,'steer':angs}
  obs,reward,done,info=self.step(action)
  return obs,reward,done,info

 def init_map(self,map_path,img_ext,rgb,flip):
  self.map_path=map_path
  if not map_path.endswith('.yaml'):
   print('Gym env - Please use a yaml file for map initialization.')
   print('Exiting...')
   sys.exit()
  map_img_path=os.path.splitext(self.map_path)[0]+img_ext
  self.map_img=np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
  self.map_img=self.map_img.astype(np.float64)
  if flip:
   self.map_img=self.map_img[::-1]
  if rgb:
   self.map_img=np.dot(self.map_img[...,:3],[0.29,0.57,0.14])
  self.map_height=self.map_img.shape[0]
  self.map_width=self.map_img.shape[1]
  self.free_thresh=0.6 
  with open(self.map_path,'r')as yaml_stream:
   try:
    map_metadata=yaml.safe_load(yaml_stream)
    self.map_resolution=map_metadata['resolution']
    self.origin=map_metadata['origin']
   except yaml.YAMLError as ex:
    print(ex)
  self.map_inited=True

 def render(self,mode='human',close=False):
  return
  
 def update_params(self,mu,h_cg,l_r,cs_f,cs_r,I_z,mass,exe_path,double_finish=False):
  self.params=[mu,h_cg,l_r,cs_f,cs_r,I_z,mass]
  self.params_set=True
  if self.sim_p is None:
   self._start_executable(exe_path)
   self._set_map()
  self.double_finish=double_finish
  update_param_proto=sim_requests_pb2.SimRequest()
  update_param_proto.type=3
  update_param_proto.update_request.mu=mu
  update_param_proto.update_request.h_cg=h_cg
  update_param_proto.update_request.l_r=l_r
  update_param_proto.update_request.cs_f=cs_f
  update_param_proto.update_request.cs_r=cs_r
  update_param_proto.update_request.I_z=I_z
  update_param_proto.update_request.mass=mass
  update_param_string=update_param_proto.SerializeToString()
  self.socket.send(update_param_string)
  update_response_string=self.socket.recv()
  update_response_proto=sim_requests_pb2.SimResponse()
  update_response_proto.ParseFromString(update_response_string)
  if update_response_proto.update_resp.result:
   print('Gym env - Update param failed')
   return None
# Created by pyminifier (https://github.com/liftoff/pyminifier)
