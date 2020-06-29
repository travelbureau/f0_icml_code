import argparse
import numpy as np
import os
import time
import torch
import random
import yaml
import gym
import csv
import multiprocessing as mp
from numba import njit
import scipy.stats as sps
import pickle
from concurrent.futures import ThreadPoolExecutor
from mpc import lattice_planner,pure_pursuit_utils
import sys

sys.path.append('./flows')
from maf import MAF
from group_maf import GroupMAF

@torch.no_grad()
def sampleFlow(model,obs,n_row,index):
 model.eval()
 u=model.base_dist.sample((n_row,1)).squeeze()
 samples,_=model.inverse(u,torch.from_numpy(obs[index].astype(np.float32)))
 return samples.numpy().astype(np.float64)

def getPlannerObs(obs):
 ego_pose=[obs['poses_x'][EGO_IDX],obs['poses_y'][EGO_IDX],obs['poses_theta'][EGO_IDX]]
 opp_pose=[obs['poses_x'][EGO_IDX+1],obs['poses_y'][EGO_IDX+1],obs['poses_theta'][EGO_IDX+1]]
 ego_vel=obs['linear_vels_x'][EGO_IDX]
 opp_vel=obs['linear_vels_x'][EGO_IDX+1]
 return ego_pose,opp_pose,ego_vel,opp_vel

def getFlowObs(obs):
 flow_obs=np.empty((NUM_AGENTS,COND_LABEL_SIZE))
 for i in range(NUM_AGENTS):
  flow_obs[i,:100]=np.take(obs['scans'][i],scan_sub_idx)/MAX_LIDAR
  flow_obs[i,100]=obs['linear_vels_x'][i]/MAX_SPEED
 return flow_obs

def unnormalizeFlow(grid):
 grid[:,0]+=FLOW_S_SHIFT
 grid[:,1]*=T_SCALE
 grid[:,2]*=THETA_SCALE
 grid[:,3:]*=V_SCALE
 return grid

def normalizeFlow(grid):
 grid[:,0]-=FLOW_S_SHIFT
 grid[:,1]/=T_SCALE
 grid[:,2]/=THETA_SCALE
 grid[:,3:]/=V_SCALE
 return grid

def getFlow(flow,flow_weights):
 b=bytes(flow_weights.astype(np.uint8))
 flow.load_state_dict(pickle.loads(b))
 return flow

def load_raceline(file_path):
 with open(file_path)as f:
  waypoints=[tuple(line)for line in csv.reader(f)]
  waypoints=np.array([(float(pt[0]),float(pt[1]),float(pt[2]),float(pt[3]),float(pt[4]),float(pt[5]))for pt in waypoints])
 return waypoints

WAYPOINT_START_IDX=250
OPP_HEADSTART=1.5
SIDE_START=1.

def generateInitPoses(waypoints=None):
 if waypoints is None:
  ego_x=0
  ego_y=0
  ego_t=0
  opp_x=ego_x+OPP_HEADSTART*np.cos(ego_t)
  opp_y=ego_y+OPP_HEADSTART*np.sin(ego_t)
 else:
  pt=waypoints[WAYPOINT_START_IDX]
  ego_t=pt[3]
  ego_x=pt[0]+SIDE_START*np.cos(ego_t+np.pi/2)
  ego_y=pt[1]+SIDE_START*np.sin(ego_t+np.pi/2)
  opp_x=pt[0]+SIDE_START*np.cos(ego_t-np.pi/2)+OPP_HEADSTART*np.cos(ego_t)
  opp_y=pt[1]+SIDE_START*np.sin(ego_t-np.pi/2)+OPP_HEADSTART*np.sin(ego_t)
 return np.array([[ego_x,ego_y,ego_t],[opp_x,opp_y,ego_t]])

NUM_FLOW_SAMPLES=50
FLOW_S_SHIFT=5.
THETA_SCALE=0.25 
T_SCALE=0.75 
V_SCALE=2.0
NUM_AGENTS=2
EGO_IDX=0
INPUT_SIZE=6
PLANNER_DT=0.1
PHYSICS_DT=0.01
N_BLOCKS=5
HIDDEN_SIZE=100
N_HIDDEN=1
COND_LABEL_SIZE=101
ACTIVATION_FCN='relu'
INPUT_ORDER='sequential'
BATCH_NORM=False
mass=3.74
l_r=0.17145
I_z=0.04712
mu=0.523
h_cg=0.074
cs_f=4.718
cs_r=5.4562
MAX_LIDAR=30.
MAX_SPEED=20.
scan_sub_idx=np.linspace(0,1079,100).astype(int)
parser=argparse.ArgumentParser()
parser.add_argument('--result_npz_path',type=str,default='cost_weights.npz')
parser.add_argument('--update_belief',type=int,default=1,help='default 1')
parser.add_argument('--ball_size',type=float,default=0.1,help='default 0.1')
parser.add_argument('--ego_frozen_flow',type=int,default=0,help='default 0')
parser.add_argument('--eval_iters',type=int,default=1,help='default 1')
parser.add_argument('--same_guy',type=int,default=0,help='default 0')
parser.add_argument('--double_finish',type=int,default=1,help='default 1')
parser.add_argument('--record_regret',type=int,default=1,help='default 1')
ARGS=parser.parse_args()

CONFIG=None
in_docker=os.environ.get('IM_IN_DOCKER',False)
viz=not in_docker
VIZ=None
if viz:
 from pango_visualizer import PangoViz

EVAL_ITERS=2*ARGS.eval_iters
GROUND_TRUTH_IDX=None
POOL=ThreadPoolExecutor(1)

def extract_cost(npz_in_path):
 return np.load(npz_in_path)['cost_weights']

OPP_COST_WEIGHTS=extract_cost(ARGS.result_npz_path)
N_OPP_TOT,NUM_COSTS=OPP_COST_WEIGHTS.shape
DPP_IDX=np.array([8,10,22,33,57,94,127,136,150,153])
BEST_IDX=33
N_ARMS=8
ROLLOUT_LENGTH=250
exp3_const=(N_OPP_TOT+N_ARMS-1)*1./N_ARMS
ETA_EXP3=5*np.sqrt(2*np.log(N_OPP_TOT)/ROLLOUT_LENGTH/exp3_const)
OPP_IDX=np.arange(N_OPP_TOT,dtype=int)
DEVICE=torch.device('cuda:0' if torch.cuda.is_available()else 'cpu')
EGO_FLOW_MODEL=MAF(N_BLOCKS,INPUT_SIZE,HIDDEN_SIZE,N_HIDDEN,COND_LABEL_SIZE,ACTIVATION_FCN,INPUT_ORDER,BATCH_NORM).to(DEVICE)
OPP_FLOW_MODEL=MAF(N_BLOCKS,INPUT_SIZE,HIDDEN_SIZE,N_HIDDEN,COND_LABEL_SIZE,ACTIVATION_FCN,INPUT_ORDER,BATCH_NORM).to(DEVICE)
NUM_GUYS=N_OPP_TOT
MAX_S=6.0
RHO_EGO=ARGS.ball_size
RHO_OPP=RHO_EGO
USE_ONLY_DPP=True
if USE_ONLY_DPP:
 NUM_GUYS=len(DPP_IDX)
 DPP_MAP={}
 for i,dpp_idx in enumerate(DPP_IDX):
  DPP_MAP[dpp_idx]=i

@torch.no_grad()
def sampleGroupFlow(group_model,obs,n_row):
 obs=torch.from_numpy(obs.astype(np.float32)).to(DEVICE)
 group_model.eval()
 group_u=group_model.base_dist.sample((group_model.group_length,n_row)).to(DEVICE)
 group_obs=obs[None,None,:].repeat(group_model.group_length,1,1).to(DEVICE)
 samples,_=group_model.inverse(group_u,group_obs)
 if DEVICE.type=='cuda':
  temp=samples.cpu().numpy().astype(np.float64)
 else:
  temp=samples.numpy().astype(np.float64)
 return temp

def getGroupFlow(flow_weights_dir,cost_idx):
 model_list=[]
 for idx in cost_idx:
  single_model=MAF(N_BLOCKS,INPUT_SIZE,HIDDEN_SIZE,N_HIDDEN,COND_LABEL_SIZE,ACTIVATION_FCN,INPUT_ORDER,BATCH_NORM).to(DEVICE)
  single_model.load_state_dict(torch.load(flow_weights_dir+'/model_state_cost_'+str(idx)+'.pt',map_location=DEVICE))
  model_list.append(single_model)
 group_model=GroupMAF(DEVICE,model_list,INPUT_SIZE,HIDDEN_SIZE,N_HIDDEN,ACTIVATION_FCN,INPUT_ORDER).to(DEVICE)
 group_model.eval()
 return group_model

def unnormalizeGroupFlow(grid):
 grid[:,:,0]+=FLOW_S_SHIFT
 grid[:,:,1]*=T_SCALE
 grid[:,:,2]*=THETA_SCALE
 grid[:,:,3:]*=V_SCALE
 return grid

def normalizeGroupFlow(grid):
 grid[:,:,0]-=FLOW_S_SHIFT
 grid[:,:,1]/=T_SCALE
 grid[:,:,2]/=THETA_SCALE
 grid[:,:,3:]/=V_SCALE

def sampleArm(n_arms,belief):
 picked=np.random.choice(OPP_IDX,n_arms,replace=True,p=belief)
 return picked

@njit(cache=True)
def cross(vec1,vec2):
 return vec1[0]*vec2[1]-vec1[1]*vec2[0]

@njit(fastmath=False,cache=True)
def updateBeliefHelper(picked_idx_unique,glob_prev_s,glob_prev_opp_pose,waypoints):
 end_idx_start=np.searchsorted(glob_prev_s,glob_prev_s[0]+2,side='right')
 hi=range(end_idx_start,len(glob_prev_s))
 if len(hi)==0:
  return None
 flow_samples=np.empty((len(hi),6))
 for end_idx in range(end_idx_start,len(glob_prev_s)):
  knots=np.linspace(0,end_idx,4).astype(np.int32)
  knots=knots[1:]
  vels=np.empty((3,))
  for i in range(knots.shape[0]):
   vels[i]=glob_prev_opp_pose[knots[i],3]
  _,min_dist,min_frac_t,min_i=pure_pursuit_utils.nearest_point_on_trajectory_py2(glob_prev_opp_pose[knots[-1],0:2],waypoints[:,0:2])
  nearest_waypoint=waypoints[min_i]
  end_theta=glob_prev_opp_pose[knots[-1],2]-nearest_waypoint[3]
  end_s=glob_prev_s[knots[-1]]-glob_prev_s[0]
  vec_to_pt=glob_prev_opp_pose[knots[-1],0:2]-nearest_waypoint[0:2]
  wpt_pt=np.array([np.cos(end_theta),np.sin(end_theta)])
  if cross(wpt_pt,vec_to_pt)<0:
   end_t=-min_dist
  else:
   end_t=min_dist
  flow_samples[end_idx-end_idx_start]=np.concatenate((np.array([end_s,end_t,end_theta]),vels))
 return flow_samples

@njit(fastmath=False,cache=True)
def EXP3(belief_vector,loss,picked_idx_unique,picked_idx_count,record_regret):
 L=np.repeat(loss,picked_idx_count)
 L_idx=np.repeat(picked_idx_unique,picked_idx_count)
 m=L.shape[0]
 for i in range(m):
  idx=L_idx[i]
  update=L[i]/belief_vector[idx]/m
  belief_vector[idx]*=np.exp(-ETA_EXP3*update)
 belief_vector/=np.sum(belief_vector)
 if record_regret:
  return np.mean(L)

def normalizeLogProb(log_prob):
 log_prob[np.isnan(log_prob)]=-9.
 log_prob=np.clip(log_prob,-9.,-6.)
 log_prob=-(log_prob+6.)/(-6.+9.)
 return log_prob

def updateBelief(belief_vector,picked_idx_unique,picked_idx_count,prev_s,prev_opp_pose,prev_opp_obs0,waypoints,opp_group_flow,opp_flow_choice,step_obs):
 if opp_flow_choice is None:
  return None
 flow_obs=torch.from_numpy(step_obs[EGO_IDX+1][None,None,:].astype(np.float32)).repeat(NUM_GUYS,opp_flow_choice.shape[0],1).to(DEVICE)
 normalized_flow_samples=torch.from_numpy(normalizeFlow(opp_flow_choice)[None,:,:].astype(np.float32)).repeat(NUM_GUYS,1,1).to(DEVICE)
 log_prob=opp_group_flow.log_prob(normalized_flow_samples,flow_obs)
 if not hasattr(updateBelief,'running_logprob'):
  updateBelief.running_logprob=log_prob.sum(1)
  updateBelief.steps=log_prob.shape[1]
 else:
  updateBelief.running_logprob=updateBelief.running_logprob+log_prob.sum(1)
  updateBelief.steps+=log_prob.shape[1]
 log_prob=updateBelief.running_logprob/updateBelief.steps
 if not USE_ONLY_DPP:
  picked_log_prob=log_prob[picked_idx_unique]
 else:
  real_idx=np.array([DPP_MAP[u]for u in picked_idx_unique])
  picked_log_prob=log_prob[real_idx]
 picked_log_prob=picked_log_prob.cpu().detach().numpy()
 loss=normalizeLogProb(picked_log_prob)
 if ARGS.record_regret:
  insta_regret=EXP3(belief_vector,loss,picked_idx_unique,picked_idx_count,ARGS.record_regret)
  if not USE_ONLY_DPP:
   insta_regret-=normalizeLogProb(np.array([log_prob[GROUND_TRUTH_IDX].item()]))
  else:
   insta_regret-=normalizeLogProb(np.array([log_prob[DPP_MAP[GROUND_TRUTH_IDX]].item()]))
 else:
  EXP3(belief_vector,loss,picked_idx_unique,picked_idx_count,ARGS.record_regret)
 if ARGS.record_regret:
  return insta_regret
 return None

def resetPlayers(ego_pose,opp_pose,ego_cost,opp_cost,ego_flow_weights,opp_flow_weights,racecar_env,waypoints,worker_directory):
 map_name=CONFIG['map_name']
 ego_flow=getFlow(EGO_FLOW_MODEL,ego_flow_weights)
 opp_flow=getFlow(OPP_FLOW_MODEL,opp_flow_weights)
 if not USE_ONLY_DPP:
  opp_group_flow=getGroupFlow(worker_directory+'./flow_weights',OPP_IDX)
 else:
  opp_group_flow=getGroupFlow(worker_directory+'./flow_weights',DPP_IDX)
 obs,step_reward,done,info=racecar_env.reset({'x':[ego_pose[0],opp_pose[0]],'y':[ego_pose[1],opp_pose[1]],'theta':[ego_pose[2],opp_pose[2]]})
 if not hasattr(resetPlayers,'ego_planner'):
  resetPlayers.ego_planner=None
  resetPlayers.opp_planner=None
  resetPlayers.multiple_planner=None
  resetPlayers.multiple_planner_opp=None
 if resetPlayers.ego_planner is None:
  resetPlayers.ego_planner=lattice_planner.RobustLatticePlanner(worker_directory+'../maps/'+map_name,waypoints,worker_directory,ego_cost,is_ego=True)
  resetPlayers.opp_planner=lattice_planner.RobustLatticePlanner(worker_directory+'../maps/'+map_name,waypoints,worker_directory,opp_cost,is_ego=False)
  resetPlayers.multiple_planner=lattice_planner.RobustLatticePlanner(worker_directory+'../maps/'+map_name,waypoints,worker_directory,cost_weights=None,is_ego=True)
  resetPlayers.multiple_planner_opp=lattice_planner.RobustLatticePlanner(worker_directory+'../maps/'+map_name,waypoints,worker_directory,cost_weights=None,is_ego=False)
 else:
  resetPlayers.ego_planner.update_cost(ego_cost)
  resetPlayers.opp_planner.update_cost(opp_cost)
 if hasattr(updateBelief,'running_logprob'):
  delattr(updateBelief,'running_logprob')
  delattr(updateBelief,'steps')
 return resetPlayers.ego_planner,resetPlayers.opp_planner,resetPlayers.multiple_planner,resetPlayers.multiple_planner_opp,ego_flow,opp_flow,obs,opp_group_flow

def groupGridHelper(group_flow,step_obs,index,picked_unique):
 group_grid=sampleGroupFlow(group_flow,step_obs[index,:],NUM_FLOW_SAMPLES)
 if not USE_ONLY_DPP:
  picked_grid=group_grid[picked_unique,:,:]
 else:
  real_idx=np.array([DPP_MAP[u]for u in picked_unique])
  picked_grid=group_grid[real_idx,:,:]
 if len(picked_grid.shape)<3:
  picked_grid=picked_grid[None,:,:]
 picked_grid=unnormalizeGroupFlow(picked_grid)
 picked_cost_weights=OPP_COST_WEIGHTS[picked_unique,:]
 if len(picked_grid.shape)<2:
  picked_cost_weights=picked_cost_weights[None,:]
 return picked_grid,picked_cost_weights

def simulationLoop(ego_pose0,opp_pose0,ego_cost,opp_cost,ego_flow_weights,opp_flow_weights,racecar_env,worker_directory,waypoints):
 prev_pose=[]
 prev_opp_pose=[]
 prev_opp_obs=[]
 prev_s=[]
 future_list=[]
 belief_vector=np.ones((N_OPP_TOT,))*(1.0/N_OPP_TOT)
 belief_vector=np.zeros((N_OPP_TOT,))
 belief_vector[DPP_IDX]=1.
 belief_vector/=np.sum(belief_vector)
 belief_vector_opp=np.ones((N_OPP_TOT,))*(1.0/N_OPP_TOT)
 belief_vector_opp=np.zeros((N_OPP_TOT,))
 belief_vector_opp[DPP_IDX]=1.
 belief_vector_opp/=np.sum(belief_vector_opp)
 ego_planner,opp_planner,multiple_planner,multiple_planner_opp,ego_flow,opp_flow,obs,opp_group_flow=resetPlayers(ego_pose0,opp_pose0,ego_cost,opp_cost,ego_flow_weights,opp_flow_weights,racecar_env,waypoints,worker_directory)
 done=False
 score=0.
 checkpoint_times=[np.inf,np.inf]
 belief_hist=[]
 regret_hist=[]
 pose_hist=[]
 belief_hist.append(np.copy(belief_vector))
 insta_regret=None
 while not done:
  ego_pose,opp_pose,ego_vel,opp_vel=getPlannerObs(obs)
  pose_hist.append([*ego_pose,ego_vel,*opp_pose,opp_vel])
  step_obs=getFlowObs(obs)
  opp_lookup_grid=sampleFlow(opp_flow,step_obs,NUM_FLOW_SAMPLES,EGO_IDX+1)
  ego_lookup_grid=sampleFlow(ego_flow,step_obs,NUM_FLOW_SAMPLES,EGO_IDX)
  opp_lookup_grid=unnormalizeFlow(opp_lookup_grid)
  ego_lookup_grid=unnormalizeFlow(ego_lookup_grid)
  prev_opp_obs.append(step_obs[EGO_IDX+1,:])
  prev_pose.append([*ego_pose,ego_vel])
  prev_opp_pose.append([*opp_pose,opp_vel])
  if len(prev_s)==0:
   prev_s.append(0.)
   continue
  else:
   prev_s.append(prev_s[-1]+np.linalg.norm(np.subtract(prev_opp_pose[-1][0:2],prev_opp_pose[-2][0:2])))
   if len(prev_opp_pose)<4:
    continue
  while prev_s[-1]>=prev_s[0]+MAX_S:
   prev_s.pop(0)
   prev_opp_pose.pop(0)
   prev_pose.pop(0)
   prev_opp_obs.pop(0)
  d_opp=np.subtract(prev_opp_pose[-1][0:2],prev_opp_pose[-2][0:2])
  ds_opp=np.linalg.norm(d_opp)
  d_ego=np.subtract(prev_pose[-1][0:2],prev_pose[-2][0:2])
  ds_ego=np.linalg.norm(d_ego)
  picked_idx_opp=sampleArm(N_ARMS,belief_vector_opp)
  picked_idx_unique_opp,picked_idx_count_opp=np.unique(picked_idx_opp,return_counts=True)
  picked_belief_opp=belief_vector_opp[picked_idx_unique_opp]
  picked_grid_opp,picked_cost_weights_opp=groupGridHelper(opp_group_flow,step_obs,EGO_IDX,picked_idx_unique_opp)
  oppego_picked_traj_list,oppego_picked_param_list=multiple_planner_opp.plan_multiple(ego_pose[0:3],opp_pose[0:3],picked_grid_opp,opp_planner.prev_traj,opp_planner.prev_param,ds_ego,ego_vel,picked_cost_weights_opp,picked_belief_opp)
  oppego_picked_traj_list=np.concatenate(oppego_picked_traj_list,axis=0)
  oppego_picked_param_list=np.vstack(oppego_picked_param_list)
  op_pp_traj,op_safety_flag,opp_flow_choice,op_all_states,op_picked_state,op_xy_grid=opp_planner.plan_robust(opp_pose[0:3],ego_pose[0:3],opp_lookup_grid,oppego_picked_traj_list,oppego_picked_param_list,ds_opp,opp_vel,picked_idx_count_opp,RHO_OPP)
  picked_idx=sampleArm(N_ARMS,belief_vector)
  picked_idx_unique,picked_idx_count=np.unique(picked_idx,return_counts=True)
  if len(future_list)>0:
   if future_list[0].done():
    belief_vector=future_list[0].result()
    future_list.pop(0)
  if ARGS.update_belief:
   insta_regret=updateBelief(belief_vector,picked_idx_unique,picked_idx_count,np.array(prev_s),np.array(prev_opp_pose),prev_opp_obs[0],waypoints,opp_group_flow,opp_flow_choice,step_obs)
  if ARGS.record_regret and insta_regret is not None:
   regret_hist.append(insta_regret)
  belief_hist.append(np.copy(belief_vector))
  picked_belief=belief_vector[picked_idx_unique]
  picked_grid,picked_cost_weights=groupGridHelper(opp_group_flow,step_obs,EGO_IDX+1,picked_idx_unique)
  opp_picked_traj_list,opp_picked_param_list=multiple_planner.plan_multiple(opp_pose[0:3],ego_pose[0:3],picked_grid,ego_planner.prev_traj,ego_planner.prev_param,ds_opp,opp_vel,picked_cost_weights,picked_belief)
  opp_picked_traj_list=np.concatenate(opp_picked_traj_list,axis=0)
  opp_picked_param_list=np.vstack(opp_picked_param_list)
  ego_pp_traj,ego_safety_flag,ego_flow_choice,ego_all_states,ego_picked_state,ego_xy_grid=ego_planner.plan_robust(ego_pose[0:3],opp_pose[0:3],ego_lookup_grid,opp_picked_traj_list,opp_picked_param_list,ds_ego,ego_vel,picked_idx_count,RHO_EGO)
  
  if viz:
   VIZ.update(obs,ego_lookup_grid,opp_lookup_grid,op_all_states,op_picked_state,ego_all_states,ego_picked_state,ego_planner.CORNER_ON,ego_xy_grid)
  for i in range(int(PLANNER_DT/PHYSICS_DT)):
   if i>0:
    ego_pose,opp_pose,ego_vel,opp_vel=getPlannerObs(obs)
   op_speed,op_steer=opp_planner.compute_action(op_pp_traj,op_safety_flag,opp_pose)
   ego_speed,ego_steer=ego_planner.compute_action(ego_pp_traj,ego_safety_flag,ego_pose)
   action={'ego_idx':EGO_IDX,'speed':[ego_speed,op_speed*lattice_planner.LatticePlanner.OPP_SPEED_SCALE],'steer':[ego_steer,op_steer]}
   obs,step_reward,done,info=racecar_env.step(action)
   score+=step_reward
   if ARGS.double_finish:
    for i,val in enumerate(info['checkpoint_done']):
     if val and(checkpoint_times[i]==np.inf):
      checkpoint_times[i]=score
   if done:
    break
 return score,checkpoint_times,np.array(regret_hist),np.array(belief_hist),np.array(pose_hist)

def worker_func(worker_directory):
 global GROUND_TRUTH_IDX
 global VIZ
 global WPTS_DIR,CONFIG
 with open(worker_directory+'config.yaml','r')as yaml_stream:
  try:
   CONFIG=yaml.safe_load(yaml_stream)
   speed_lut_name=CONFIG['speed_lut_name']
   range_lut_name=CONFIG['range_lut_name']
   csv_name=CONFIG['csv_name']
   map_img_ext=CONFIG['map_img_ext']
   map_name=CONFIG['map_name']
   map_prefix=CONFIG['map_prefix']
  except yaml.YAMLError as ex:
   print(ex)
 WPTS_DIR=worker_directory+'../maps/'+csv_name
 waypoints=load_raceline(WPTS_DIR)
 if viz:
  VIZ=PangoViz(worker_directory,worker_directory+'../maps/'+map_prefix+map_img_ext,worker_directory+'../maps/'+map_name,waypoints,False)
 racecar_env=gym.make('f110_gym:f110-v0')
 racecar_env.init_map(worker_directory+'../maps/'+map_name,map_img_ext,False,False)
 racecar_env.update_params(mu,h_cg,l_r,cs_f,cs_r,I_z,mass,worker_directory+'../build/',ARGS.double_finish)
 poses0=generateInitPoses(waypoints)
 flow_weights_dir=worker_directory+'flow_weights/model_state_cost_'
 counter=0
 for opp_idx in DPP_IDX:
  score_histhist=[]
  checkpoint_histhist=[]
  regret_histhist=[]
  belief_histhist=[]
  pose_histhist=[]
  print('ground truth opp idx',opp_idx)
  if ARGS.record_regret:
   GROUND_TRUTH_IDX=opp_idx
  ego_idx=BEST_IDX
  if ARGS.same_guy:
   ego_idx=opp_idx
  ego_cost=OPP_COST_WEIGHTS[ego_idx]
  opp_cost=OPP_COST_WEIGHTS[opp_idx]
  if ARGS.ego_frozen_flow:
   ego_flow_weights=np.array(bytearray(pickle.dumps(torch.load(worker_directory+'flows/model_state.pt',map_location=DEVICE)))).astype(np.float64)
  else:
   ego_flow_weights=np.array(bytearray(pickle.dumps(torch.load(flow_weights_dir+str(ego_idx)+'.pt',map_location=DEVICE)))).astype(np.float64)
  opp_flow_weights=np.array(bytearray(pickle.dumps(torch.load(flow_weights_dir+str(opp_idx)+'.pt',map_location=DEVICE)))).astype(np.float64)
  for ii in range(EVAL_ITERS):
   seed=int(ego_cost[0]*1e6+ii)
   np.random.seed(seed)
   random.seed(seed)
   torch.manual_seed(seed)
   if DEVICE.type=='cuda':
    torch.cuda.manual_seed(seed)
   start_time=time.time()
   if ii<EVAL_ITERS/2:
    score,checkpoint_times,regret_hist,belief_hist,pose_hist=simulationLoop(poses0[EGO_IDX],poses0[EGO_IDX+1],ego_cost,opp_cost,ego_flow_weights,opp_flow_weights,racecar_env,worker_directory,waypoints)
    print('checkpoint',checkpoint_times)
   else:
    score,checkpoint_times,regret_hist,belief_hist,pose_hist=simulationLoop(poses0[EGO_IDX+1],poses0[EGO_IDX],ego_cost,opp_cost,ego_flow_weights,opp_flow_weights,racecar_env,worker_directory,waypoints)
    print('checkpoint',checkpoint_times)
   print('Iteration time: '+str(time.time()-start_time),'Score',score)
   score_histhist.append(score)
   checkpoint_histhist.append(checkpoint_times)
   regret_histhist.append(regret_hist)
   belief_histhist.append(belief_hist)
   pose_histhist.append(pose_hist)
  savestring='belief'+str(ARGS.update_belief)+'_'+'ball_size'+str(ARGS.ball_size)+'_'+'frozen'+str(ARGS.ego_frozen_flow)+'_'+'iters'+str(ARGS.eval_iters)+'_'+'sameguy'+str(ARGS.same_guy)+'_'+str(opp_idx)
  np.savez_compressed(savestring+'.npz',score=score_histhist,checkpoint_times=checkpoint_histhist,regret_hist=regret_histhist,belief_hist=belief_histhist,pose_hist=pose_histhist,args=ARGS)

if __name__=="__main__":
 worker_func('./')
# Created by pyminifier (https://github.com/liftoff/pyminifier)
