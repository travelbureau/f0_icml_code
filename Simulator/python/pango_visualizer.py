import numpy as np
from PIL import Image
import OpenGL.GL as gl
import pypangolin as pangolin
import transformations
import yaml
import msgpack
from mpc import pure_pursuit_utils
import time

class PangoViz(object):
 def __init__(self,worker_dir,map_img_path,map_yaml_path,waypoints,show_laser=False):
  with open(map_yaml_path,'r')as yaml_stream:
   try:
    map_metadata=yaml.safe_load(yaml_stream)
    map_resolution=map_metadata['resolution']
    origin=map_metadata['origin']
    map_origin_x=origin[0]
    map_origin_y=origin[1]
   except yaml.YAMLError as ex:
    print(ex)
  with open(worker_dir+'config.yaml','r')as yaml_stream:
   try:
    config=yaml.safe_load(yaml_stream)
    speed_lut_name=config['speed_lut_name']
    zoom=config['zoom']
   except yaml.YAMLError as ex:
    print(ex)

  self.speed_lut=msgpack.unpack(open(worker_dir+speed_lut_name,'rb'),use_list=False)
  self.waypoints=waypoints
  self.waypoints_plot=np.copy(waypoints[:,0:3])
  self.waypoints_plot[:,2]*=0.
  self.show_laser=show_laser
  self.map_img=np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float64)
  if len(self.map_img.shape)>2:
   print('map image not grayscale')
   self.map_img=np.dot(self.map_img[...,:3],[0.29,0.57,0.14])
   self.map_img=np.floor(self.map_img)
  map_height=self.map_img.shape[0]
  map_width=self.map_img.shape[1]
  range_x=np.arange(self.map_img.shape[1])
  range_y=np.arange(self.map_img.shape[0])
  map_x,map_y=np.meshgrid(range_x,range_y)
  map_x=(map_x*map_resolution+map_origin_x).flatten()
  map_y=(map_y*map_resolution+map_origin_y).flatten()
  map_z=np.zeros(map_y.shape)
  map_coords=np.vstack((map_x,map_y,map_z))
  map_mask=self.map_img==0.0
  map_mask_flat=map_mask.flatten()
  self.map_points=map_coords[:,map_mask_flat].T
  pangolin.CreateWindowAndBind('sim',930,1080)
  gl.glEnable(gl.GL_DEPTH_TEST)
  self.scam=pangolin.OpenGlRenderState(pangolin.ProjectionMatrix(640,480,120,120,320,280,0.2,200),pangolin.ModelViewLookAt(-0.1,0,zoom,0,0,0,pangolin.AxisDirection.AxisZ))
  self.handler=pangolin.Handler3D(self.scam)
  self.dcam=pangolin.CreateDisplay()
  self.dcam.SetBounds(pangolin.Attach(0.0),pangolin.Attach(1.0),pangolin.Attach(0.0),pangolin.Attach(1.0),-1920.0/1080)
  self.dcam.SetHandler(self.handler)
  angle_min=-4.7/2
  angle_max=4.7/2
  num_beams=1080
  self.scan_angles=np.linspace(angle_min,angle_max,num_beams)

 def update(self,obs,ego_grid,opp_grid,op_all_states,op_picked_state,ego_all_states,ego_picked_state,is_corner,ego_xy_grid):
  gl.glClear(gl.GL_COLOR_BUFFER_BIT|gl.GL_DEPTH_BUFFER_BIT)
  gl.glClearColor(37/255,37/255,38/255,1.0)
  self.dcam.Activate(self.scam)
  ego_x=obs['poses_x'][0]
  ego_y=obs['poses_y'][0]
  ego_theta=obs['poses_theta'][0]
  op_x=obs['poses_x'][1]
  op_y=obs['poses_y'][1]
  op_theta=obs['poses_theta'][1]
  ego_pose=transformations.rotation_matrix(ego_theta,(0,0,1))
  ego_pose[0,3]=ego_x
  ego_pose[1,3]=ego_y
  op_pose=transformations.rotation_matrix(op_theta,(0,0,1))
  op_pose[0,3]=op_x
  op_pose[1,3]=op_y
  ego_size=np.array([0.58,0.31,0.1])
  op_size=np.array([0.58,0.31,0.1])
  gl.glLineWidth(1)
  gl.glColor3f(1.0,1.0,1.0)
  pangolin.DrawBoxes([ego_pose],[ego_size])
  gl.glColor(231/256.,34/256.,46/256.)
  pangolin.DrawBoxes([op_pose],[op_size])
  gl.glPointSize(2)
  gl.glColor3f(0.2,0.2,0.2)
  pangolin.DrawPoints(self.map_points)
  gl.glPointSize(2)
  gl.glColor3f(0.0,0.5,1.0)
  if ego_xy_grid is None:
   pangolin.FinishFrame()
   return
   
  gl.glPointSize(2)
  gl.glColor3f(0.0,0.5,1.0)
  rot=np.array([[np.cos(ego_theta),np.sin(ego_theta)],[-np.sin(ego_theta),np.cos(ego_theta)]])
  xy_grid=np.dot(ego_xy_grid[:,:2],rot)
  temp=np.hstack([xy_grid,np.zeros((xy_grid.shape[0],1))])
  if self.show_laser:
   rot_mat=transformations.rotation_matrix(ego_theta,(0,0,1))
   ego_scan=obs['scans'][0]
   ego_scan=np.asarray(ego_scan)
   ego_scan_x=np.multiply(ego_scan,np.sin(self.scan_angles))
   ego_scan_y=np.multiply(ego_scan,np.cos(self.scan_angles))
   ego_scan_arr=np.zeros((ego_scan_x.shape[0],3))
   ego_scan_arr[:,0]=ego_scan_y
   ego_scan_arr[:,1]=ego_scan_x
   ego_scan_arr=np.dot(rot_mat[0:3,0:3],ego_scan_arr.T)
   ego_scan_arr=ego_scan_arr+np.array([[ego_x],[ego_y],[0]])
   gl.glPointSize(1)
   gl.glColor3f(1.0,0.0,0.0)
   pangolin.DrawPoints(ego_scan_arr.T)
  if ego_all_states is not None:
   gl.glPointSize(2)
   gl.glColor3f(0.8,0.0,0.5)
  if op_picked_state is not None:
   gl.glPointSize(3)
   if op_all_states is None:
    gl.glColor3f(231/256.,34/256.,46/256.)
   else:
    gl.glColor3f(231/256.,34/256.,46/256.)
   pangolin.DrawPoints(np.hstack([op_picked_state[:,0:2],np.zeros((op_picked_state.shape[0],1))]))
  if ego_picked_state is not None:
   gl.glPointSize(5)
   if ego_all_states is None:
    gl.glColor3f(1.,1.,1.)
   else:
    gl.glColor3f(1.,1.,1.)
   pangolin.DrawPoints(np.hstack([ego_picked_state[:,0:2],np.zeros((ego_picked_state.shape[0],1))]))
  gl.glPointSize(2)
  gl.glColor3f(22/256.,88/256.,142/256.)
  pangolin.DrawPoints(self.waypoints_plot)
  pangolin.FinishFrame()
# Created by pyminifier (https://github.com/liftoff/pyminifier)
