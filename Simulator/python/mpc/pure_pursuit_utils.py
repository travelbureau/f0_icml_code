import numpy as np
from numpy import genfromtxt
import numpy.linalg as LA
import scipy.interpolate as interpolate
import json,time,collections
from numba import njit

EPSILON=0.00000000001

@njit(fastmath=False,cache=True)
def get_rotation_matrix(theta):
 c,s=np.cos(theta),np.sin(theta)
 return np.array([[c,-s],[s,c]])

def nearest_point_on_line_segment(point,start,end):
 diff=start-end
 l2=np.dot(diff,diff)
 if l2==0.0:
  return start,0.0
 t=np.clip(np.dot(point-start,end-start)/l2,0.0,1.0)
 projection=start+t*(end-start)
 return projection,t

@njit(fastmath=False,cache=True)
def nearest_point_on_trajectory_py2(point,trajectory):
 diffs=trajectory[1:,:]-trajectory[:-1,:]
 l2s =diffs[:,0]**2+diffs[:,1]**2
 dots=np.empty((trajectory.shape[0]-1,))
 for i in range(dots.shape[0]):
  dots[i]=np.dot((point-trajectory[i,:]),diffs[i,:])
 t=dots/l2s
 t[t<0.0]=0.0
 t[t>1.0]=1.0
 projections=trajectory[:-1,:]+(t*diffs.T).T
 dists=np.empty((projections.shape[0],))
 for i in range(dists.shape[0]):
  temp=point-projections[i]
  dists[i]=np.sqrt(np.sum(temp*temp))
 min_dist_segment=np.argmin(dists)
 return projections[min_dist_segment],dists[min_dist_segment],t[min_dist_segment],min_dist_segment

@njit(fastmath=False,cache=True)
def first_point_on_trajectory_intersecting_circle(point,radius,trajectory,t=0.0,wrap=False):
 start_i=int(t)
 start_t=t%1.0
 first_t=None
 first_i=None
 first_p=None
 trajectory=np.ascontiguousarray(trajectory)
 for i in range(start_i,trajectory.shape[0]-1):
  start=trajectory[i,:]
  end=trajectory[i+1,:]+1e-6
  V=np.ascontiguousarray(end-start)
  a=np.dot(V,V)
  b=2.0*np.dot(V,start-point)
  c=np.dot(start,start)+np.dot(point,point)-2.0*np.dot(start,point)-radius*radius
  discriminant=b*b-4*a*c
  if discriminant<0:
   continue
  discriminant=np.sqrt(discriminant)
  t1=(-b-discriminant)/(2.0*a)
  t2=(-b+discriminant)/(2.0*a)
  if i==start_i:
   if t1>=0.0 and t1<=1.0 and t1>=start_t:
    first_t=t1
    first_i=i
    first_p=start+t1*V
    break
   if t2>=0.0 and t2<=1.0 and t2>=start_t:
    first_t=t2
    first_i=i
    first_p=start+t2*V
    break
  elif t1>=0.0 and t1<=1.0:
   first_t=t1
   first_i=i
   first_p=start+t1*V
   break
  elif t2>=0.0 and t2<=1.0:
   first_t=t2
   first_i=i
   first_p=start+t2*V
   break
 if wrap and first_p is None:
  for i in range(-1,start_i):
   start=trajectory[i%trajectory.shape[0],:]
   end=trajectory[(i+1)%trajectory.shape[0],:]+1e-6
   V=end-start
   a=np.dot(V,V)
   b=2.0*np.dot(V,start-point)
   c=np.dot(start,start)+np.dot(point,point)-2.0*np.dot(start,point)-radius*radius
   discriminant=b*b-4*a*c
   if discriminant<0:
    continue
   discriminant=np.sqrt(discriminant)
   t1=(-b-discriminant)/(2.0*a)
   t2=(-b+discriminant)/(2.0*a)
   if t1>=0.0 and t1<=1.0:
    first_t=t1
    first_i=i
    first_p=start+t1*V
    break
   elif t2>=0.0 and t2<=1.0:
    first_t=t2
    first_i=i
    first_p=start+t2*V
    break
 return first_p,first_i,first_t

@njit(fastmath=False,cache=True)
def get_actuation(pose_theta,lookahead_point,position,lookahead_distance,wheelbase):
 waypoint_y=np.dot(np.array([np.sin(-pose_theta),np.cos(-pose_theta)]),lookahead_point[0:2]-position)
 speed=lookahead_point[2]
 if np.abs(waypoint_y)<1e-6:
  return speed,0.
 radius=1/(2.0*waypoint_y/lookahead_distance**2)
 steering_angle=np.arctan(wheelbase/radius)
 return speed,steering_angle

def polar_to_euclid(coords):
 xs=ranges*np.cos(angles)
 ys=ranges*np.sin(angles)
 return(xs,ys)

def angular_deflection_magnitude(points):
 lines=np.zeros((points.shape[0]-1,3))
 thetas=np.zeros(points.shape[0]-2)
 for i in range(1,points.shape[0]):
  p0=points[i-1,:]
  p1=points[i,:]
  A=p0[1]-p1[1]
  B=p1[0]-p0[0]
  C=p0[0]*p1[1]-p1[0]*p0[1]
  lines[i-1]=(A,B,C)
 for i in range(1,lines.shape[0]):
  A1=lines[i-1,0]
  B1=lines[i-1,1]
  A2=lines[i,0]
  B2=lines[i,1]
  bottom=(A1**2+B1**2)*(A2**2+B2**2)
  if bottom>0:
   inner=(A1*A2+B1*B2)/np.sqrt(bottom)
   if np.abs(np.abs(inner)-1.0)<EPSILON:
    thetas[i-1]=0.0
   else:
    thetas[i-1]=np.arccos(inner)
 return thetas

def piecewise_linear_local_waypoints_polar(points):
 thetas=angular_deflection_magnitude(points)
 local_points_polar=np.zeros((points.shape[0]-1,2))
 for i in range(1,points.shape[0]-1):
  local_points_polar[i-1,0]=np.linalg.norm(points[i,:]-points[i-1,:])
  local_points_polar[i,1]=thetas[i-1]
 local_points_polar[-1,0]=np.linalg.norm(points[-1,:]-points[-2,:])
 return local_points_polar

class AckermannModel(object):
 def __init__(self,wheelbase):
  self.L=wheelbase

 def path_radius(self,steering_angle):
  return self.L/np.tan(steering_angle)

 def yaw_rate(self,steering_angle,speed):
  if steering_angle==0.0:
   return 0.0
  return speed/self.path_radius(steering_angle)

 def dx(self,speed,dt,steering_angle):
  if steering_angle==0.0:
   return speed*dt
  R=self.path_radius(steering_angle)
  d=dt*speed
  dx=R*np.sin(d/R)
  return dx

 def dy(self,speed,dt,steering_angle):
  if steering_angle==0.0:
   return 0.0
  R=self.path_radius(steering_angle)
  d=dt*speed
  dy=R*(1.0-np.cos(d/R))
  return dy

 def steering_angle(self,point):
  theta=np.arctan2(point[1],point[0])
  return np.arctan(2.0*self.L*np.sin(theta)/np.linalg.norm(point))

 def steering_angle_polar(self,polar_point):
  theta=polar_point[1]
  radius=polar_point[0]
  return np.arctan(2.0*self.L*np.sin(theta)/radius)
  
def max_angle(min_turning_radius,radius):
 tr2=2.0*min_turning_radius
 if radius<tr2:
  r2=radius*radius
  y=r2/(2.0*min_turning_radius)
  x=np.sqrt(r2-y*y)
  max_angle=np.arctan(y/x)
 else:
  max_angle=np.pi/2.0
 return max_angle
# Created by pyminifier (https://github.com/liftoff/pyminifier)
