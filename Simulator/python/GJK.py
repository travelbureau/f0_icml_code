import numpy as np
import numba

@numba.njit(cache=True)
def perpendicular(pt):
 temp=pt[0]
 pt[0]=pt[1]
 pt[1]=-1*temp
 return pt

@numba.njit(cache=True)
def tripleProduct(a,b,c):
 ac=a.dot(c)
 bc=b.dot(c)
 return b*ac-a*bc

@numba.njit(cache=True)
def avgPoint(vertices):
 return np.sum(vertices,axis=0)/vertices.shape[0]

@numba.njit(cache=True)
def indexOfFurthestPoint(vertices,d):
 return np.argmax(vertices.dot(d))

@numba.njit(cache=True)
def support(vertices1,vertices2,d):
 i=indexOfFurthestPoint(vertices1,d)
 j=indexOfFurthestPoint(vertices2,-d)
 return vertices1[i]-vertices2[j]

@numba.njit(cache=True)
def collision(vertices1,vertices2):
 index=0
 simplex=np.empty((3,2))
 position1=avgPoint(vertices1)
 position2=avgPoint(vertices2)
 d=position1-position2
 if d[0]==0 and d[1]==0:
  d[0]=1.0
 a=support(vertices1,vertices2,d)
 simplex[index,:]=a
 if d.dot(a)<=0:
  return 0
 d=-a
 iter_count=0
 while iter_count<1e3:
  a=support(vertices1,vertices2,d)
  index+=1
  simplex[index,:]=a
  if d.dot(a)<=0:
   return 0
  ao=-a
  if index<2:
   b=simplex[0,:]
   ab=b-a
   d=tripleProduct(ab,ao,ab)
   if np.linalg.norm(d)<1e-10:
    d=perpendicular(ab)
   continue
  b=simplex[1,:]
  c=simplex[0,:]
  ab=b-a
  ac=c-a
  acperp=tripleProduct(ab,ac,ac)
  if acperp.dot(ao)>=0:
   d=acperp
  else:
   abperp=tripleProduct(ac,ab,ab)
   if abperp.dot(ao)<0:
    return 1
   simplex[0,:]=simplex[1,:]
   d=abperp
  simplex[1,:]=simplex[2,:]
  index-=1
  iter_count+=1
 assert(1==0)
 return 0

@numba.njit(cache=True)
def collision_dist(vertices1,vertices2):
 index=0
 simplex=np.empty((3,2))
 position1=avgPoint(vertices1)
 position2=avgPoint(vertices2)
 d=position1-position2
 if d[0]==0 and d[1]==0:
  d[0]=1.0
 a=support(vertices1,vertices2,d)
 simplex[index,:]=a
 if d.dot(a)<=0:
  return distance(vertices1,vertices2,d)
 d=-a
 iter_count=0
 while iter_count<1e3:
  a=support(vertices1,vertices2,d)
  index+=1
  simplex[index,:]=a
  if d.dot(a)<=0:
   return distance(vertices1,vertices2,d)
  ao=-a
  if index<2:
   b=simplex[0,:]
   ab=b-a
   d=tripleProduct(ab,ao,ab)
   if np.linalg.norm(d)<1e-10:
    d=perpendicular(ab)
   continue
  b=simplex[1,:]
  c=simplex[0,:]
  ab=b-a
  ac=c-a
  acperp=tripleProduct(ab,ac,ac)
  if acperp.dot(ao)>=0:
   d=acperp
  else:
   abperp=tripleProduct(ac,ab,ab)
   if abperp.dot(ao)<0:
    return 0.0 
   simplex[0,:]=simplex[1,:]
   d=abperp
  simplex[1,:]=simplex[2,:]
  index-=1
  iter_count+=1
 assert(1==0)
 return distance(vertices1,vertices2,d)

@numba.njit(cache=True)
def distance(vertices1,vertices2,direc):
 a=support(vertices1,vertices2,direc)
 b=support(vertices1,vertices2,-direc)
 d=closestPoint2Origin(a,b)
 dist=np.linalg.norm(d)
 while True:
  if dist<1e-10:
   return dist
  d=-d
  c=support(vertices1,vertices2,d)
  temp1=c.dot(d)
  temp2=a.dot(d)
  if(temp1-temp2)<1e-10:
   return dist
  p1=closestPoint2Origin(a,c)
  p2=closestPoint2Origin(c,b)
  dist1=np.linalg.norm(p1)
  dist2=np.linalg.norm(p2)
  if dist1<dist2:
   b=c
   d=p1
   dist=dist1
  else:
   a=c
   d=p2
   dist=dist2

@numba.njit(cache=True)
def closestPoint2Origin(a,b):
 ab=b-a
 ao=-a
 length=ab.dot(ab)
 if length<1e-10:
  return a
 frac=ao.dot(ab)/length
 if frac<0:
  return a
 if frac>1:
  return b
 return frac*ab+a
# Created by pyminifier (https://github.com/liftoff/pyminifier)
