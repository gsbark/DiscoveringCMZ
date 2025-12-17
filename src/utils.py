import numpy as np 
import torch 
from src.wp_utils import wp_MLP
import torch.nn as nn


#Create CZM
#--------------------------------
def CZ_circle(num_elements,Geometry):
   
   length_y = Geometry['length_y']
   length_x = Geometry['length_x']
   l_edge = Geometry['l_edge']
   radius = Geometry['radius']

   num_points = num_elements + 1

   points = np.zeros((num_points,2))
   centroid_id = np.zeros((num_points-1,2),dtype=np.int32)
   angle_increment = (np.pi/2) / num_points

   x2 = l_edge - 1.0/2.0 * (l_edge-length_x)
   y2 = l_edge - 1.0/2.0 * (l_edge - length_y)

   center = (x2,y2)
   #center = (l_edge/2, l_edge/2)
   for i in range(num_points):
      angle = 2*np.pi - i * angle_increment + 3*np.pi/2
      x = center[0] + radius * np.cos(angle)
      y = center[1] + radius * np.sin(angle)
      points[i] = np.array([x,y])
      if i <num_points-1:centroid_id[i] = np.array([i,i+1])
      #else: centroid_id[i] = np.array([i,0])
   return points,centroid_id

def CZ_two(num_points,Geometry):
   
   length_y = Geometry['length_y']
   length_x = Geometry['length_x']
   l_edge = Geometry['l_edge']
   
   #points = np.zeros((num_points,2))
   centroid_id = np.zeros((num_points-1,2),dtype=np.int32)
   
   x2 = l_edge - 1.0/2.0 * (l_edge-length_x)
   x1 = 1.0/2.0 * (l_edge - length_x) + 10.0*l_edge/128.0

   center=(l_edge/2, l_edge/2)
   #points = np.linspace(x1,x2,num_points)
   points_x = np.linspace(x1,x2,num_points)[::-1]
   points_y = np.ones_like(points_x)*center[1]
   points = np.stack((points_x,points_y),axis=1)
   centroid_id = np.stack((np.arange(0,num_points)[:-1],np.arange(0,num_points)[1:]),axis=1).astype(np.int32)

   return points,centroid_id
#--------------------------------

#Plot traction surface
def find_surface(wp_mlp:wp_MLP,torch_MLP:nn.Module,scale_inp,scale_out):
   
    model = torch_MLP
    scale_x = torch.tensor(scale_inp.numpy())
    scale_y = torch.tensor(scale_out.numpy())

    w1 = wp_mlp.w1.numpy()
    w2 = wp_mlp.w2.numpy()
    w3 = wp_mlp.w3.numpy()

    model.layers[0].weight.data = torch.tensor(w1)
    model.layers[0].bias.data = torch.tensor(wp_mlp.b1.numpy())
    model.layers[2].weight.data = torch.tensor(w2)
    model.layers[2].bias.data = torch.tensor(wp_mlp.b2.numpy())
    model.layers[4].weight.data = torch.tensor(w3)
    model.layers[4].bias.data = torch.tensor(wp_mlp.b3.numpy())

    Dn = np.linspace(0,0.02,50)
    Dt = np.linspace(0,0.02,50)

    X,Y = np.meshgrid(Dn,Dt)
    input = torch.tensor(np.stack((X.reshape(-1,),Y.reshape(-1,)),axis=1),dtype=torch.float32,requires_grad=False)
    with torch.no_grad():
        ip = input*scale_x
        out = (model(ip)*scale_y).numpy()
    return out[:,0].reshape(50,50),out[:,1].reshape(50,50)
   
