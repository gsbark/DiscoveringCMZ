import warp as wp 
import numpy as np 
from torch import Tensor

@wp.struct
class MPM_state_2D:
   
   x: wp.array2d(dtype=wp.vec2f)
   v: wp.array2d(dtype=wp.vec2f)
   F: wp.array2d(dtype=wp.mat22f)
   L: wp.array2d(dtype=wp.mat22f)
   stress: wp.array2d(dtype=wp.mat22f)

   grid_v: wp.array3d(dtype=wp.vec2f) 
   grid_mv: wp.array3d(dtype=wp.vec2f) 
   grid_m: wp.array3d(dtype=wp.float32)
   grid_CMZ_force: wp.array3d(dtype=wp.vec2f)

   particles_num: wp.array(dtype=int)
   flag:wp.array(dtype=wp.float32)

   def initialize(self,steps:int,n_grid:int,num_particles:int,requires_grad:bool,device):

      self.x = wp.zeros(shape=(steps,num_particles),dtype=wp.vec2f,requires_grad=requires_grad,device=device)
      self.v = wp.zeros(shape=(steps,num_particles),dtype=wp.vec2f,requires_grad=requires_grad,device=device)
      self.F = wp.zeros(shape=(steps,num_particles),dtype=wp.mat22f,requires_grad=requires_grad,device=device)
      self.L = wp.zeros(shape=(steps,num_particles),dtype=wp.mat22f,requires_grad=requires_grad,device=device)
      self.stress = wp.zeros(shape=(steps,num_particles),dtype=wp.mat22f,requires_grad=requires_grad,device=device)

      self.grid_v = wp.zeros(shape=(steps,n_grid,n_grid),dtype=wp.vec2f,requires_grad=requires_grad,device=device)
      self.grid_mv = wp.zeros(shape=(steps,n_grid,n_grid),dtype=wp.vec2f,requires_grad=requires_grad,device=device)
      self.grid_m = wp.zeros(shape=(steps,n_grid,n_grid),dtype=wp.float32,requires_grad=requires_grad,device=device)
      self.grid_CMZ_force = wp.zeros(shape=(steps,n_grid,n_grid),dtype=wp.vec2f,requires_grad=requires_grad,device=device)

      self.particles_num = wp.zeros(shape=(1),dtype=int,requires_grad=requires_grad,device=device)
      self.flag = wp.zeros(num_particles,dtype=wp.float32,requires_grad=requires_grad,device=device)

   def reset_particles(self):   
      self.x.zero_()
      self.v.zero_()
      self.F.zero_()
      self.L.zero_()
      self.stress.zero_()
      self.particles_num.zero_()
      self.flag.zero_()   

   def reset_grid(self):
      self.grid_v.zero_()
      self.grid_m.zero_()
      self.grid_mv.zero_()
      self.grid_CMZ_force.zero_()

   def reset_substeps(self,steps):
      self.x[0].assign(self.x[-1])
      self.v[0].assign(self.v[-1])
      self.F[0].assign(self.F[-1])
      self.L[0].assign(self.L[-1])
      self.stress[0].assign(self.stress[-1])

      for i in range(1,steps):
         self.x[i].zero_()
         self.v[i].zero_()
         self.F[i].zero_()
         self.L[i].zero_()
         self.stress[i].zero_()
      self.reset_grid()

@wp.struct
class CMZ_state_2D:

   CMZ_vertices_mat1:wp.array2d(dtype=wp.vec2f)
   CMZ_vertices_mat2:wp.array2d(dtype=wp.vec2f)
   CMZ_connectivity:wp.array(dtype=wp.vec2i)
   CMZ_pairs:wp.array(dtype=wp.vec2i)

   CMZ_area:wp.array2d(dtype=wp.float32)
   CMZ_centroid_mat1:wp.array2d(dtype=wp.vec2f)
   CMZ_centroid_mat2:wp.array2d(dtype=wp.vec2f)
   CMZ_disp_jump:wp.array2d(dtype=wp.vec2f)
   CMZ_normal:wp.array2d(dtype=wp.vec2f)
   CMZ_tangent:wp.array2d(dtype=wp.vec2f)
   
   def initialize(self,steps:int,num_CMZ_vertices:int,num_CMZ_elements:int,requires_grad:bool,device):
      
      #Vertices 
      self.CMZ_vertices_mat1 = wp.zeros(shape=(steps,num_CMZ_vertices),dtype=wp.vec2f,requires_grad=requires_grad,device=device)
      self.CMZ_vertices_mat2 = wp.zeros(shape=(steps,num_CMZ_vertices),dtype=wp.vec2f,requires_grad=requires_grad,device=device)
      
      #Elements
      self.CMZ_connectivity = wp.zeros(shape=(num_CMZ_elements),dtype=wp.vec2i,requires_grad=requires_grad,device=device)
      self.CMZ_pairs = wp.zeros(shape=(num_CMZ_elements),dtype=wp.vec2i,requires_grad=requires_grad,device=device)
      self.CMZ_area = wp.zeros(shape=(steps,num_CMZ_elements),dtype=wp.float32,requires_grad=requires_grad,device=device)
      
      self.CMZ_centroid_mat1 = wp.zeros(shape=(steps,num_CMZ_elements),dtype=wp.vec2f,requires_grad=requires_grad,device=device)
      self.CMZ_centroid_mat2 = wp.zeros(shape=(steps,num_CMZ_elements),dtype=wp.vec2f,requires_grad=requires_grad,device=device)
      
      self.CMZ_disp_jump = wp.zeros(shape=(steps,num_CMZ_elements),dtype=wp.vec2f,requires_grad=requires_grad,device=device)
      self.CMZ_normal = wp.zeros(shape=(steps,num_CMZ_elements),dtype=wp.vec2f,requires_grad=requires_grad,device=device)
      self.CMZ_tangent = wp.zeros(shape=(steps,num_CMZ_elements),dtype=wp.vec2f,requires_grad=requires_grad,device=device)

   
   def reset_CMZ(self):
      self.CMZ_vertices_mat1.zero_()
      self.CMZ_vertices_mat2.zero_()
      self.CMZ_pairs.zero_()

   def assign_CMZ(self,points,point_id):
      self.reset_CMZ()
      self.CMZ_vertices_mat1[0].assign(points)
      self.CMZ_vertices_mat2[0].assign(points)

      pairs =  np.tile(np.arange(point_id.shape[0],dtype=np.int32),[2,1]).T
      self.CMZ_pairs.assign(pairs)
      self.CMZ_connectivity.assign(point_id)
      
   def reset_substeps(self,steps):
      self.CMZ_vertices_mat1[0].assign(self.CMZ_vertices_mat1[-1])
      self.CMZ_vertices_mat2[0].assign(self.CMZ_vertices_mat2[-1])
      for i in range(1,steps):
         self.CMZ_vertices_mat1[i].zero_()
         self.CMZ_vertices_mat2[i].zero_()

@wp.struct
class MPM_vars_2D:

   inv_dx:wp.float32
   dx:wp.float32
   dt:wp.float32
   p_rho:wp.float32
   p_vol:wp.float32
   p_mass:wp.float32
   l_edge:wp.float32
   n_grid:int

   length_x:wp.float32
   length_y:wp.float32
   r0:wp.float32

   E:wp.float32
   n:wp.float32
   K:wp.float32
   mu:wp.float32
   lamda:wp.float32

   gp_pos:wp.array(dtype=wp.vec2f)
   grid_nodes:wp.array2d(dtype=wp.vec2f)

   def initialize(self,Geometry,MPM_config,Material):
      
      self.n_grid = MPM_config['n_grid']
      self.length_x = Geometry['length_x']
      self.length_y = Geometry['length_y']
      self.r0 = Geometry['radius']
      self.p_rho = MPM_config['p_rho']
      self.l_edge =  Geometry['l_edge']
      self.dt = MPM_config['dt']
      self.dx = self.l_edge / float(self.n_grid)
      self.inv_dx = 1.0/ self.dx
      self.p_vol = self.dx * self.dx / float(4)
      self.p_mass = self.p_vol * self.p_rho
      
      self.E = Material['E']
      self.n = Material['n']

      self.mu = float(self.E/(2*(1+self.n)))
      self.lamda = float((self.E*self.n)/((1.0+self.n)*(1.0-2.0*self.n)))
      self.K  = float(self.E/(3.0*(1.0-2.0*self.n)))
      
      n_grid = MPM_config['n_grid']

      self.grid_nodes = wp.zeros(shape=(n_grid,n_grid),dtype=wp.vec2f,requires_grad=False)

      gauss_points = self.dx * np.array(
         [[0.5 - np.sqrt(3.0)/6,0.5 - np.sqrt(3.0)/6],
          [0.5 - np.sqrt(3.0)/6,0.5 + np.sqrt(3.0)/6],
          [0.5 + np.sqrt(3.0)/6,0.5 - np.sqrt(3.0)/6],
          [0.5 + np.sqrt(3.0)/6,0.5 + np.sqrt(3.0)/6]],
          dtype=np.float32)
      
      self.gp_pos = wp.from_numpy(gauss_points)

class wp_MLP:
   def __init__(self,params:Tensor,batch:int,steps:int,device):
      
      self.w1 = wp.array(params['layers.0.weight'].numpy(),dtype=wp.float32,requires_grad=True,device=device)
      self.b1 = wp.array(params['layers.0.bias'].numpy(),dtype=wp.float32,requires_grad=True,device=device)      
      self.out1 = wp.zeros((steps,self.w1.shape[0],batch),dtype=wp.float32,requires_grad=True,device=device)

      self.w2 = wp.array(params['layers.2.weight'].numpy(),requires_grad=True,device=device)
      self.b2 = wp.array(params['layers.2.bias'].numpy(),requires_grad=True,device=device)
      self.out2 = wp.zeros((steps,self.w2.shape[0],batch),dtype=wp.float32,requires_grad=True,device=device)

      self.w3 = wp.array(params['layers.4.weight'].numpy(),dtype=wp.float32,requires_grad=True,device=device)
      self.b3 = wp.array(params['layers.4.bias'].numpy(),dtype=wp.float32,requires_grad=True,device=device)
      self.out3 = wp.zeros((steps,self.w3.shape[0],batch),dtype=wp.float32,requires_grad=True,device=device)

      self.Traction = wp.zeros((steps,self.w3.shape[0],batch),dtype=wp.float32,requires_grad=True,device=device)

   def reset(self):

      self.out1.zero_()
      self.out2.zero_()
      self.out3.zero_()
      self.Traction.zero_()

@wp.func
def Linear(x:float):
    return x

@wp.func
def ReLu(x:float):
   return wp.max(0.0,x)

@wp.func
def softplus(x:float):
   return wp.log(1.0+ wp.exp(x))

@wp.func
def ELU(x:float):
   if x>=0:return x
   else:return wp.exp(x)-1.0
   
@wp.kernel
def scale_f(A: wp.array2d(dtype=wp.float32),scale:wp.array(dtype=wp.float32),B: wp.array2d(dtype=wp.float32)): # type:ignore

   i = wp.tid()
   B[0,i] = A[0,i]*scale[0]
   B[1,i] = A[1,i]*scale[0]

@wp.kernel
def add(A:wp.array(dtype=wp.float32),B:wp.array(dtype=wp.float32)): # type: ignore
   
   i = wp.tid()
   if wp.isnan(B[i]) or wp.isinf(B[i]):
      a = 0.0
   else:
      a = B[i]   
   wp.atomic_add(A,i,a)

@wp.kernel
def MSE_2D(
   x_pred:wp.array(dtype=wp.vec2f),   # type:ignore
   x_true:wp.array(dtype=wp.vec2f),   # type:ignore 
   loss:wp.array(dtype=wp.float32),   # type:ignore
   flag:wp.array(dtype=wp.float32),   # type:ignore
   factor:wp.float32
   ):  

   tid = wp.tid()
   loss_i = (x_pred[tid][0] - x_true[tid][0])**wp.float32(2.0) + \
            (x_pred[tid][1] - x_true[tid][1])**wp.float32(2.0)
   wp.atomic_add(loss,0,loss_i*factor)

@wp.kernel
def NN_input_2D(
   mlp_inp:wp.array2d(dtype=wp.float32),  # type:ignore
   CMZ_normal:wp.array(dtype=wp.vec2f),   # type:ignore
   CMZ_tangent:wp.array(dtype=wp.vec2f),  # type:ignore
   disp_jump:wp.array(dtype=wp.vec2f),    # type:ignore
   scale:wp.array(dtype=wp.float32)       # type:ignore
   ):
   
   tid = wp.tid()
   normal_sep = wp.dot(CMZ_normal[tid],disp_jump[tid])
   tangent_sep = wp.dot(CMZ_tangent[tid],disp_jump[tid])
   mlp_inp[0,tid] = normal_sep * scale[0]
   mlp_inp[1,tid] = tangent_sep * scale[1]


@wp.kernel
def run_mlp(
   w1:wp.array2d(dtype=wp.float32),    # type:ignore
   b1:wp.array(dtype=wp.float32),      # type:ignore
   w2:wp.array2d(dtype=wp.float32),    # type:ignore
   b2:wp.array(dtype=wp.float32),      # type:ignore
   w3:wp.array2d(dtype=wp.float32),    # type:ignore
   b3:wp.array(dtype=wp.float32),      # type:ignore
   inp:wp.array2d(dtype=wp.float32),   # type:ignore
   out1:wp.array2d(dtype=wp.float32),  # type:ignore
   out2:wp.array2d(dtype=wp.float32),  # type:ignore
   out3:wp.array2d(dtype=wp.float32)   # type:ignore
   ):

   tid = wp.tid()
   wp.mlp(w1,b1,ELU,tid,inp,out1)
   wp.mlp(w2,b2,ELU,tid,out1,out2)
   wp.mlp(w3,b3,ReLu,tid,out2,out3)

@wp.func
def Circle_domain(x:wp.vec2,MPM:MPM_vars_2D):
   flag = 0.0
   ls = 0.0
   x1 = 1.0/2.0 * (MPM.l_edge - MPM.length_x)
   x2 = MPM.l_edge - 1.0/2.0 * (MPM.l_edge - MPM.length_x)
   y1 = 1.0/2.0 * (MPM.l_edge - MPM.length_y)
   y2 = MPM.l_edge - 1.0/2.0 * (MPM.l_edge - MPM.length_y)
   cx = x2
   cy = y2
   rr = wp.sqrt((x[0] - cx) **2.0 + (x[1] - cy) **2.0)    
   if x[0] > x1 and x[0] < x2 and x[1] > y1 and x[1] < y2:
      ls = 1.0 # material 1
      if rr < MPM.r0: 
         ls = -1.0 # material 2
         if rr<MPM.r0-2.0*MPM.dx:
            flag = 0.0
         else:
            flag =1.0
   elif x[0]>x2 and x[0]<x2+ 5.0*MPM.dx and x[1]<y2 and x[1]> y2-MPM.r0:
      ls = -1.0
   return ls,flag

@wp.func
def DC_Beam(x:wp.vec2,MPM:MPM_vars_2D):
   flag = 0.0
   ls = 0.0
   x1 = 1.0/2.0 * (MPM.l_edge - MPM.length_x)
   x2 = MPM.l_edge - 1.0/2.0 * (MPM.l_edge - MPM.length_x)
   
   y1 = 1.0/2.0 * (MPM.l_edge - MPM.length_y)
   y2 = MPM.l_edge - 1.0/2.0 * (MPM.l_edge - MPM.length_y)
     
   if x[0] > x1 and x[0] < x2 and x[1] > y1 and x[1] < y2:
      ls = 1.0 # material 1
      flag = 1.0
      if x[1] > (y1+y2)/2.0:
         ls = -1.0 # material 2
         flag = 1.0
   
   elif x[0]>x1 and x[0]<x1+10.0*MPM.dx and x[1] > y2 and x[1] < y2+10.0*MPM.dx:
      ls = -1.0 # material 2
      flag = 1.0
   elif x[0]>x1 and x[0]<x1+10.0*MPM.dx and x[1] < y1 and x[1] > y1-10.0*MPM.dx:
      ls = 1.0 # material 1
      flag = 1.0
   return ls,flag

@wp.kernel
def geometry_2D(
   MPM:MPM_vars_2D,
   Material_1:MPM_state_2D,
   Material_2:MPM_state_2D,
   n_grid:int,
   example:int,
   ):
   
   for i in range(n_grid):
      for j in range(n_grid):
            xx = wp.vec2(MPM.dx*float(i),MPM.dx*float(j))
            MPM.grid_nodes[i,j] = xx
            for p in range(4):
               if example==1:ls,flag = DC_Beam(xx+MPM.gp_pos[p],MPM)
               elif example==2:ls,flag = Circle_domain(xx+MPM.gp_pos[p],MPM)
               if ls >0.5:
                  Material_1.x[0,Material_1.particles_num[0]] = xx + MPM.gp_pos[p]
                  Material_1.v[0,Material_1.particles_num[0]] = wp.vec2(0.0,0.0)
                  Material_1.F[0,Material_1.particles_num[0]] = wp.mat22(1.0,0.0,0.0,1.0)
                  Material_1.L[0,Material_1.particles_num[0]] = wp.mat22(0.0,0.0,0.0,0.0)
                  Material_1.flag[Material_1.particles_num[0]] = flag
                  wp.atomic_add(Material_1.particles_num,0,1)
               elif ls <-0.5:
                  Material_2.x[0,Material_2.particles_num[0]] = xx + MPM.gp_pos[p]
                  Material_2.v[0,Material_2.particles_num[0]] = wp.vec2(0.0,0.0)
                  Material_2.F[0,Material_2.particles_num[0]] = wp.mat22(1.0,0.0,0.0,1.0)
                  Material_2.L[0,Material_2.particles_num[0]] = wp.mat22(0.0,0.0,0.0,0.0)
                  Material_2.flag[Material_2.particles_num[0]] = flag
                  wp.atomic_add(Material_2.particles_num,0,1)
