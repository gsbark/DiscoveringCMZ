import warp as wp 
from src.wp_utils import*

@wp.func
def Needleman_exp(Delta_n:wp.float32,Delta_t:wp.float32):
    
   phi_n = 30.0 * 1e-3
   phi_t = 30.0 * 1e-3

   s_max = 30.0
   t_max = 30.0

   d_n = phi_n/(s_max*wp.exp(1.0))
   d_t = phi_t/(wp.sqrt(wp.exp(1.0)/2.0)*t_max)
   Delta_n_star = 0.0
   
   exp_t = wp.exp(-(Delta_t**2.0/d_t**2.0))
   exp_n = wp.exp(-Delta_n/d_n)
   q = phi_t/phi_n
   r = Delta_n_star/d_n

   T_n = phi_n/d_n*exp_n*(Delta_n/d_n*exp_t+(1.0-q)/(r-1.0)*(1.0-exp_t)*(r-Delta_t/d_n))
   T_t = phi_n/d_n*2.0*d_n/d_t*Delta_t/d_t*(q+(r-q)/(r-1.0)*Delta_n/d_n)*exp_n*exp_t

   return T_n,T_t

@wp.func
def Tvergaard(Delta_n:wp.float32,Delta_t:wp.float32):
    
   Tmax = 25.0
   d_n = 0.002
   d_t = 0.002

   D_bar = wp.sqrt((Delta_n/d_n)**2.0+(Delta_t/d_t)**2.0)
   D_bar_out = wp.min(D_bar,1.0)

   T = 27.0/4.0 * Tmax*(D_bar_out)*(1.0-2.0*D_bar_out+D_bar_out**2.0)
   if Delta_n<0.0:
      T = -5000.0*D_bar
   return T,0.0

@wp.func
def safe_sqrt(x:float):
   if x>0.0:
      return wp.sqrt(x)
   else: return 0.0

@wp.func
def Identity2():
   v = wp.vec2f(wp.float32(1.0),wp.float32(1.0))
   return wp.diag(v)

@wp.func
def stress_update_2D(F:wp.mat22f,E:float,n:float):
    
   lamda = (E * n) /( (1.0+n)*(1.0-2.0*n))
   mu = E /( 2.0 * (1.0+n))
   Ee = 0.5*(wp.transpose(F)@F - Identity2())
   S = lamda* wp.trace(Ee)*Identity2() + 2.0*mu * Ee
   J = wp.determinant(F)
   stress = 1.0/J * (F @ S @ wp.transpose(F))
   return stress

@wp.func
def kernel_weights_2D(x:wp.vec2f,inv_dx:float):

   base_x = int(x[0]*inv_dx-0.5)
   base_y = int(x[1]*inv_dx-0.5)

   f_x = x[0]*inv_dx - wp.float32(base_x)
   f_y = x[1]*inv_dx - wp.float32(base_y)

   w = wp.mat(
      0.5 *(1.5-f_x)**2.0, 0.5 *(1.5-f_y)**2.0,
      0.75-(f_x-1.0)**2.0, 0.75-(f_y-1.0)**2.0,
      0.5 *(f_x-0.5)**2.0, 0.5 *(f_y-0.5)**2.0,
      shape=(3,2))

   return w,wp.vec2i(base_x,base_y)

@wp.kernel
def p2g_2D(
   MPM:MPM_vars_2D,
   x:wp.array(dtype=wp.vec2f),
   v:wp.array(dtype=wp.vec2f),
   L:wp.array(dtype=wp.mat22f),
   F:wp.array(dtype=wp.mat22f),
   stress:wp.array(dtype=wp.mat22f),
   grid_m:wp.array2d(dtype=wp.float32),
   grid_mv:wp.array2d(dtype=wp.vec2f)
   ):
   
   tid = wp.tid()
   base_x = int(x[tid][0]*MPM.inv_dx-0.5)
   base_y = int(x[tid][1]*MPM.inv_dx-0.5)

   f_x = x[tid][0]*MPM.inv_dx - wp.float32(base_x)
   f_y = x[tid][1]*MPM.inv_dx - wp.float32(base_y)

   w = wp.mat(
      0.5 *(1.5-f_x)**2.0, 0.5 *(1.5-f_y)**2.0,
      0.75-(f_x-1.0)**2.0, 0.75-(f_y-1.0)**2.0,
      0.5 *(f_x-0.5)**2.0, 0.5 *(f_y-0.5)**2.0,
      shape=(3,2))

   St = stress_update_2D(F[tid],MPM.E,MPM.n)
   stress[tid] = St       
   Jdet = wp.determinant(F[tid])
   MLS_stress = (-MPM.dt*MPM.p_vol*Jdet*4.0*MPM.inv_dx*MPM.inv_dx)*St
   MLS_affine = MLS_stress + MPM.p_mass*L[tid]

   for i in range(3):
      for j in range(3):
            offset = wp.vec2(float(i), float(j))
            dpos =  wp.vec2((offset[0] - f_x) * MPM.dx, (offset[1] - f_y) * MPM.dx)
            weight = w[i,0] * w[j,1]
            momentum =  weight * (MPM.p_mass* v[tid] + MLS_affine @ dpos)
            mass = weight * MPM.p_mass
            wp.atomic_add(grid_mv,i=base_x+i ,j=base_y+j,value=momentum)
            wp.atomic_add(grid_m,i=base_x+i ,j=base_y+j,value=mass)

@wp.kernel
def grid_update_2D(
   MPM:MPM_vars_2D,
   Grid1_mv:wp.array2d(dtype=wp.vec2f),
   Grid1_m:wp.array2d(dtype=wp.float32),
   Grid1_v:wp.array2d(dtype=wp.vec2f),
   Grid1_CMZ_f:wp.array2d(dtype=wp.vec2f),
   Grid2_mv:wp.array2d(dtype=wp.vec2f),
   Grid2_m:wp.array2d(dtype=wp.float32),
   Grid2_v:wp.array2d(dtype=wp.vec2f),
   Grid2_CMZ_f:wp.array2d(dtype=wp.vec2f)
   ):

   i,j = wp.tid()
   Grid1_mv[i,j] += MPM.dt * Grid1_CMZ_f[i,j]
   Grid2_mv[i,j] +=  MPM.dt * Grid2_CMZ_f[i,j]

   if Grid1_m[i,j]>0.0:
      Grid1_v[i,j] = Grid1_mv[i,j]/Grid1_m[i,j]
      
   if Grid2_m[i,j]>0.0:
      Grid2_v[i,j] = Grid2_mv[i,j]/Grid2_m[i,j]
      
@wp.kernel
def BCs_ex1(
   MPM:MPM_vars_2D,
   Grid1_m:wp.array2d(dtype=wp.float32),
   Grid1_v:wp.array2d(dtype=wp.vec2f),
   Grid2_m:wp.array2d(dtype=wp.float32),
   Grid2_v:wp.array2d(dtype=wp.vec2f)
   ):

   i,j = wp.tid()
   y1 = 1.0/2.0 * (MPM.l_edge - MPM.length_y)
   y2 = MPM.l_edge - 1.0/2.0 * (MPM.l_edge - MPM.length_y)
   
   x1 = 1.0/2.0 * (MPM.l_edge - MPM.length_x)
   x2 = MPM.l_edge - 1.0/2.0 * (MPM.l_edge - MPM.length_x)

   if Grid1_m[i,j]>0.0:
      dist_x = MPM.dx * wp.float32(i)
      dist_y = MPM.dx * wp.float32(j)
      if dist_x < x1 + 15.0*MPM.dx and dist_y<y1-5.0*MPM.dx: 
         Grid1_v[i, j][1] = -5.0 
      
      if dist_x > MPM.l_edge - 0.5 * (MPM.l_edge - MPM.length_x): 
         Grid1_v[i, j][0] = 0.0
         Grid1_v[i, j][1] = 0.0

   if Grid2_m[i,j]>0.0:
      #Boundary conditions
      dist_x = MPM.dx * wp.float32(i)
      dist_y = MPM.dx * wp.float32(j)
      
      if dist_x < x1 + 15.0*MPM.dx and dist_y>y2+5.0*MPM.dx: 
         Grid2_v[i, j][1] = 5.0 
      
      if dist_x > MPM.l_edge - 0.5 * (MPM.l_edge - MPM.length_x): 
         Grid2_v[i, j][0] = 0.0
         Grid2_v[i, j][1] = 0.0
   
@wp.kernel
def BCs_ex2(
   MPM:MPM_vars_2D,
   Grid1_m:wp.array2d(dtype=wp.float32),
   Grid1_v:wp.array2d(dtype=wp.vec2f),
   Grid2_m:wp.array2d(dtype=wp.float32),
   Grid2_v:wp.array2d(dtype=wp.vec2f)
   ):

   i,j = wp.tid()
   #Boundary conditions
   dist_x = MPM.dx * wp.float32(i)
   if Grid1_m[i,j]>0.0:
      if dist_x < 0.5 * (MPM.l_edge - MPM.length_x) + 2.0*MPM.dx:
         Grid1_v[i, j][0] = -5.0 
   if Grid2_m[i,j]>0.0:
      if dist_x > MPM.l_edge - 1.0/2.0 * (MPM.l_edge - MPM.length_x) + 1.0*MPM.dx: 
         Grid2_v[i, j][0] = 5.0

@wp.kernel
def g2p_2D(
   MPM:MPM_vars_2D,
   x:wp.array(dtype=wp.vec2f),
   F:wp.array(dtype=wp.mat22f),
   x_new:wp.array(dtype=wp.vec2f),
   v_new:wp.array(dtype=wp.vec2f),
   L_new:wp.array(dtype=wp.mat22f),
   F_new:wp.array(dtype=wp.mat22f),
   grid_v:wp.array2d(dtype=wp.vec2f)
   ):
   
   tid = wp.tid()
   base_x = int(x[tid][0]*MPM.inv_dx-0.5)
   base_y = int(x[tid][1]*MPM.inv_dx-0.5)

   f_x = x[tid][0]*MPM.inv_dx - wp.float32(base_x)
   f_y = x[tid][1]*MPM.inv_dx - wp.float32(base_y)

   w = wp.mat(
      0.5 *(1.5-f_x)**2.0, 0.5 *(1.5-f_y)**2.0,
      0.75-(f_x-1.0)**2.0, 0.75-(f_y-1.0)**2.0,
      0.5 *(f_x-0.5)**2.0, 0.5 *(f_y-0.5)**2.0,
      shape=(3,2))
   
   new_v = wp.vec2(0.0,0.0)
   new_L = wp.mat22(0.0,0.0,0.0,0.0)

   for i in range(3):
      for j in range(3):
            weight = w[i][0] * w[j][1]
            g_v = grid_v[base_x + i,base_y + j]
            new_v = new_v + weight * g_v
            dpos =  wp.vec2((float(i)- f_x) * MPM.dx, (float(j) - f_y) * MPM.dx)
            new_L =  new_L + (4.0*weight*MPM.inv_dx*MPM.inv_dx)*wp.outer(g_v,dpos)

   v_new[tid] = new_v
   L_new[tid] = new_L
   x_new[tid] = x[tid] + MPM.dt * new_v
   F_new[tid] = F[tid] + MPM.dt * (new_L @ F[tid])

@wp.kernel
def CMZ_update_2D(
   MPM:MPM_vars_2D,
   Grid_v:wp.array2d(dtype=wp.vec2f),
   Grid_m:wp.array2d(dtype=wp.float32),
   CMZ_vertices:wp.array(dtype=wp.vec2f),
   CMZ_vertices_new:wp.array(dtype=wp.vec2f)
   ):

   p = wp.tid()
   base_x = int(CMZ_vertices[p][0]*MPM.inv_dx-0.5)
   base_y = int(CMZ_vertices[p][1]*MPM.inv_dx-0.5)
   
   f_x = CMZ_vertices[p][0]*MPM.inv_dx - wp.float32(base_x)
   f_y = CMZ_vertices[p][1]*MPM.inv_dx - wp.float32(base_y)
   
   w = wp.mat(0.5 *(1.5-f_x)**2.0, 0.5 *(1.5-f_y)**2.0,
              0.75-(f_x-1.0)**2.0, 0.75-(f_y-1.0)**2.0,
              0.5 *(f_x-0.5)**2.0, 0.5 *(f_y-0.5)**2.0,shape=(3,2))
   
   new_v1 = wp.vec2(0.0,0.0)
   sum_weight = 0.0
   for i in range(3):
      for j in range(3):
            weight = w[i][0] * w[j][1]
            g_v1 = wp.vec2(0.0,0.0)
            if Grid_m[base_x + i,base_y + j]>0.0:
               g_v1 = Grid_v[base_x + i,base_y + j]
               sum_weight += weight
            new_v1 += weight* g_v1
   CMZ_vertices_new[p] = CMZ_vertices[p]+ MPM.dt * new_v1/sum_weight

@wp.kernel
def CMZ_separation_2D(
   CMZ_vertices_mat1:wp.array(dtype=wp.vec2f),
   CMZ_vertices_mat2:wp.array(dtype=wp.vec2f),
   CMZ_conn:wp.array(dtype=wp.vec2i),
   CMZ_pair:wp.array(dtype=wp.vec2i),
   CMZ_centroid_mat1:wp.array(dtype=wp.vec2f),
   CMZ_centroid_mat2:wp.array(dtype=wp.vec2f),
   CMZ_area:wp.array(dtype=wp.float32),
   CMZ_disp_jump:wp.array(dtype=wp.vec2f),
   CMZ_normal:wp.array(dtype=wp.vec2f),
   CMZ_tangent:wp.array(dtype=wp.vec2f)
   ):
   
   p = wp.tid()
   index_mat1 = CMZ_pair[p][0]
   index_mat2 = CMZ_pair[p][1]
   
   centroid_mat1 = wp.vec2f()
   centroid_mat2 = wp.vec2f()
   for i in range(2):
      ipoint_mat1 = CMZ_conn[index_mat1][i]
      ipoint_mat2 = CMZ_conn[index_mat2][i]
      centroid_mat1+=CMZ_vertices_mat1[ipoint_mat1]/2.0
      centroid_mat2+=CMZ_vertices_mat2[ipoint_mat2]/2.0
   
   delta_mat2 = CMZ_vertices_mat2[CMZ_conn[index_mat2][1]] - CMZ_vertices_mat2[CMZ_conn[index_mat2][0]]
   normal_mat2 = wp.normalize(wp.vec2(-delta_mat2[1],delta_mat2[0]))
   tangent_mat2 = wp.normalize(wp.vec2(delta_mat2[0],delta_mat2[1]))
   
   area = safe_sqrt(delta_mat2[0]**2.0+delta_mat2[1]**2.0)
   separation = centroid_mat1-centroid_mat2

   CMZ_centroid_mat1[p] = centroid_mat1
   CMZ_centroid_mat2[p] = centroid_mat2
   CMZ_area[p] = area
   CMZ_disp_jump[p] = separation
   CMZ_normal[p] = normal_mat2
   CMZ_tangent[p] = tangent_mat2

@wp.kernel
def CMZ_Force_updateNN_2D(
   MPM:MPM_vars_2D,
   CMZ_centroid_mat1:wp.array(dtype=wp.vec2f),
   CMZ_centroid_mat2:wp.array(dtype=wp.vec2f),
   CMZ_area:wp.array(dtype=wp.float32),
   CMZ_normal:wp.array(dtype=wp.vec2f),
   CMZ_tangent:wp.array(dtype=wp.vec2f),
   NN_traction:wp.array2d(dtype=wp.float32),
   grid1_m:wp.array2d(dtype=wp.float32),
   grid2_m:wp.array2d(dtype=wp.float32),
   grid1_CMZ_f:wp.array2d(dtype=wp.vec2f),
   grid2_CMZ_f:wp.array2d(dtype=wp.vec2f),
   case:int
   ):
   
   p = wp.tid() 

   if case==1:
      separation = CMZ_centroid_mat1[p]-CMZ_centroid_mat2[p]
      Delta_n = wp.dot(separation,CMZ_normal[p])
      Delta_t = wp.dot(separation,CMZ_tangent[p])
      D_bar = safe_sqrt((Delta_n/0.002)**2.0+(Delta_t/0.002)**2.0)
      c = 5000.0*D_bar
      T = (NN_traction[0,p]) * (wp.sign(Delta_n)+1.0)/2.0 + c * (wp.sign(Delta_n)-1.0)/2.0
      traction = T*CMZ_normal[p] 
   
   elif case==2:
      traction = NN_traction[0,p]*CMZ_normal[p] + NN_traction[1,p]*CMZ_tangent[p]
   
   force = traction*CMZ_area[p]
   w_mat1,base_mat1 = kernel_weights_2D(CMZ_centroid_mat1[p],MPM.inv_dx)
   w_mat2,base_mat2 = kernel_weights_2D(CMZ_centroid_mat2[p],MPM.inv_dx)

   #Get mass weight
   mass_weight_mat1 = 0.0
   mass_weight_mat2 = 0.0
   for i in range(3):
      for j in range(3):
         weight_mat1 = w_mat1[i][0] * w_mat1[j][1]
         weight_mat2 = w_mat2[i][0] * w_mat2[j][1]
         mass_weight_mat1 += grid1_m[base_mat1[0] + i,base_mat1[1] + j] * weight_mat1 
         mass_weight_mat2 += grid2_m[base_mat2[0] + i,base_mat2[1] + j] * weight_mat2
            
   for i in range(3):
      for j in range(3):
         weight_mat1 = w_mat1[i][0] * w_mat1[j][1]
         weight_mat2 = w_mat2[i][0] * w_mat2[j][1]
         mass_w1 = grid1_m[base_mat1[0] + i,base_mat1[1] + j]/mass_weight_mat1
         mass_w2 = grid2_m[base_mat2[0] + i,base_mat2[1] + j]/mass_weight_mat2

         cohesive_force_1 = -weight_mat1*force*mass_w1
         cohesive_force_2 = weight_mat2*force*mass_w2
         wp.atomic_add(grid1_CMZ_f,base_mat1[0]+i ,base_mat1[1]+j, cohesive_force_1)    
         wp.atomic_add(grid2_CMZ_f,base_mat2[0]+i ,base_mat2[1]+j, cohesive_force_2)






