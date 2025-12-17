
import os 
import shutil
import torch
import torch.nn as nn
import warp as wp 
import numpy as np 
from warp.types import matrix,vector
from warp.optim import Adam
from src.wp_utils import *
from src.MPM_funcs import * 
from src.utils import find_surface,CZ_circle,CZ_two

class MLP(torch.nn.Module):
   def __init__(self,hidden_size):
      super().__init__()
      self.layers = torch.nn.ModuleList()
      self.layers_num = len(hidden_size)-1
      for i in range(self.layers_num):
         self.layers.append(torch.nn.Linear(hidden_size[i],hidden_size[i+1],bias=True))
         if i != self.layers_num - 1:
            self.layers.append(torch.nn.ELU())
   def forward(self, x):
        for _,layer in enumerate(self.layers):
            x = layer(x)   
        return nn.functional.relu(x)

class matrix32(matrix(shape=(3,2),dtype=wp.float64)):
    pass

class MPM_program_inv:
    def __init__(self,MPM_config,Material,Cohesive_Zone,Geometry,Optimization,device):
        
        self.case = MPM_config['case']
        self.dataset_type = MPM_config['type']
        self.num_p_mat1 = MPM_config['particles_mat1']
        self.num_p_mat2 = MPM_config['particles_mat2']
        self.n_grid = MPM_config['n_grid']
        self.steps = MPM_config['steps']
        
        #Optimization params
        self.loss_factor = 1.0
        self.acc_every = Optimization['acc_every']
        self.learning_rate = Optimization['lr']
        dataset_steps = self.steps//self.acc_every

        self.MPM_variables = MPM_vars_2D()
        self.states_mat1 = MPM_state_2D()
        self.states_mat2 = MPM_state_2D()
        self.cmz_state = CMZ_state_2D()
            
        self.num_CMZ_elements = Cohesive_Zone['elements']
        
        if self.case==1:
            self.CMZ_vertices,self.CMZ_connectivity = CZ_two(self.num_CMZ_elements,Geometry)
        elif self.case==2:
            self.CMZ_vertices,self.CMZ_connectivity = CZ_circle(self.num_CMZ_elements,Geometry)
        
        self.update = self.step_2D
        self.dx_true1 = wp.zeros(shape=(dataset_steps,self.num_p_mat1),dtype=wp.vec2f,requires_grad=True,device=device)
        self.dx_true2 = wp.zeros(shape=(dataset_steps,self.num_p_mat2),dtype=wp.vec2f,requires_grad=True,device=device)
        self.num_CMZ_vertices = self.CMZ_vertices.shape[0]
        
        self.MPM_variables.initialize(
            Geometry,
            MPM_config,
            Material
            )

        self.states_mat1.initialize(
            self.acc_every+1,
            self.n_grid,
            self.num_p_mat1,
            requires_grad=True,
            device=device
            )
        
        self.states_mat2.initialize(
            self.acc_every+1,
            self.n_grid,
            self.num_p_mat2,
            requires_grad=True,
            device=device
            )
        
        self.cmz_state.initialize(
            self.acc_every+1,
            self.num_CMZ_vertices,
            self.num_CMZ_elements,
            requires_grad=True,
            device=device
            )
        
        if self.dataset_type=='noiseless':
            self.dataset_x1 = np.load(f'./datasets/Example{self.case}/noiseless/x1.npy')[::self.acc_every][1:]
            self.dataset_x2 = np.load(f'./datasets/Example{self.case}/noiseless/x2.npy')[::self.acc_every][1:]
        elif self.dataset_type=='noise_small':
            self.dataset_x1 = np.load(f'./datasets/Example{self.case}/noise_small/x1.npy')[1:]
            self.dataset_x2 = np.load(f'./datasets/Example{self.case}/noise_small/x2.npy')[1:]
        elif self.dataset_type=='noise_large':
            self.dataset_x1 = np.load(f'./datasets/Example{self.case}/noise_large/x1.npy')[1:]
            self.dataset_x2 = np.load(f'./datasets/Example{self.case}/noise_large/x2.npy')[1:]
        else: 
            raise NotImplementedError
    
        self.dx_true1.assign(self.dataset_x1)
        self.dx_true2.assign(self.dataset_x2)

        # Warp NN arrays
        #----------------------------------------------------
        #Random initialization here 
        self.torch_mlp = MLP(hidden_size=[2,64,64,2])
        torch_weights = self.torch_mlp.state_dict()
    
        self.wp_scale_inp = wp.array([1.0/0.01,1.0/0.01],dtype=wp.float32,requires_grad=True,device=device)
        self.wp_scale_out = wp.array([25.0],dtype=wp.float32,requires_grad=True,device=device)

        self.mlp_inp = wp.zeros((self.acc_every,2,self.num_CMZ_elements),dtype=wp.float32,requires_grad=True,device=device)
        self.wp_mlp = wp_MLP(torch_weights,self.num_CMZ_elements,self.acc_every,device=device)
        self.loss_arr = wp.zeros((1), dtype=wp.float32,requires_grad=True,device=device)        

        #Custom Gradients arrays to accumulate gradients 
        self.grad_w1 = wp.zeros_like(self.wp_mlp.w1.grad.flatten(),device=device)
        self.grad_b1 = wp.zeros_like(self.wp_mlp.b1.grad.flatten(),device=device)
        self.grad_w2 = wp.zeros_like(self.wp_mlp.w2.grad.flatten(),device=device)
        self.grad_b2 = wp.zeros_like(self.wp_mlp.b2.grad.flatten(),device=device)
        self.grad_w3 = wp.zeros_like(self.wp_mlp.w3.grad.flatten(),device=device)
        self.grad_b3 = wp.zeros_like(self.wp_mlp.b3.grad.flatten(),device=device)

        self.optimizer = Adam(
            [self.wp_mlp.w1.flatten(),self.wp_mlp.b1.flatten(),
             self.wp_mlp.w2.flatten(),self.wp_mlp.b2.flatten(),
             self.wp_mlp.w3.flatten(),self.wp_mlp.b3.flatten()],
             lr=self.learning_rate)
        
        self.best_loss = np.inf
        self.epochs = 1000

        self.learned_model = np.zeros((self.epochs,2,50,50))
        shutil.rmtree('./Output/Saved_model',ignore_errors=True)
        os.makedirs('./Output/Saved_model',exist_ok=True)
        self.learned_model[1,0],self.learned_model[1,1] = find_surface(self.wp_mlp,self.torch_mlp,self.wp_scale_inp,self.wp_scale_out)


    def save_model(self):
        os.makedirs('./Output/Saved_model/best',exist_ok=True)
        for attr_name, attr_value in self.wp_mlp.__dict__.items():
            if 'b' in attr_name or 'w' in attr_name:
                np.save(f'./Output/Saved_model/best/{attr_name}.npy',attr_value.numpy())

    def Gradient_step(self,grads):
        self.optimizer.step([*grads])

    def Add_grads(self,total_grads,grads):
        total_grads = [*total_grads]
        gradients = [*grads]
        for grad,acc_grad in zip(gradients,total_grads):
            wp.launch(
                kernel=add,
                dim=grad.shape,
                inputs=[acc_grad,grad]
                )

    def reset_grads(self):

        self.grad_w1.zero_()
        self.grad_w2.zero_()
        self.grad_w3.zero_()
        
        self.grad_b1.zero_()
        self.grad_b2.zero_()
        self.grad_b3.zero_()

    def reset_geo(self):
        
        self.states_mat1.reset_grid()
        self.states_mat1.reset_particles()

        self.states_mat2.reset_grid()
        self.states_mat2.reset_particles()

        self.reset_grads()
        self.cmz_state.assign_CMZ(self.CMZ_vertices,self.CMZ_connectivity)

        wp.launch(
            kernel=geometry_2D,
            dim=[1],
            inputs=[self.MPM_variables,
                    self.states_mat1,
                    self.states_mat2,
                    self.n_grid,
                    self.case]
                    )
        assert self.num_p_mat1 == self.states_mat1.particles_num.numpy()[0], self.states_mat1.particles_num.numpy()[0]
        assert self.num_p_mat2 == self.states_mat2.particles_num.numpy()[0], self.states_mat2.particles_num.numpy()[0]
    
    def run_NN(self,mlp_inp,out1,out2,out3,out_true):

        wp.launch(
            kernel=run_mlp,
            dim=[self.num_CMZ_elements],
            inputs=[self.wp_mlp.w1,
                    self.wp_mlp.b1,
                    self.wp_mlp.w2,
                    self.wp_mlp.b2,
                    self.wp_mlp.w3,
                    self.wp_mlp.b3,
                    mlp_inp,
                    out1,
                    out2,
                    out3]
                    )
        
        wp.launch(
            kernel=scale_f,
            dim=[self.num_CMZ_elements],
            inputs=[out3,
                    self.wp_scale_out,
                    out_true]
                    )
         
    def step_2D(self,step):
        
        wp.launch(
            kernel=p2g_2D,
            dim=[self.num_p_mat1],
            inputs=[self.MPM_variables,
                    self.states_mat1.x[step],
                    self.states_mat1.v[step],
                    self.states_mat1.L[step],
                    self.states_mat1.F[step],
                    self.states_mat1.stress[step],
                    self.states_mat1.grid_m[step],
                    self.states_mat1.grid_mv[step]]
                    )
        
        wp.launch(
            kernel=p2g_2D,
            dim=[self.num_p_mat2],
            inputs=[self.MPM_variables,
                    self.states_mat2.x[step],
                    self.states_mat2.v[step],
                    self.states_mat2.L[step],
                    self.states_mat2.F[step],
                    self.states_mat2.stress[step],
                    self.states_mat2.grid_m[step],
                    self.states_mat2.grid_mv[step]]
                    )
        
        wp.launch(
            kernel=CMZ_separation_2D,
            dim=[self.num_CMZ_elements],
            inputs=[self.cmz_state.CMZ_vertices_mat1[step],
                    self.cmz_state.CMZ_vertices_mat2[step],
                    self.cmz_state.CMZ_connectivity,
                    self.cmz_state.CMZ_pairs,
                    self.cmz_state.CMZ_centroid_mat1[step],
                    self.cmz_state.CMZ_centroid_mat2[step],
                    self.cmz_state.CMZ_area[step],
                    self.cmz_state.CMZ_disp_jump[step],
                    self.cmz_state.CMZ_normal[step],
                    self.cmz_state.CMZ_tangent[step]]
                    )
        
        wp.launch(
            kernel=NN_input_2D,
            dim=[self.num_CMZ_elements],
            inputs=[self.mlp_inp[step],
                    self.cmz_state.CMZ_normal[step],
                    self.cmz_state.CMZ_tangent[step],
                    self.cmz_state.CMZ_disp_jump[step],
                    self.wp_scale_inp]
                    )

        self.run_NN(self.mlp_inp[step],
                    self.wp_mlp.out1[step],
                    self.wp_mlp.out2[step],
                    self.wp_mlp.out3[step],
                    self.wp_mlp.Traction[step])
        
        wp.launch(
            kernel=CMZ_Force_updateNN_2D,
            dim=[self.num_CMZ_elements],
            inputs=[self.MPM_variables,
                    self.cmz_state.CMZ_centroid_mat1[step],
                    self.cmz_state.CMZ_centroid_mat2[step],
                    self.cmz_state.CMZ_area[step],
                    self.cmz_state.CMZ_normal[step],
                    self.cmz_state.CMZ_tangent[step],
                    self.wp_mlp.Traction[step],
                    self.states_mat1.grid_m[step],
                    self.states_mat2.grid_m[step],
                    self.states_mat1.grid_CMZ_force[step],
                    self.states_mat2.grid_CMZ_force[step],
                    self.case]
                    )
        
        wp.launch(
            kernel=grid_update_2D,
            dim=[self.n_grid,self.n_grid],
            inputs=[self.MPM_variables,
                    self.states_mat1.grid_mv[step],
                    self.states_mat1.grid_m[step],
                    self.states_mat1.grid_v[step],
                    self.states_mat1.grid_CMZ_force[step],
                    self.states_mat2.grid_mv[step],
                    self.states_mat2.grid_m[step],
                    self.states_mat2.grid_v[step],
                    self.states_mat2.grid_CMZ_force[step]]
                    )
        
        if self.case==1:
            wp.launch(
                kernel=BCs_ex1,
                dim=[self.n_grid,self.n_grid],
                inputs=[self.MPM_variables,
                        self.states_mat1.grid_m[step],
                        self.states_mat1.grid_v[step],
                        self.states_mat2.grid_m[step],
                        self.states_mat2.grid_v[step]]
                        )
            
        elif self.case==2:
            wp.launch(
                kernel=BCs_ex2,
                dim=[self.n_grid,self.n_grid],
                inputs=[self.MPM_variables,
                        self.states_mat1.grid_m[step],
                        self.states_mat1.grid_v[step],
                        self.states_mat2.grid_m[step],
                        self.states_mat2.grid_v[step]]
                        )
    
        wp.launch(
            kernel=g2p_2D,
            dim=[self.num_p_mat1],
            inputs=[self.MPM_variables,
                    self.states_mat1.x[step],
                    self.states_mat1.F[step],
                    self.states_mat1.x[step+1],
                    self.states_mat1.v[step+1],
                    self.states_mat1.L[step+1],
                    self.states_mat1.F[step+1],
                    self.states_mat1.grid_v[step]]
                    )
        
        wp.launch(
            kernel=g2p_2D,
            dim=[self.num_p_mat2],
            inputs=[self.MPM_variables,
                    self.states_mat2.x[step],
                    self.states_mat2.F[step],
                    self.states_mat2.x[step+1],
                    self.states_mat2.v[step+1],
                    self.states_mat2.L[step+1],
                    self.states_mat2.F[step+1],
                    self.states_mat2.grid_v[step]]
                    )
        
        wp.launch(
            kernel=CMZ_update_2D,
            dim=[self.num_CMZ_vertices],
            inputs=[self.MPM_variables,
                    self.states_mat1.grid_v[step],
                    self.states_mat1.grid_m[step],
                    self.cmz_state.CMZ_vertices_mat1[step],
                    self.cmz_state.CMZ_vertices_mat1[step+1]]
                    )
        
        wp.launch(
            kernel=CMZ_update_2D,
            dim=[self.num_CMZ_vertices],
            inputs=[self.MPM_variables,
                    self.states_mat2.grid_v[step],
                    self.states_mat2.grid_m[step],
                    self.cmz_state.CMZ_vertices_mat2[step],
                    self.cmz_state.CMZ_vertices_mat2[step+1]]
                    )
                    
    def get_loss(self,substep,istep):
        
        wp.launch(
            kernel=MSE_2D,
            dim=[self.num_p_mat1],
            inputs=[self.states_mat1.x[substep+1],
                    self.dx_true1[istep],
                    self.loss_arr,
                    self.states_mat1.flag,
                    self.loss_factor]
                    )
        
        wp.launch(
            kernel=MSE_2D,
            dim=[self.num_p_mat2],
            inputs=[self.states_mat2.x[substep+1],
                    self.dx_true2[istep],
                    self.loss_arr,
                    self.states_mat2.flag,
                    self.loss_factor]
                    )

    def train(self):
        loss_epoch = np.zeros(self.epochs)
        iter_loss = {'train_loss':0.0}
        for epoch in range(self.epochs):
            step = 0
            iter_loss['train_loss'] = 0.0
            self.reset_geo()
            losses1 = np.zeros((self.num_p_mat1))
            losses2 = np.zeros((self.num_p_mat2))
            with wp.ScopedTimer('Sim'):
                for istep in range(self.steps//self.acc_every):
                    tape = wp.Tape()
                    if step>0:
                        self.states_mat1.reset_substeps(self.acc_every+1)
                        self.states_mat2.reset_substeps(self.acc_every+1)
                        self.cmz_state.reset_substeps(self.acc_every+1)
                        self.wp_mlp.reset()
                    with tape:
                        for substep in range(self.acc_every):
                            step = istep*self.acc_every + substep
                            self.update(substep)
                        self.get_loss(substep,istep)
                    
                    tape.backward(loss=self.loss_arr)    
                    iter_loss['train_loss'] +=self.loss_arr.numpy()
                    self.loss_arr.zero_()
                    self.Add_grads(
                        [self.grad_w1,self.grad_b1,self.grad_w2,self.grad_b2,self.grad_w3,self.grad_b3],
                        [self.wp_mlp.w1.grad.flatten(),self.wp_mlp.b1.grad.flatten(),
                         self.wp_mlp.w2.grad.flatten(),self.wp_mlp.b2.grad.flatten(),
                         self.wp_mlp.w3.grad.flatten(),self.wp_mlp.b3.grad.flatten()]
                         )
                    
                    tape.reset()
                    losses1 += np.sum(((self.states_mat1.x[-1].numpy()- self.dx_true1[istep].numpy()))**2.0,axis=1)
                    losses2 += np.sum(((self.states_mat2.x[-1].numpy()- self.dx_true2[istep].numpy()))**2.0,axis=1)
    
                self.Gradient_step(
                    [self.grad_w1,self.grad_b1,
                     self.grad_w2,self.grad_b2,
                     self.grad_w3,self.grad_b3])
            
            self.learned_model[epoch,0],self.learned_model[epoch,1] = find_surface(self.wp_mlp,self.torch_mlp,self.wp_scale_inp,self.wp_scale_out)
            loss_epoch[epoch] = sum(losses1) + sum(losses2) 

            if iter_loss['train_loss']<self.best_loss:
                self.best_loss=iter_loss['train_loss']
                self.save_model()

            print('Epoch:',epoch,' Displacement loss:',iter_loss['train_loss'])
            np.save('./Output/loss.npy',loss_epoch[:epoch])
            np.save('./Output/out_surf.npy',self.learned_model[:epoch])

