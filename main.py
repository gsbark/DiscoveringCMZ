
import yaml
import warp as wp 
from src.MPM_inverse import MPM_program_inv

with open('./input_files/ex1.yaml', 'r') as file:
    Inp_file = yaml.safe_load(file)

wp.init()
if wp.is_cuda_available():
    device = 'cuda'
else:
    device = 'cpu'

MPM_config = MPM_program_inv(**Inp_file,device=device)
MPM_config.train()


