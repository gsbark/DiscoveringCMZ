import argparse
import yaml
import warp as wp 
from src.MPM_inverse import MPM_program_inv

parser = argparse.ArgumentParser(description="Run example cases")
parser.add_argument(
    "--case",
    choices=["ex1", "ex2"],
    default="ex1"
)
args = parser.parse_args()

with open(f'./input_files/{args.case}.yaml', 'r') as file:
    Inp_file = yaml.safe_load(file)

wp.init()
if wp.is_cuda_available():
    device = 'cuda'
else:
    device = 'cpu'

print(f'Running:{args.case}')
MPM_config = MPM_program_inv(**Inp_file,device=device)
MPM_config.train()

