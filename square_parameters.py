import numpy as np
import torch
import argparse

def strToBool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def noneOrStr(value):
    if value == 'None':
        return None
    return value

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='dssx')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--cr', type=float, default=0.)
parser.add_argument('--mr', type=float, default=0.)
parser.add_argument('--n_data', type=int, default=1000)
parser.add_argument('--log_pre', type=str, default='/tmp')

args = parser.parse_args()
model = args.model
seed = args.seed
device = args.device
cr = args.cr
mr = args.mr
n_data = args.n_data
log_pre = args.log_pre