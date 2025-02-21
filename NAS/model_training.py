import argparse, logging
import random, time, sys
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

def solution_evaluation(model)