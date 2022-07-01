import random
import numpy as np
import paddle
import os

def fetch_tensor(tensor_dict, tensor_type, device):
    return paddle.to_tensor(tensor_dict[tensor_type], dtype='float64', place=device)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    paddle.seed(seed)

