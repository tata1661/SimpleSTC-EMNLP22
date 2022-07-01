import random
import numpy as np
import torch
import os

def fetch_tensor(tensor_dict, tensor_type, device):
    result = torch.tensor(tensor_dict[tensor_type], dtype=torch.float, device=device)
    return result


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


