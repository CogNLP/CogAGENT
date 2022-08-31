import torch
import random
import numpy as np
import os
import datetime
from cogagent.utils.log_utils import init_logger

def init_cogagent(
        output_path,
        device_id=None,
        folder_tag="",
        seed=0,
        rank=-1,
):
    # set the random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if not isinstance(device_id, int):
        if rank == -1:
            raise ValueError("Device Id can noly be int in a single process!")

    if rank in [0, -1]:
        # calculate the output path
        new_output_path = os.path.join(
            output_path,
            folder_tag + "--" + str(datetime.datetime.now())[:-4].replace(':', '-').replace(' ', '--')
        )
        if not os.path.exists(os.path.abspath(new_output_path)):
            os.makedirs(os.path.abspath(new_output_path))

        # initialize the logger configuration
        init_logger(os.path.join(new_output_path, "log.txt"), rank=rank)

        # set the cuda device
        if rank == -1:
            device = torch.device('cuda:{}'.format(device_id) if torch.cuda.is_available() == True else "cpu")
        else:
            device = torch.device("cuda:0")
    else:
        new_output_path = None
        init_logger(new_output_path, rank=rank)
        device = torch.device("cuda:0")

    return device, new_output_path