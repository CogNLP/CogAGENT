import torch
import torch.distributed as dist


def move_dict_value_to_device(batch, device, rank=-1, non_blocking=False):
    if rank == -1 and not isinstance(device, torch.device):
        raise TypeError(f"device must be `torch.device` in a single process, got `{type(device)}`")

    _move_dict_value_to_device(batch, device, rank, non_blocking)


def _move_dict_value_to_device(batch, device, rank=-1, non_blocking=False):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if rank == -1:
                batch[key] = value.to(device, non_blocking=non_blocking)
            else:
                batch[key] = value.to(torch.device("cuda:{}".format(rank)), non_blocking=non_blocking)
        elif isinstance(value, dict):
            _move_dict_value_to_device(batch[key], device, rank, non_blocking)


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt / nprocs
    return rt