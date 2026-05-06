import os
import torch
import torch.distributed as dist

import socket
from datetime import timedelta


def distributed_available() -> bool:
    return dist.is_available()


def distributed_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if distributed_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if distributed_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed():
    """
    Initializes DDP if launched with torchrun.
    Returns:
        is_distributed, rank, world_size, local_rank, device
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
        return True, rank, world_size, local_rank, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, 1, 0, device



# def setup_distributed():
#     if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
#         rank = int(os.environ["RANK"])
#         world_size = int(os.environ["WORLD_SIZE"])
#         local_rank = int(os.environ["LOCAL_RANK"])

#         print(
#             f"[pre-setup] host={socket.gethostname()} pid={os.getpid()} "
#             f"rank={rank} local_rank={local_rank} world_size={world_size}",
#             flush=True,
#         )

#         print(f"[rank {rank}] before set_device({local_rank})", flush=True)
#         torch.cuda.set_device(local_rank)
#         print(f"[rank {rank}] after set_device({local_rank})", flush=True)
#         print(
#             f"[rank {rank}] MASTER_ADDR={os.environ.get('MASTER_ADDR')} "
#             f"MASTER_PORT={os.environ.get('MASTER_PORT')}",
#             flush=True,
#         )
#         print(f"[rank {rank}] before init_process_group", flush=True)
#         dist.init_process_group(
#             backend="nccl",
#             init_method="env://",
#             timeout=timedelta(minutes=5),
#         )
#         print(f"[rank {rank}] after init_process_group", flush=True)

#         device = torch.device(f"cuda:{local_rank}")
#         return True, rank, world_size, local_rank, device

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     return False, 0, 1, 0, device


def cleanup_distributed():
    if distributed_initialized():
        dist.destroy_process_group()


@torch.no_grad()
def reduce_tensor_sum(x: torch.Tensor) -> torch.Tensor:
    if not distributed_initialized():
        return x
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y


@torch.no_grad()
def reduce_scalar_sum(value: float, device: torch.device) -> float:
    t = torch.tensor([value], dtype=torch.float64, device=device)
    t = reduce_tensor_sum(t)
    return float(t.item())


def barrier():
    if distributed_initialized():
        dist.barrier()