import os

import torch.distributed
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor_size", type=int, default=4096, help="tensor sizes")
    parser.add_argument("--iteration", type=int, default=100, help="number of iterations")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))
    torch.distributed.init_process_group(backend="nccl")
    t = torch.ones(args.tensor_size, device="cuda")
    for _ in range(args.iteration):
        torch.distributed.all_reduce(t)

    torch.cuda.synchronize()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
