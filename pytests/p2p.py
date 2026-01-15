import os

import torch
import torch.distributed as dist
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor_size", type=int, default=4096, help="tensor sizes")
    parser.add_argument("--num_tensors", type=int, default=4, help="number of tensors to send/recv")
    parser.add_argument("--iteration", type=int, default=100, help="number of iterations")
    args = parser.parse_args()

    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size != 2:
        raise ValueError(f"This test requires exactly 2 ranks, got {world_size}")

    # Create tensors for send/recv
    tensors = [torch.ones(args.tensor_size, device="cuda") * (rank + 1) for _ in range(args.num_tensors)]
    recv_tensors = [torch.zeros(args.tensor_size, device="cuda") for _ in range(args.num_tensors)]

    for iteration in range(args.iteration):
        requests = []

        if rank == 0:
            # Rank 0 sends to rank 1
            for i, tensor in enumerate(tensors):
                req = dist.isend(tensor, dst=1)
                requests.append(req)
            # Rank 0 receives from rank 1
            for i, recv_tensor in enumerate(recv_tensors):
                req = dist.irecv(recv_tensor, src=1)
                requests.append(req)
        else:
            # Rank 1 receives from rank 0
            for i, recv_tensor in enumerate(recv_tensors):
                req = dist.irecv(recv_tensor, src=0)
                requests.append(req)
            # Rank 1 sends to rank 0
            for i, tensor in enumerate(tensors):
                req = dist.isend(tensor, dst=0)
                requests.append(req)

        # Wait for all operations to complete
        for req in requests:
            req.wait()

        # Verify received data
        if rank == 0:
            for i, recv_tensor in enumerate(recv_tensors):
                expected_value = 2.0  # rank 1 sends tensors filled with 2
                assert torch.allclose(recv_tensor, torch.ones_like(recv_tensor) * expected_value), \
                    f"Rank 0: Received incorrect data in tensor {i} at iteration {iteration}"
        else:
            for i, recv_tensor in enumerate(recv_tensors):
                expected_value = 1.0  # rank 0 sends tensors filled with 1
                assert torch.allclose(recv_tensor, torch.ones_like(recv_tensor) * expected_value), \
                    f"Rank 1: Received incorrect data in tensor {i} at iteration {iteration}"

    torch.cuda.synchronize()

    if rank == 0:
        print(f"Successfully completed {args.iteration} iterations of P2P communication")
        print(f"Exchanged {args.num_tensors} tensors of size {args.tensor_size} per iteration")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

