import torch
import torch.multiprocessing as mp

from src.training.ddp_train import DDPTrain


class DDP:
    def __init__(self):
        self.world_size = torch.cuda.device_count()
        if self.world_size < 2:
            raise ValueError("This script is intended to be run with at least 2 GPUs")

    def main_worker(self, rank):
        ddp_trainer = DDPTrain(rank)
        ddp_trainer.model_training()

    def run(self):
        print(f"Starting DDP training with {self.world_size} GPUs")
        mp.spawn(self.main_worker, nprocs=self.world_size, join=True)
        print("DDP training completed")


if __name__ == "__main__":
    ddp = DDP()
    ddp.run()
