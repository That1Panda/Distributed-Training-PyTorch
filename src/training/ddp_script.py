import os
import torch
from src.training.ddp_train import DDPTrain


class DDP:
    def __init__(self):
        self.rank = int(
            os.getenv("LOCAL_RANK", 0)
        )  # Local rank for the current process
        self.world_size = int(os.getenv("WORLD_SIZE", 1))  # Total number of processes
        if self.world_size <= 1:
            raise ValueError(
                f"Number of processes should be greater than 1, got {self.world_size}"
            )

    def run(self):
        print(
            f"Starting DDP training on rank {self.rank} out of {self.world_size} processes"
        )
        ddp_trainer = DDPTrain(self.rank)
        ddp_trainer.model_training()
        print(f"DDP training completed on rank {self.rank}")


if __name__ == "__main__":
    ddp = DDP()
    ddp.run()
