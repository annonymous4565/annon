import pyrallis
from configs import DiscreteSGConfig
from training.trainer_ddp import StructuredSGDDPTrainer


@pyrallis.wrap()
def main(opt: DiscreteSGConfig):
    trainer = StructuredSGDDPTrainer(opt)
    trainer.train()


if __name__ == "__main__":
    main()