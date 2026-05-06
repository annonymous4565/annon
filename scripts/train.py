import pyrallis
import torch

from configs import DiscreteSGConfig
from models.denoiser import SceneGraphDenoiser
from training.trainer import DiscreteSGTrainer


@pyrallis.wrap()
def main(opt: DiscreteSGConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SceneGraphDenoiser(
        num_obj_classes=opt.num_obj_classes,   # set from dataset or override below
        num_rel_classes=opt.num_rel_classes,
        d_model=opt.d_model,
        num_layers=opt.num_mp_layers,
        dropout=opt.dropout,
    )

    trainer = DiscreteSGTrainer(
        opt=opt,
        model=model,
        train_npz_path=opt.train_npz_path,
        val_npz_path=opt.val_npz_path,
        device=device,
    )

    # overwrite class counts from loaded dataset if config leaves them dummy
    trainer.model = SceneGraphDenoiser(
        num_obj_classes=len(trainer.train_dataset.object_vocab),
        num_rel_classes=len(trainer.train_dataset.relation_vocab),
        d_model=opt.d_model,
        num_layers=opt.num_mp_layers,
        dropout=opt.dropout,
    ).to(device)

    trainer.optimizer = torch.optim.AdamW(
        trainer.model.parameters(),
        lr=opt.lr,
        weight_decay=opt.weight_decay,
    )

    trainer.train()


if __name__ == "__main__":
    main()