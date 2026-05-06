import os
from dataclasses import asdict
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets_.visual_genome.dataset import SceneGraphDataset
from datasets_.visual_genome.collate import scene_graph_collate_fn
from diffusion.objective_generator import DiscreteSGObjectiveGenerator
from training.losses import compute_discrete_sg_loss

def build_simple_relation_class_weights(
    num_rel_classes: int,
    no_rel_token_id: int,
    no_rel_weight: float,
    device: torch.device,
) -> torch.Tensor:
    weights = torch.ones(num_rel_classes, dtype=torch.float32, device=device)
    weights[no_rel_token_id] = no_rel_weight
    return weights

def build_relation_class_weights_simple(
    num_rel_classes: int,
    no_rel_token_id: int,
    no_rel_loss_weight: float,
    device: torch.device,
) -> torch.Tensor:
    weights = torch.ones(num_rel_classes, dtype=torch.float32, device=device)
    weights[no_rel_token_id] = no_rel_loss_weight
    return weights


def build_dataloader(
    npz_path: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    return_boxes: bool = False,
    return_metadata: bool = False,
) -> tuple[SceneGraphDataset, DataLoader]:
    dataset = SceneGraphDataset(
        npz_path=npz_path,
        return_boxes=return_boxes,
        return_metadata=return_metadata,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=scene_graph_collate_fn,
    )
    return dataset, loader


class DiscreteSGTrainer:
    def __init__(
        self,
        opt,
        model: torch.nn.Module,
        train_npz_path: str,
        val_npz_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        self.opt = opt
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_dataset, self.train_loader = build_dataloader(
            npz_path=train_npz_path,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            return_boxes=False,
            return_metadata=False,
        )

        self.val_dataset = None
        self.val_loader = None
        if val_npz_path is not None:
            self.val_dataset, self.val_loader = build_dataloader(
                npz_path=val_npz_path,
                batch_size=opt.eval_batch_size,
                shuffle=False,
                num_workers=opt.num_workers,
                return_boxes=False,
                return_metadata=False,
            )

        self.model = model.to(self.device)

        # Objective generator uses the train dataset for empirical priors
        self.obj_gen = DiscreteSGObjectiveGenerator(
            cfg=opt,
            num_obj_classes=len(self.train_dataset.object_vocab),
            num_rel_classes=len(self.train_dataset.relation_vocab),
            device=self.device,
            dataset_for_priors=self.train_dataset,
        )

        # self.rel_class_weights = build_relation_class_weights_simple(
        #     num_rel_classes=len(self.train_dataset.relation_vocab),
        #     no_rel_token_id=opt.no_rel_token_id,
        #     no_rel_loss_weight=opt.no_rel_loss_weight,
        #     device=self.device,
        # )

        self.rel_class_weights = build_simple_relation_class_weights(
            num_rel_classes=len(self.train_dataset.relation_vocab),
            no_rel_token_id=opt.no_rel_token_id,
            no_rel_weight=opt.no_rel_loss_weight,
            device=self.device,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=opt.lr,
            weight_decay=opt.weight_decay,
        )

        self.global_step = 0
        self.best_val_loss = float("inf")

        os.makedirs(opt.checkpoint_dir, exist_ok=True)

    def train(self):
        for epoch in range(self.opt.num_epochs):
            train_metrics = self.train_one_epoch(epoch)

            print(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_obj={train_metrics['obj_loss']:.4f} "
                f"train_rel={train_metrics['rel_loss']:.4f}"
            )

            if self.val_loader is not None:
                val_metrics = self.evaluate(epoch)
                print(
                    f"[Epoch {epoch:03d}] "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_obj={val_metrics['obj_loss']:.4f} "
                    f"val_rel={val_metrics['rel_loss']:.4f}"
                )

                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.save_checkpoint(epoch, is_best=True)

            self.save_checkpoint(epoch, is_best=False)

    def train_one_epoch(self, epoch: int):
        self.model.train()

        running_loss = 0.0
        running_obj = 0.0
        running_rel = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Train {epoch}", leave=False)

        for batch in pbar:
            self.optimizer.zero_grad(set_to_none=True)

            batch_t = self.obj_gen.get_training_batch(batch)

            model_out = self.model(
                obj_t=batch_t["obj_t"],
                rel_t=batch_t["rel_t"],
                t=batch_t["t"],
                node_mask=batch_t["node_mask"],
                edge_mask=batch_t["edge_mask"],
            )

            loss_dict = compute_discrete_sg_loss(
                model_out=model_out,
                batch_t=batch_t,
                rel_class_weights=self.rel_class_weights,
                lambda_obj=self.opt.lambda_obj,
                lambda_rel=self.opt.lambda_rel,
            )

            loss = loss_dict["loss"]
            loss.backward()

            if self.opt.grad_clip_norm is not None and self.opt.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.grad_clip_norm)

            self.optimizer.step()

            running_loss += float(loss.detach().item())
            running_obj += float(loss_dict["obj_loss"].item())
            running_rel += float(loss_dict["rel_loss"].item())
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix(
                loss=f"{running_loss / num_batches:.4f}",
                obj=f"{running_obj / num_batches:.4f}",
                rel=f"{running_rel / num_batches:.4f}",
            )

        return {
            "loss": running_loss / max(num_batches, 1),
            "obj_loss": running_obj / max(num_batches, 1),
            "rel_loss": running_rel / max(num_batches, 1),
        }

    @torch.no_grad()
    def evaluate(self, epoch: int):
        self.model.eval()

        running_loss = 0.0
        running_obj = 0.0
        running_rel = 0.0
        num_batches = 0

        pbar = tqdm(self.val_loader, desc=f"Eval {epoch}", leave=False)

        for batch in pbar:
            batch_t = self.obj_gen.get_training_batch(batch)

            model_out = self.model(
                obj_t=batch_t["obj_t"],
                rel_t=batch_t["rel_t"],
                t=batch_t["t"],
                node_mask=batch_t["node_mask"],
                edge_mask=batch_t["edge_mask"],
            )

            loss_dict = compute_discrete_sg_loss(
                model_out=model_out,
                batch_t=batch_t,
                rel_class_weights=self.rel_class_weights,
                lambda_obj=self.opt.lambda_obj,
                lambda_rel=self.opt.lambda_rel,
            )

            running_loss += float(loss_dict["loss"].item())
            running_obj += float(loss_dict["obj_loss"].item())
            running_rel += float(loss_dict["rel_loss"].item())
            num_batches += 1

            pbar.set_postfix(
                loss=f"{running_loss / num_batches:.4f}",
                obj=f"{running_obj / num_batches:.4f}",
                rel=f"{running_rel / num_batches:.4f}",
            )

        return {
            "loss": running_loss / max(num_batches, 1),
            "obj_loss": running_obj / max(num_batches, 1),
            "rel_loss": running_rel / max(num_batches, 1),
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        ckpt = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": asdict(self.opt) if hasattr(self.opt, "__dataclass_fields__") else None,
        }

        latest_path = os.path.join(self.opt.checkpoint_dir, "latest.pt")
        torch.save(ckpt, latest_path)

        epoch_path = os.path.join(self.opt.checkpoint_dir, f"epoch_{epoch:03d}.pt")
        torch.save(ckpt, epoch_path)

        if is_best:
            best_path = os.path.join(self.opt.checkpoint_dir, "best.pt")
            torch.save(ckpt, best_path)