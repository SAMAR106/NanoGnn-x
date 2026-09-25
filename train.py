"""
PyTorch Lightning training loop for NanoGNN-X.

Run: python train.py

Expects data/raw_cifs/ to contain your .cif files plus a targets.csv
(see data/raw_cifs/targets.csv for the expected format). Checkpoints,
logs, and the fitted target normalizer land in CFG.checkpoint_dir.
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch_geometric.loader import DataLoader

from src.config import CFG
from src.dataset import NanoGNNXDataset
from src.model import NanoGNNX
from src.normalizer import TargetNormalizer


class NanoGNNXLightning(pl.LightningModule):
    def __init__(self, normalizer: TargetNormalizer, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        self.model = NanoGNNX(cfg)
        # Materials databases (Materials Project, OQMD, etc.) commonly
        # contain DFT convergence errors and outliers; Huber loss is far
        # more robust to those than plain MSE.
        #
        # The network is trained against *normalized* targets (see
        # src/normalizer.py) -- bandgap and formation energy live on
        # different scales, and fitting both jointly with one Huber
        # delta and no normalization biases training toward whichever
        # target has the larger raw magnitude. MAE is logged in
        # physical units regardless, by inverse-transforming
        # predictions before comparing them to the raw targets.
        #
        # `normalizer` is required, not optional-with-a-None-default:
        # every _step() call needs it unconditionally, so a None default
        # would just turn a missing normalizer into an AttributeError on
        # the first training step instead of failing at construction
        # time where it's actually easy to diagnose.
        self.normalizer = normalizer
        self.save_hyperparameters(ignore=["cfg", "normalizer"])

    def forward(self, data):
        return self.model(data)

    def _step(self, batch, stage: str):
        pred_norm = self(batch)  # (B, 2), normalized
        target_raw = batch.y.view(-1, 2)
        target_norm = self.normalizer.transform(target_raw)

        loss = F.huber_loss(pred_norm, target_norm, delta=self.cfg.huber_delta)

        with torch.no_grad():
            pred_raw = self.normalizer.inverse_transform(pred_norm)
            bandgap_mae = (pred_raw[:, 0] - target_raw[:, 0]).abs().mean()
            formation_mae = (pred_raw[:, 1] - target_raw[:, 1]).abs().mean()

        batch_size = target_raw.size(0)
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}_bandgap_mae_eV", bandgap_mae, prog_bar=True, batch_size=batch_size)
        self.log(
            f"{stage}_formation_mae_eV_per_atom",
            formation_mae,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=self.cfg.lr_factor, patience=self.cfg.lr_patience
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


def build_splits(cfg=CFG):
    """Splits the dataset and fits the target normalizer on the
    *training* split only -- fitting on validation/test data would leak
    information about their distribution into training and make the
    validation metrics an overly optimistic measure of generalization.

    Uses composition-aware splitting (src/splitting.py) rather than a
    plain random split: structures sharing a chemical system are kept
    together in one split, so a near-duplicate compound can't leak
    from train into validation/test and inflate the metrics. Verified
    to have essentially no effect on the bundled 101-structure demo
    dataset specifically (100 of 101 chemical systems are unique there)
    -- this matters once you scale up via
    scripts/download_materials_project.py, where compositional
    duplicates become common.
    """
    from src.splitting import composition_aware_split, get_chemical_system

    full_dataset = NanoGNNXDataset(raw_dir=cfg.raw_cif_dir)
    chemical_systems = [
        get_chemical_system(f"{cfg.raw_cif_dir}/{fn}") for fn in full_dataset.df["filename"]
    ]
    train_indices, val_indices, test_indices = composition_aware_split(
        chemical_systems,
        val_frac=cfg.val_split,
        test_frac=cfg.test_split,
        seed=cfg.seed,
    )
    train_set = torch.utils.data.Subset(full_dataset, train_indices)
    val_set = torch.utils.data.Subset(full_dataset, val_indices)
    test_set = torch.utils.data.Subset(full_dataset, test_indices)

    train_targets = torch.stack([full_dataset.get(i).y.squeeze(0) for i in train_set.indices])
    normalizer = TargetNormalizer().fit(train_targets)

    return train_set, val_set, test_set, normalizer


def main(cfg=CFG):
    pl.seed_everything(cfg.seed)

    train_set, val_set, test_set, normalizer = build_splits(cfg)

    import os

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    normalizer.save(cfg.normalizer_path)

    # PyG's DataLoader batches variable-sized graphs into a single
    # block-diagonal adjacency matrix, so a batch of N crystals with
    # different atom counts still trains as one forward pass.
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    lit_model = NanoGNNXLightning(normalizer=normalizer, cfg=cfg)

    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.checkpoint_dir,
        filename="nanognn-x-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )
    early_stop_cb = EarlyStopping(monitor="val_loss", patience=20, mode="min")

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_cb, early_stop_cb],
        log_every_n_steps=10,
    )

    trainer.fit(lit_model, train_loader, val_loader)
    trainer.test(lit_model, test_loader, ckpt_path="best")
    return trainer, lit_model


if __name__ == "__main__":
    main()
