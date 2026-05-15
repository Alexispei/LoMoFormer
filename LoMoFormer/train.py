from config import FullConfig
from model_builder import build_model
from trainer import Trainer
import torch
import data_utils
import os
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from utils import *


def _load_shared_resources(config: FullConfig):
    """
    Load tokenizer, molecular map, and gene expression once.
    """
    path_cfg = config.path

    tokenizer = Mol_Tokenizer(path_cfg.token_id_file)
    map_dict = np.load(path_cfg.molecular_info, allow_pickle=True).item()
    cell_df = pd.read_csv(path_cfg.cell_features)
    gene_expression_dict = data_utils.build_gene_expression_dict(cell_df)

    return {
        "tokenizer": tokenizer,
        "map_dict": map_dict,
        "gene_expression_dict": gene_expression_dict
    }


def _build_dataset(df, resources):
    return data_utils.FastGraphBertDataset(
        data_set=df,
        tokenizer=resources["tokenizer"],
        map_dict=resources["map_dict"],
        gene_expression_dict=resources["gene_expression_dict"],
        label_field='label'
    )


def get_kfold_dataloaders(config: FullConfig, resources, k=5, fold_idx=0, seed=42):
    assert 0 <= fold_idx < k, f"fold_idx must be in range [0, {k})"

    path_cfg = config.path
    all_df = pd.read_csv(path_cfg.data_folder).reset_index(drop=True)

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    indices = np.arange(len(all_df))
    train_idx, val_idx = list(kf.split(indices))[fold_idx]

    train_df = all_df.iloc[train_idx].reset_index(drop=True)
    val_df = all_df.iloc[val_idx].reset_index(drop=True)

    train_dataset = _build_dataset(train_df, resources)
    val_dataset = _build_dataset(val_df, resources)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        collate_fn=data_utils.fast_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        collate_fn=data_utils.fast_collate_fn
    )

    return train_loader, val_loader


def get_pre_split_dataloaders(config: FullConfig, resources):
    """
    Use pre-split CSVs (train/val/test) defined in config.path.*_split_file.
    """
    path_cfg = config.path
    required_paths = {
        "train_split_file": path_cfg.train_split_file,
        "val_split_file": path_cfg.val_split_file,
        "test_split_file": path_cfg.test_split_file
    }
    for name, p in required_paths.items():
        if not p:
            raise ValueError(f"{name} is empty; set it in PathConfig when use_pre_split=True.")
        if not os.path.exists(p):
            raise FileNotFoundError(f"{name} not found: {p}")

    train_df = pd.read_csv(path_cfg.train_split_file).reset_index(drop=True)
    val_df = pd.read_csv(path_cfg.val_split_file).reset_index(drop=True)
    test_df = pd.read_csv(path_cfg.test_split_file).reset_index(drop=True)

    train_dataset = _build_dataset(train_df, resources)
    val_dataset = _build_dataset(val_df, resources)
    test_dataset = _build_dataset(test_df, resources)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        collate_fn=data_utils.fast_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        collate_fn=data_utils.fast_collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        collate_fn=data_utils.fast_collate_fn
    )

    return train_loader, val_loader, test_loader


def main():
    config = FullConfig()
    resources = _load_shared_resources(config)
    tokenizer = resources["tokenizer"]
    os.makedirs(config.path.save_dir, exist_ok=True)

    device = torch.device(f'cuda:{config.system.gpu_id}' if torch.cuda.is_available() else 'cpu')

    if config.train.use_pre_split:
        train_loader, val_loader, test_loader = get_pre_split_dataloaders(config, resources)

        drug_encoder, trans_encoder, predictor = build_model(config, tokenizer)

        trainer = Trainer(
            drug_encoder=drug_encoder,
            trans_encoder=trans_encoder,
            predictor=predictor,
            config=config,
            device=device,
            fold=0
        )

        best_val_loss = float("inf")

        for epoch in range(config.train.num_epochs):
            trainer.train_epoch(train_loader, epoch)
            val_loss = trainer.evaluate(val_loader, epoch, tag="Val")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "drug_encoder": drug_encoder.state_dict(),
                    "trans_encoder": trans_encoder.state_dict(),
                    "predictor": predictor.state_dict(),
                }, f"{config.path.save_dir}/best_model_pre_split.pt")

        trainer.evaluate(test_loader, config.train.num_epochs, tag="Test")
        print(f"Pre-split training finished. Best Val Loss: {best_val_loss:.4f}")
    else:
        for fold in range(5):
            print(f"\n======== Training Fold {fold+1}/5 ========")
            
            train_loader, val_loader = get_kfold_dataloaders(config, resources, k=5, fold_idx=fold)

            drug_encoder, trans_encoder, predictor = build_model(config, tokenizer)
            
            trainer = Trainer(
                drug_encoder=drug_encoder,
                trans_encoder=trans_encoder,
                predictor=predictor,
                config=config,
                device=device,
                fold=fold
            )

            best_val_loss = float("inf")

            for epoch in range(config.train.num_epochs):
                trainer.train_epoch(train_loader, epoch)
                val_loss = trainer.evaluate(val_loader, epoch, tag=f"Val-Fold{fold+1}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                torch.save({
                    "drug_encoder": drug_encoder.state_dict(),
                    "trans_encoder": trans_encoder.state_dict(),
                    "predictor": predictor.state_dict(),
                }, f"{config.path.save_dir}/best_model_fold{fold+1}.pt")

            print(f"Fold {fold+1} Finished. Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
