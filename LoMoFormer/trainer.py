import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

class Trainer:
    def __init__(self, drug_encoder, trans_encoder, predictor, config, device, fold):
        self.drug_encoder = drug_encoder.to(device)
        self.trans_encoder = trans_encoder.to(device)
        self.predictor = predictor.to(device)

        self.config = config
        self.device = device
        self.fold = fold
        self.criterion = nn.MSELoss()

        self.optimizer = torch.optim.AdamW(
            list(self.drug_encoder.parameters()) +
            list(self.trans_encoder.parameters()) +
            list(self.predictor.parameters()),
            lr=config.train.learning_rate,
            betas=(config.train.beta1, config.train.beta2),
            weight_decay=config.train.weight_decay,
            amsgrad=config.train.amsgrad
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(config.path.tensorboard_log_dir, f"run_fold{self.fold}_{timestamp}")
        
        self.writer = SummaryWriter(log_dir=log_dir)

        self.writer.add_hparams(
            {
                'drug_d_model': config.drug.d_model,
                'drug_dff': config.drug.dff,
                'drug_num_heads': config.drug.num_heads,
                'drug_num_layers': config.drug.num_layers,
                'drug_dropout': config.drug.dropout_rate,
                'drug_coatt_k': config.drug.coattention_k,

                'trans_input_dim': config.transcript.input_dim,
                'trans_hidden_dim': config.transcript.hidden_dim,
                'trans_dropout': config.transcript.dropout_rate,

                'lr': config.train.learning_rate,
                'batch_size': config.train.batch_size,
                'weight_decay': config.train.weight_decay,
                'fold': self.fold
            },
            {
                'hparam/init_placeholder': 0.0
            }
        )
    def train_epoch(self, dataloader, epoch):
        self.drug_encoder.train()
        self.trans_encoder.train()
        self.predictor.train()

        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        all_preds = []
        all_labels = []

        for batch in loop:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(self.device)

            drug1_emb, drug2_emb = self.drug_encoder(batch)
            gene_emb = self.trans_encoder(batch["gene_expression"])
            pred = self.predictor(drug1_emb, drug2_emb, gene_emb)

            label = batch["label"].view(-1, 1).to(self.device)
            loss = self.criterion(pred, label)

            all_preds.append(pred.detach().cpu())
            all_labels.append(label.detach().cpu())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            # print("Batch label:", label[:5].cpu().numpy())
            # print("Pred:", pred[:5].detach().cpu().numpy())
            # print("Loss:", loss.item())

        all_preds = torch.cat(all_preds).numpy().flatten()
        all_labels = torch.cat(all_labels).numpy().flatten()
        rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
        mae = mean_absolute_error(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds)
        pearson = pearsonr(all_labels, all_preds)[0]
        spearman = spearmanr(all_labels, all_preds)[0]
        avg_loss = total_loss / len(dataloader)

        self.writer.add_scalar(f"Fold{self.fold}Train/RMSE", rmse, epoch)
        self.writer.add_scalar(f"Fold{self.fold}Train/MAE", mae, epoch)
        self.writer.add_scalar(f"Fold{self.fold}Train/R2", r2, epoch)
        self.writer.add_scalar(f"Fold{self.fold}Train/Pearson", pearson, epoch)
        self.writer.add_scalar(f"Fold{self.fold}Train/Spearman", spearman, epoch)
        self.writer.add_scalar(f"Fold{self.fold}Train/Loss", avg_loss, epoch)
        return avg_loss

    def evaluate(self, dataloader, epoch, tag="Val", return_metrics=False):
        self.drug_encoder.eval()
        self.trans_encoder.eval()
        self.predictor.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.device)

                drug1_emb, drug2_emb = self.drug_encoder(batch)
                gene_emb = self.trans_encoder(batch["gene_expression"])
                pred = self.predictor(drug1_emb, drug2_emb, gene_emb)

                label = batch["label"].view(-1, 1).to(self.device)
                loss = self.criterion(pred, label)
                
                total_loss += loss.item()
                all_preds.append(pred.detach().cpu())
                all_labels.append(label.detach().cpu())
        all_preds = torch.cat(all_preds).numpy().flatten()
        all_labels = torch.cat(all_labels).numpy().flatten()
        
        rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
        mae = mean_absolute_error(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds)
        pearson = pearsonr(all_labels, all_preds)[0]
        spearman = spearmanr(all_labels, all_preds)[0]
        avg_loss = total_loss / len(dataloader)
        
        self.writer.add_scalar(f"Fold{self.fold}/{tag}/Loss", avg_loss, epoch)
        self.writer.add_scalar(f"Fold{self.fold}/{tag}/MAE", mae, epoch)
        self.writer.add_scalar(f"Fold{self.fold}/{tag}/RMSE", rmse, epoch)
        self.writer.add_scalar(f"Fold{self.fold}/{tag}/R2", r2, epoch)
        self.writer.add_scalar(f"Fold{self.fold}/{tag}/Pearson", pearson, epoch)
        self.writer.add_scalar(f"Fold{self.fold}/{tag}/Spearman", spearman, epoch)

        if return_metrics:
            return {
                "loss": avg_loss,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "pearson": pearson,
                "spearman": spearman
            }
        return avg_loss
