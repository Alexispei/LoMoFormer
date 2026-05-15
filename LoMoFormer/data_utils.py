from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np


def encode_drug(drug_data, global_token):
    """Encode a single drug into motif- and atom-level tensors."""
    seq = torch.tensor([global_token] + drug_data['nums_list'], dtype=torch.long)

    motif_adj = (1 - torch.block_diag(torch.tensor([[1.0]]), torch.tensor(drug_data['adj_matrix'], dtype=torch.float32))) * -1e9
    motif_dist = torch.ones_like(motif_adj)
    motif_dist[1:, 1:] = torch.tensor(drug_data['dist_matrix'], dtype=torch.float32)
    motif_dist[0, 0] = 0

    atom_dict = drug_data['single_dict']
    atom_feat = torch.tensor(atom_dict['input_atom_features'], dtype=torch.float32)
    atom_adj = (1 - torch.tensor(atom_dict['adj_matrix'], dtype=torch.float32)) * -1e9
    atom_dist = torch.tensor(atom_dict['dist_matrix'], dtype=torch.float32)
    atom_match = torch.tensor(atom_dict['atom_match_matrix'], dtype=torch.float32)
    atom_sum = torch.tensor(atom_dict['sum_atoms'], dtype=torch.float32).view(-1)

    return {
        'seq': seq,
        'adj': motif_adj,
        'dist': motif_dist,
        'atom_feat': atom_feat,
        'atom_adj': atom_adj,
        'atom_dist': atom_dist,
        'atom_match': atom_match,
        'atom_sum': atom_sum
    }


def build_gene_expression_dict(cell_df: pd.DataFrame):
    """Lowercase cell ids and convert expression rows to tensors."""
    gene_expr_dict = {}
    for _, row in cell_df.iterrows():
        cell_name = str(row.iloc[0]).lower()
        values = pd.to_numeric(row.iloc[1:], errors='coerce').fillna(0).values.astype(np.float32)
        gene_expr_dict[cell_name] = torch.tensor(values)
    return gene_expr_dict


class FastGraphBertDataset(Dataset):
    def __init__(self, data_set, tokenizer, map_dict, gene_expression_dict, label_field='label'):
        self.data_set = data_set.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.map_dict = map_dict
        self.label_field = label_field
        self.pad_value = tokenizer.vocab.get('<pad>', 0)
        self.global_token = tokenizer.vocab['<global>']
        self.gene_expression_dict = gene_expression_dict

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        row = self.data_set.iloc[idx]
        drug1, drug2 = row['drug1'], row['drug2']
        cell = str(row['cell']).lower()
        label = torch.tensor([float(row[self.label_field])], dtype=torch.float32)

        d1 = encode_drug(self.map_dict[drug1], self.global_token)
        d2 = encode_drug(self.map_dict[drug2], self.global_token)
        gene_expr = self.gene_expression_dict[cell]

        return {
            'seq1': d1['seq'],
            'seq2': d2['seq'],
            'adj1': d1['adj'],
            'adj2': d2['adj'],
            'dist1': d1['dist'],
            'dist2': d2['dist'],
            'atom_feat1': d1['atom_feat'],
            'atom_feat2': d2['atom_feat'],
            'adj1_atom': d1['atom_adj'],
            'adj2_atom': d2['atom_adj'],
            'dist1_atom': d1['atom_dist'],
            'dist2_atom': d2['atom_dist'],
            'match1': d1['atom_match'],
            'match2': d2['atom_match'],
            'sum1': d1['atom_sum'],
            'sum2': d2['atom_sum'],
            'cell': cell,
            'gene_expr': gene_expr,
            'label': label
        }


def fast_collate_fn(batch):
    def pad_tensor_list(tensors, target_shape, pad_val=0.0):
        """Pad list of 1D/2D tensors to a common shape."""
        padded = []
        for t in tensors:
            if t.ndim == 1:
                pad = [0, target_shape[0] - t.shape[0]]
            elif t.ndim == 2:
                pad = [0, target_shape[1] - t.shape[1], 0, target_shape[0] - t.shape[0]]
            else:
                raise ValueError("Only 1D or 2D tensors are supported.")
            padded.append(F.pad(t, pad, value=pad_val))
        return torch.stack(padded)

    def pad_square_matrices(matrices, size, pad_val=-1e9):
        """Pad list of square matrices to (size, size)."""
        return torch.stack([
            F.pad(m, (0, size - m.shape[1], 0, size - m.shape[0]), value=pad_val)
            for m in matrices
        ])

    seq1 = [b['seq1'] for b in batch]
    seq2 = [b['seq2'] for b in batch]
    adj1 = [b['adj1'] for b in batch]
    adj2 = [b['adj2'] for b in batch]
    dist1 = [b['dist1'] for b in batch]
    dist2 = [b['dist2'] for b in batch]
    atom_feat1 = [b['atom_feat1'] for b in batch]
    atom_feat2 = [b['atom_feat2'] for b in batch]
    adj1_atom = [b['adj1_atom'] for b in batch]
    adj2_atom = [b['adj2_atom'] for b in batch]
    dist1_atom = [b['dist1_atom'] for b in batch]
    dist2_atom = [b['dist2_atom'] for b in batch]
    match1 = [b['match1'] for b in batch]
    match2 = [b['match2'] for b in batch]
    sum1 = [b['sum1'] for b in batch]
    sum2 = [b['sum2'] for b in batch]
    cell_names = [b['cell'] for b in batch]
    gene_exprs = [b['gene_expr'] for b in batch]
    labels = [b['label'] for b in batch]

    max_motif = max(max(s.shape[0] for s in seq1), max(s.shape[0] for s in seq2))
    max_atom = max(max(f.shape[0] for f in atom_feat1), max(f.shape[0] for f in atom_feat2))
    match_shape = (max_motif - 1, max_atom)

    batch_dict = {
        "molecule_seq1": pad_tensor_list(seq1, (max_motif,), pad_val=0).long(),
        "molecule_seq2": pad_tensor_list(seq2, (max_motif,), pad_val=0).long(),
        "adjoin1": pad_square_matrices(adj1, max_motif),
        "adjoin2": pad_square_matrices(adj2, max_motif),
        "dist1": pad_square_matrices(dist1, max_motif),
        "dist2": pad_square_matrices(dist2, max_motif),
        "atom_features1": pad_tensor_list(atom_feat1, (max_atom, atom_feat1[0].shape[1]), pad_val=0.0),
        "atom_features2": pad_tensor_list(atom_feat2, (max_atom, atom_feat2[0].shape[1]), pad_val=0.0),
        "adjoin1_atom": pad_square_matrices(adj1_atom, max_atom),
        "adjoin2_atom": pad_square_matrices(adj2_atom, max_atom),
        "dist1_atom": pad_square_matrices(dist1_atom, max_atom),
        "dist2_atom": pad_square_matrices(dist2_atom, max_atom),
        "atom_match1": pad_tensor_list(match1, match_shape, pad_val=0.0),
        "atom_match2": pad_tensor_list(match2, match_shape, pad_val=0.0),
        "sum_atoms1": pad_tensor_list(sum1, (match_shape[0],), pad_val=1.0),
        "sum_atoms2": pad_tensor_list(sum2, (match_shape[0],), pad_val=1.0),
        "cell_name": list(cell_names),
        "gene_expression": torch.stack(gene_exprs),
        "label": torch.cat(labels)
    }

    return batch_dict
