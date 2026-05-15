import torch
import torch.nn as nn
import atom_encoder
import moitf_encoder


class CellEncoder(nn.Module):
    def __init__(self, input_dim=978, hidden_dim=512, dropout=0.2):
        super(CellEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):  # x: (batch_size, input_dim)
        return self.encoder(x)  # (batch_size, hidden_dim)


class DrugEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, vocab_size, dff, k=256, dropout=0.1,
                 use_atom_distance_bias=True, use_motif_distance_bias=True,
                 distance_lambda=0.5, distance_decay="linear"):
        super(DrugEncoder, self).__init__()
        self.atom_encoder = atom_encoder.EncoderModel_atom(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            use_distance_bias=use_atom_distance_bias,
            distance_lambda=distance_lambda,
            distance_decay=distance_decay
        )

        self.motif_encoder = moitf_encoder.EncoderModelMotif(
            num_layers=num_layers,
            input_vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            use_distance_bias=use_motif_distance_bias,
            distance_lambda=distance_lambda,
            distance_decay=distance_decay
        )

        self.co_attention = moitf_encoder.CoAttentionLayer(
            graph_feat_size=dff,
            k=k
        )

        self.adapter = nn.Sequential(
            nn.LayerNorm(dff),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dff, dff)
        )

    def forward(self, batch: dict):
        # ---- Atom-Level Features ----
        atom1_features, *_ = self.atom_encoder(
            x=batch['atom_features1'],
            adjoin_matrix=batch['adjoin1_atom'],
            dist_matrix=batch['dist1_atom'],
            atom_match_matrix=batch['atom_match1'],
            sum_atoms=batch['sum_atoms1']
        )

        atom2_features, *_ = self.atom_encoder(
            x=batch['atom_features2'],
            adjoin_matrix=batch['adjoin2_atom'],
            dist_matrix=batch['dist2_atom'],
            atom_match_matrix=batch['atom_match2'],
            sum_atoms=batch['sum_atoms2']
        )

        # ---- Motif-Level Features ----
        motif1_features, *_ = self.motif_encoder(
            x=batch['molecule_seq1'],
            training=self.training,
            atom_level_features=atom1_features,
            adjoin_matrix=batch['adjoin1'],
            dist_matrix=batch['dist1']
        )

        motif2_features, *_ = self.motif_encoder(
            x=batch['molecule_seq2'],
            training=self.training,
            atom_level_features=atom2_features,
            adjoin_matrix=batch['adjoin2'],
            dist_matrix=batch['dist2']
        )

        # ---- Co-Attention ----
        drug1_emb, drug2_emb, _, _ = self.co_attention(motif1_features, motif2_features)

        # ---- Projection ----
        drug1_emb = self.adapter(drug1_emb)
        drug2_emb = self.adapter(drug2_emb)

        return drug1_emb, drug2_emb


class DrugCellPredictor(nn.Module):
    def __init__(self, d_model: int, c_dim: int, hidden_dim: int = 1024, dropout=0.2):
        super(DrugCellPredictor, self).__init__()
        
        input_dim = d_model * 2 + c_dim  # drug1 + drug2 + cell

        self.ln = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 2048), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(2048, 1024), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
    def forward(self, drug1_emb, drug2_emb, cell_emb):
        x = torch.cat([drug1_emb, drug2_emb, cell_emb], dim=-1)
        x = self.ln(x)
        out = self.mlp(x)
        return out  # (batch,)
