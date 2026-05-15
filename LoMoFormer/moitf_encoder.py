import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def rescale_distance_matrix(w):
    constant_value = torch.tensor(1.0, dtype=torch.float32)
    return (constant_value + torch.exp(constant_value))/(constant_value + torch.exp(constant_value-w))

def gelu(x):
    return 0.5 * x * (1 + torch.erf(x / math.sqrt(2.)))

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
    
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    return torch.from_numpy(pos_encoding).float()

def create_padding_mask_motif(batch_data):
    padding_mask = (torch.sum(batch_data, dim=-1) == 0).float()
    return padding_mask.unsqueeze(1).unsqueeze(2)

def scaled_dot_product_attention(q, k, v, mask, adjoin_matrix, dist_matrix,
                                 use_distance_bias=False, distance_lambda=0.5, distance_decay="linear"):
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))
    dk = float(k.size(-1))
    scaled_attention_logits = matmul_qk / math.sqrt(dk)

    if use_distance_bias and dist_matrix is not None:
        # Clamp/clean distance to avoid inf/nan from padded entries (-1e9 padding).
        dist_matrix = torch.where(dist_matrix < 0, torch.full_like(dist_matrix, 1e4), dist_matrix)
        dist_matrix = torch.where(torch.isfinite(dist_matrix), dist_matrix, torch.zeros_like(dist_matrix))
        dist_matrix = dist_matrix.clamp(min=0.0, max=10.0)
        if distance_decay == "exp":
            dist_bias = torch.exp(-distance_lambda * dist_matrix) - 1.0
        else:
            dist_bias = -distance_lambda * dist_matrix
        scaled_attention_logits = scaled_attention_logits + dist_bias

    if adjoin_matrix is not None:
        scaled_attention_logits = scaled_attention_logits + adjoin_matrix
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Guard against any inf/nan that may still appear after masking/bias.
    scaled_attention_logits = torch.nan_to_num(scaled_attention_logits, nan=0.0, posinf=1e4, neginf=-1e4)

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)

    return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, use_distance_bias=False, distance_lambda=0.5, distance_decay="linear"):
        super(MultiHeadAttention, self).__init__()
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.use_distance_bias = use_distance_bias
        self.distance_lambda = distance_lambda
        self.distance_decay = distance_decay
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        self.dense = nn.Linear(d_model, d_model)
        
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, q, k, v, mask, adjoin_matrix, dist_matrix):
        batch_size = q.size(0)
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask, adjoin_matrix, dist_matrix,
            use_distance_bias=self.use_distance_bias,
            distance_lambda=self.distance_lambda,
            distance_decay=self.distance_decay
        )
        
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)
        
        output = self.dense(concat_attention)
        return output, attention_weights

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate, use_distance_bias=True, distance_lambda=0.5, distance_decay="linear"):
        super(EncoderLayer, self).__init__()
        self.use_distance_bias = use_distance_bias
        self.mha1 = MultiHeadAttention(d_model//2, num_heads, use_distance_bias=use_distance_bias,
                                       distance_lambda=distance_lambda, distance_decay=distance_decay)
        self.mha2 = MultiHeadAttention(d_model//2, num_heads, use_distance_bias=use_distance_bias,
                                       distance_lambda=distance_lambda, distance_decay=distance_decay)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.GELU(),
            nn.Linear(dff, d_model)
        )
        
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        
    def forward(self, x, training, encoder_padding_mask, adjoin_matrix, dist_matrix):
        x1, x2 = torch.split(x, x.size(-1)//2, dim=-1)
        
        dist_bias = dist_matrix if self.use_distance_bias else None
        x_l, attention_weights_local = self.mha1(x1, x1, x1, encoder_padding_mask, adjoin_matrix, dist_bias)
        x_g, attention_weights_global = self.mha2(x2, x2, x2, encoder_padding_mask, None, dist_bias)
        
        attn_output = torch.cat([x_l, x_g], dim=-1)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2, attention_weights_local, attention_weights_global


import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderModelMotif(nn.Module):
    def __init__(self, num_layers, input_vocab_size, d_model, num_heads, dff, rate=0.1,
                 use_distance_bias=True, distance_lambda=0.5, distance_decay="linear"):
        super(EncoderModelMotif, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_distance_bias = use_distance_bias
        self.distance_lambda = distance_lambda
        self.distance_decay = distance_decay

        # Embedding layer
        self.embedding = nn.Embedding(input_vocab_size, d_model, padding_idx=0)
        self.dropout = nn.Dropout(rate)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, dff, rate,
                         use_distance_bias=use_distance_bias,
                         distance_lambda=distance_lambda,
                         distance_decay=distance_decay)
            for _ in range(num_layers)
        ])


    

    def forward(self, x, training, atom_level_features, adjoin_matrix=None, dist_matrix=None):
        """
        Forward pass for the motif encoder.

        Args:
            x: Input sequences (batch_size, seq_len).
            training: Boolean indicating training mode.
            atom_level_features: Features at the atom level.
            adjoin_matrix: Adjacency matrix (optional).
            dist_matrix: Distance matrix (optional).

        Returns:
            x: Output feature representations.
            attention_weights_list_local: List of local attention weights.
            attention_weights_list_global: List of global attention weights.
            encoder_padding_mask: Padding mask for the input.
        """
        # encoder_padding_mask = create_padding_mask_motif(x)

        if adjoin_matrix is not None:
            adjoin_matrix = adjoin_matrix.unsqueeze(1)
        if dist_matrix is not None:
            dist_matrix = dist_matrix.unsqueeze(1)

        # Embedding and scaling
        
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        encoder_padding_mask = create_padding_mask_motif(x)
        x = self.dropout(x)
        
        # Incorporate atom-level features
        x_temp = x[:, 1:, :] + atom_level_features
        x = torch.cat([x[:, 0:1, :], x_temp], dim=1)
    
        
        attention_weights_list_local = []
        attention_weights_list_global = []
    
        # Pass through encoder layers
        for i in range(self.num_layers):
            x, attn_local, attn_global = self.encoder_layers[i](
                x, training, encoder_padding_mask, adjoin_matrix, dist_matrix
            )
            attention_weights_list_local.append(attn_local)
            attention_weights_list_global.append(attn_global)

        return x, attention_weights_list_local, attention_weights_list_global, encoder_padding_mask


class CoAttentionLayer(nn.Module):
    def __init__(self, graph_feat_size, k):
        super(CoAttentionLayer, self).__init__()
        self.k = k
        self.graph_feat_size = graph_feat_size

        # Trainable weights
        self.W_m = nn.Parameter(torch.empty(k, graph_feat_size))
        self.W_v = nn.Parameter(torch.empty(k, graph_feat_size))
        self.W_q = nn.Parameter(torch.empty(k, graph_feat_size))
        self.W_h = nn.Parameter(torch.empty(1, k))

        # Initialize weights
        nn.init.xavier_uniform_(self.W_m)
        nn.init.xavier_uniform_(self.W_v)
        nn.init.xavier_uniform_(self.W_q)
        nn.init.normal_(self.W_h, mean=0.0, std=0.02)

    def forward(self, V_n, Q_n):
        """
        Forward pass for co-attention layer.

        Args:
            V_n: Visual features (batch_size, seq_len, graph_feat_size).
            Q_n: Query features (batch_size, seq_len, graph_feat_size).

        Returns:
            vector_v: Co-attended visual feature vector.
            vector_q: Co-attended query feature vector.
            alpha_v: Attention weights for visual features.
            alpha_q: Attention weights for query features.
        """
        V_0 = V_n[:, 0, :].unsqueeze(-1)  # (batch_size, graph_feat_size, 1)
        Q_0 = Q_n[:, 0, :].unsqueeze(-1)  # (batch_size, graph_feat_size, 1)

        V_r = V_n[:, 1:, :]  # (batch_size, graph_feat_size, seq_len-1)
        Q_r = Q_n[:, 1:, :]  # (batch_size, graph_feat_size, seq_len-1)

        # M_0 = V_0 * Q_0  # Element-wise multiplication
        M_0 = (V_n[:, 0, :] * Q_n[:, 0, :]).unsqueeze(1)
        
        # Compute co-attention
        H_v = torch.tanh(F.linear(V_r, self.W_v)) * torch.tanh(F.linear(M_0, self.W_m))
        H_q = torch.tanh(F.linear(Q_r, self.W_q)) * torch.tanh(F.linear(M_0, self.W_m))

        # Attention weights
        alpha_v = F.softmax(torch.matmul(H_v, self.W_h.transpose(0, 1)), dim=1)  # (batch_size, 1, seq_len-1)
        alpha_q = F.softmax(torch.matmul(H_q, self.W_h.transpose(0, 1)), dim=1)  # (batch_size, 1, seq_len-1)

        # Co-attended vectors
        vector_v = torch.sum(alpha_v * V_r, dim=1)  # (batch_size, graph_feat_size)
        vector_q = torch.sum(alpha_q * Q_r, dim=1)  # (batch_size, graph_feat_size)

        return vector_v, vector_q, alpha_v, alpha_q
