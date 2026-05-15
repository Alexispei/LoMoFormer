from config import FullConfig
import models

def build_model(config: FullConfig, tokenizer):
    

    vocab_size = tokenizer.get_vocab_size

    drug_encoder = models.DrugEncoder(
        num_layers=config.drug.num_layers,
        d_model=config.drug.d_model,
        num_heads=config.drug.num_heads,
        vocab_size=vocab_size,
        dff=config.drug.dff,
        k=config.drug.coattention_k,
        dropout=config.drug.dropout_rate,
        use_atom_distance_bias=config.drug.use_atom_distance_bias,
        use_motif_distance_bias=config.drug.use_motif_distance_bias,
        distance_lambda=config.drug.distance_lambda,
        distance_decay=config.drug.distance_decay
    )

    trans_encoder = models.CellEncoder(
        input_dim=config.transcript.input_dim,
        hidden_dim=config.transcript.hidden_dim,
        dropout=config.transcript.dropout_rate
    )
    
    predictor = models.DrugCellPredictor(
        d_model=config.drug.dff,
        c_dim=config.transcript.hidden_dim,
        hidden_dim=config.predictor.hidden_dim,
        dropout=config.predictor.dropout_rate
    )

    return drug_encoder, trans_encoder, predictor
