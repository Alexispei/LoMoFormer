from dataclasses import dataclass, field

@dataclass
class PathConfig:
    data_folder: str = '/root/FuncAwareSynergy/FuncAwareSynergy/data/DrugComb/items/drugcombProteinExpMapped.csv'
    molecular_info: str = '/root/FuncAwareSynergy/FuncAwareSynergy/data/DrugComb/features/preprocessed_molecular_drugcomb_drug.npy'
    cell_features: str = '/root/FuncAwareSynergy/FuncAwareSynergy/data/DrugComb/features/cell_exp_normal.csv'
    token_id_file: str = '/root/FuncAwareSynergy/FuncAwareSynergy/data/DrugComb/features/token_id.json'

    # Optional pre-split files (train/val/test). Leave empty to use k-fold.
    train_split_file: str = ''
    val_split_file: str = ''
    test_split_file: str = ''

    save_dir: str = '/root/FuncAwareSynergy/FuncAwareSynergy/data/DrugComb/saveDir/path_result'
    tensorboard_log_dir: str = '/root/FuncAwareSynergy/FuncAwareSynergy/data/DrugComb/saveDir/tensorboard'
    tensorboard_log_name: str = 'Drugcomb_regression'


@dataclass
class DrugEncoderConfig:
    d_model: int = 512
    dff: int = 512
    num_heads: int = 8
    num_layers: int = 3
    dropout_rate: float = 0.1
    coattention_k: int = 64
    # ICIC: distance-decayed attention bias
    use_atom_distance_bias: bool = True
    use_motif_distance_bias: bool = True
    distance_lambda: float = 0.5
    distance_decay: str = "linear"  # "linear" -> -lambda*SPD, "exp" -> exp(-lambda*SPD)-1


@dataclass
class TranscriptomicConfig:
    input_dim: int = 963
    hidden_dim: int = 512
    dropout_rate: float = 0.2

@dataclass
class TrainingConfig:
    batch_size: int = 256
    num_workers: int = 4
    learning_rate: float = 2e-4
    beta1: float = 0.9
    beta2: float = 0.999
    amsgrad: bool = True
    num_epochs: int = 100
    weight_decay: float = 5e-5
    use_pre_split: bool = True


@dataclass
class SystemConfig:
    gpu_id: int = 0
    note: str = ''

@dataclass
class PredictorConfig:
    dropout_rate: float = 0.2
    hidden_dim: int = 2048
    
    

@dataclass
class FullConfig:
    path: PathConfig = field(default_factory=PathConfig)
    drug: DrugEncoderConfig = field(default_factory=DrugEncoderConfig)
    transcript: TranscriptomicConfig = field(default_factory=TranscriptomicConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
