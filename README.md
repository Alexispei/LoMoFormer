# LoMoFormer

LoMoFormer is a PyTorch project for drug combination synergy prediction. It encodes two drugs with atom-level and motif-level Transformer modules, combines the two drug representations with co-attention, and integrates cell-line gene expression features to predict a synergy score.

## Project Structure

```text
LoMoFormer/
|-- LoMoFormer/
|   |-- atom_encoder.py      # Atom-level Transformer encoder
|   |-- moitf_encoder.py     # Motif-level Transformer encoder and co-attention
|   |-- models.py            # Drug encoder, cell encoder, and predictor modules
|   |-- model_builder.py     # Model construction from config
|   |-- data_utils.py        # Dataset and batch collation utilities
|   |-- features.py          # Molecular feature helpers
|   |-- trainer.py           # Training and evaluation loop
|   |-- train.py             # Main training entry point
|   |-- config.py            # Paths and hyperparameters
|   `-- utils.py             # Molecular graph and tokenizer utilities
|-- data/                    # Local datasets; ignored by Git
|-- .gitignore
`-- README.md
```

## Main Features

- Dual drug representation using atom-level and motif-level molecular encoders.
- Distance-aware attention bias for atom and motif graph structure.
- Co-attention module for modeling interactions between the two drugs.
- Cell-line transcriptomic encoder for gene expression features.
- Regression training with MSE loss and evaluation metrics including RMSE, MAE, R2, Pearson, and Spearman.
- Supports either predefined train/validation/test splits or 5-fold cross validation.

## Requirements

The project uses Python and PyTorch. Main dependencies include:

- `torch`
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `tqdm`
- `tensorboard`
- `rdkit`
- `IPython`

Install dependencies in your preferred environment. For example:

```bash
pip install torch numpy pandas scikit-learn scipy tqdm tensorboard rdkit ipython
```

If your platform has trouble installing RDKit through `pip`, install it with Conda:

```bash
conda install -c conda-forge rdkit
```

## Data

Dataset files are not tracked in this repository. Place them under the local `data/` directory, or update the paths in `LoMoFormer/config.py`.

The training code expects files such as:

- item CSV files containing drug pairs, cell-line identifiers, and labels
- preprocessed molecular feature `.npy` files
- cell expression feature CSV files
- molecular tokenizer JSON files

Configure these paths in `PathConfig`:

```python
data_folder = "path/to/items.csv"
molecular_info = "path/to/preprocessed_molecular.npy"
cell_features = "path/to/cell_exp_normal.csv"
token_id_file = "path/to/token_id.json"
train_split_file = "path/to/train.csv"
val_split_file = "path/to/val.csv"
test_split_file = "path/to/test.csv"
save_dir = "path/to/save_dir"
tensorboard_log_dir = "path/to/tensorboard"
```

## Training

Edit `LoMoFormer/config.py` first to match your dataset and output paths.

Run training from the source directory:

```bash
cd LoMoFormer
python train.py
```

By default, `TrainingConfig.use_pre_split` is `True`, so the code uses `train_split_file`, `val_split_file`, and `test_split_file`. Set it to `False` to use 5-fold cross validation from `data_folder`.

Model checkpoints are saved to `PathConfig.save_dir`, and TensorBoard logs are written to `PathConfig.tensorboard_log_dir`.

## Configuration

Important options are defined in `LoMoFormer/config.py`:

- `DrugEncoderConfig`: model dimension, feed-forward dimension, number of heads, number of layers, dropout, co-attention size, and distance-bias options.
- `TranscriptomicConfig`: gene expression input dimension and hidden dimension.
- `PredictorConfig`: predictor hidden dimension and dropout.
- `TrainingConfig`: batch size, learning rate, optimizer settings, number of epochs, weight decay, and split mode.
- `SystemConfig`: GPU device id.

## Outputs

Training writes:

- best model checkpoints, such as `best_model_pre_split.pt` or `best_model_fold*.pt`
- TensorBoard event files
- printed validation/test metrics during training

## Notes

- Large datasets, checkpoints, logs, and generated outputs are ignored by Git.
- Before running on a new machine, verify that all paths in `LoMoFormer/config.py` exist.
- The code uses CUDA automatically when available; otherwise it falls back to CPU.
