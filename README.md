# IRVI: Immune Repertoire Variation Inference

A multimodal Transformer-based Variational Autoencoder for joint analysis of gene expression and T-cell receptor (TCR) sequence data in single-cell immune repertoire studies.

## Features

- **Multimodal Analysis**: Jointly analyzes single-cell gene expression and T-cell receptor amino acid sequences
- **Transformer Architecture**: Uses transformer encoders to capture TCR sequence patterns
- **Variational Autoencoder**: Learns meaningful latent representations for downstream analysis
- **scvi-tools Integration**: Built on the scvi-tools framework for single-cell analysis
- **Flexible**: Supports different gene expression likelihoods (ZINB, NB, Gaussian)

## Installation

### Install from GitHub

```bash
pip install git+https://github.com/liuwd15/irvi.git
```

### Development Installation

```bash
git clone https://github.com/liuwd15/irvi.git
cd irvi
pip install -e ".[dev]"
```


## Quick Start

```python
import anndata as ad
import numpy as np
from irvi import IRVI

# Create example data
n_obs, n_vars = 1000, 200
X = np.random.negative_binomial(10, 0.3, size=(n_obs, n_vars))

# TCR sequences as amino acid strings
tcr_seqs = ["CASSLAPGTQVQETQY", "CASSLVGQNTEAFF", "CASRPGQGATEAFF"] * (n_obs // 3)
if n_obs % 3:
    tcr_seqs.extend(["CSVGDTQYF"] * (n_obs % 3))

# Create AnnData object
adata = ad.AnnData(X=X.astype(np.float32))
adata.obs['TCR'] = tcr_seqs[:n_obs]
adata.var_names = [f"Gene_{i}" for i in range(n_vars)]

# Setup and train model
IRVI.setup_anndata(adata, tcr_key='TCR')
model = IRVI(adata, n_latent=10, n_hidden=128)

# Train the model
model.train(max_epochs=100)

# Get latent embeddings
latent = model.get_latent_representation()

# Get gene expression reconstruction
reconstruction = model.get_gene_expression_reconstruction()

# Get separate modality embeddings
embeddings = model.get_modality_embeddings()
```

## Data Requirements

### Gene Expression Data
- Should be in `adata.X` as count data (raw UMI counts recommended)
- Genes should be in `adata.var`
- Cells should be in `adata.obs`

### TCR Sequence Data
- Amino acid sequences as strings (e.g., "CASSLAPGTQVQETQY")
- Store in `adata.obs[tcr_key]` where `tcr_key` is specified during setup
- Sequences should represent CDR3 regions of T-cell receptors
- Invalid amino acids will be converted to UNK tokens

## API Reference

### Model Class

#### `IRVI(adata, n_latent=10, n_hidden=128, ...)`

Main model class for multimodal analysis.

**Parameters:**
- `adata`: AnnData object with gene expression and TCR data
- `n_latent`: Dimensionality of latent space (default: 10)
- `n_hidden`: Hidden layer dimensions (default: 128)
- `tcr_d_model`: Transformer model dimension (default: 128)
- `gene_likelihood`: Gene expression likelihood ("zinb", "nb", "gaussian")

#### Key Methods

- `setup_anndata(adata, tcr_key="TCR")`: Prepare data for modeling
- `train(max_epochs=100)`: Train the model
- `get_latent_representation()`: Get latent embeddings
- `get_gene_expression_reconstruction()`: Get reconstructed gene expression
- `get_modality_embeddings()`: Get separate gene and TCR embeddings

## Architecture

The IRVI model consists of:

1. **Gene Expression Encoder**: Fully connected layers for gene expression data
2. **TCR Encoder**: Transformer-based encoder for amino acid sequences
3. **Fusion Layer**: Combines gene and TCR representations
4. **Latent Space**: Variational latent representations
5. **Decoders**: Separate reconstruction for each modality

## Examples

See the `examples/` directory for detailed tutorials on:
- Basic usage with synthetic data
- Analysis of real immune repertoire datasets
- Visualization and interpretation of results
- Integration with scanpy workflows

## Citation

If you use IRVI in your research, please cite:

```
@software{irvi2025,
  author = {Wendao Liu},
  title = {IRVI: Immune Repertoire Variation Inference},
  url = {https://github.com/liuwd15/irvi},
  year = {2025}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Issues

Report issues on [GitHub Issues](https://github.com/liuwd15/irvi/issues).
