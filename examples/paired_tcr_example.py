"""
Example of using IRVI with paired TCR sequences (alpha and beta chains).

This example demonstrates how to:
1. Create paired TCR sequences using the separator token '_'
2. Use IRVI with paired TCR data
3. Extract embeddings for both chains combined
"""

import anndata as ad
import numpy as np

from irvi import IRVI
from irvi.module import AminoAcidTokenizer

# Set random seed for reproducibility
np.random.seed(42)


def create_paired_tcr_data(n_cells=500, n_genes=200):
    """Create synthetic data with paired TCR sequences (alpha and beta chains)."""

    # Generate synthetic gene expression data
    X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes)).astype(np.float32)

    # Create paired TCR sequences (alpha_beta format)
    # Example TCR alpha chains
    alpha_chains = [
        "CAVNTGNQFYF",
        "CALSEAGFQKLVF",
        "CAASISKKLFF",
        "CILRVGDTQYF",
        "CAVRNTGGKLTF",
    ]

    # Example TCR beta chains
    beta_chains = [
        "CASSLAPGTQVQETQY",
        "CASSLVGQNTEAFF",
        "CASRPGQGATEAFF",
        "CSVGDTQYF",
        "CASSPGQGATEAFF",
    ]

    # Create tokenizer to help with pairing
    tokenizer = AminoAcidTokenizer()

    # Generate paired sequences
    paired_tcr_sequences = []
    for i in range(n_cells):
        alpha = np.random.choice(alpha_chains)
        beta = np.random.choice(beta_chains)
        # Use the tokenizer's convenience method to create paired sequences
        paired_seq = tokenizer.create_paired_sequence(alpha, beta)
        paired_tcr_sequences.append(paired_seq)

    # Create AnnData object
    adata = ad.AnnData(X=X)
    adata.obs["paired_TCR"] = paired_tcr_sequences
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]

    return adata


def demonstrate_separator_token():
    """Demonstrate the separator token functionality."""
    print("=== Separator Token Demonstration ===")

    tokenizer = AminoAcidTokenizer(max_length=30, add_special_tokens=True)

    # Create paired sequence
    alpha = "CAVNTGNQFYF"
    beta = "CASSLAPGTQVQETQY"
    paired = tokenizer.create_paired_sequence(alpha, beta)

    print(f"Alpha chain: {alpha}")
    print(f"Beta chain: {beta}")
    print(f"Paired sequence: {paired}")
    print(f"Separator token ID: {tokenizer.sep_token_id}")
    print(f"Separator token: '{tokenizer.inverse_vocab[tokenizer.sep_token_id]}'")

    # Encode and decode
    encoded = tokenizer.encode(paired)
    decoded = tokenizer.decode(encoded)

    print(f"Encoded length: {len(encoded)}")
    print(f"Contains separator: {'_' in decoded}")
    print(f"Decoded: {decoded}")

    # Split back into individual chains
    split_alpha, split_beta = tokenizer.split_paired_sequence(decoded)
    print(f"Split alpha: {split_alpha}")
    print(f"Split beta: {split_beta}")
    print()


def main():
    """Run the complete paired TCR analysis example."""

    # Demonstrate separator token first
    demonstrate_separator_token()

    print("=== Paired TCR Analysis with IRVI ===")

    # Create data with paired TCR sequences
    print("Creating synthetic data with paired TCR sequences...")
    adata = create_paired_tcr_data(n_cells=300, n_genes=150)

    print(f"Created data with {adata.n_obs} cells and {adata.n_vars} genes")
    print(f"Sample paired TCR sequences:")
    for i in range(5):
        print(f"  Cell {i}: {adata.obs['paired_TCR'].iloc[i]}")

    # Setup IRVI model
    print("\nSetting up IRVI model for paired TCR data...")
    IRVI.setup_anndata(adata, tcr_key="paired_TCR")

    model = IRVI(
        adata,
        n_latent=10,
        n_hidden=64,
        tcr_d_model=32,
        tcr_nhead=4,
        tcr_num_layers=2,
        gene_likelihood="zinb",
    )

    print(f"Model vocab size: {model.module.tcr_tokenizer.vocab_size}")
    print(f"Separator token supported: {'_' in model.module.tcr_tokenizer.vocab}")

    # Get latent representation
    print("\nExtracting latent representations...")
    latent = model.get_latent_representation()

    # Get modality-specific embeddings
    embeddings = model.get_modality_embeddings()
    gene_embedding = embeddings["gene"]
    tcr_embedding = embeddings["tcr"]  # This includes both alpha and beta chain info

    print(f"Latent representation shape: {latent.shape}")
    print(f"Gene embedding shape: {gene_embedding.shape}")
    print(f"TCR embedding shape: {tcr_embedding.shape}")

    # Add results to AnnData
    adata.obsm["X_irvi"] = latent
    adata.obsm["gene_embedding"] = gene_embedding
    adata.obsm["paired_tcr_embedding"] = tcr_embedding

    print("\nAnalysis completed successfully!")
    print("The TCR embeddings now contain information from both alpha and beta chains.")
    print("\nResults stored in adata.obsm:")
    for key in adata.obsm.keys():
        print(f"  - {key}: {adata.obsm[key].shape}")

    # Demonstrate how to access individual chain information if needed
    print("\n=== Individual Chain Analysis ===")
    tokenizer = AminoAcidTokenizer()

    sample_paired_tcr = adata.obs["paired_TCR"].iloc[0]
    alpha_chain, beta_chain = tokenizer.split_paired_sequence(sample_paired_tcr)

    print(f"Sample paired TCR: {sample_paired_tcr}")
    print(f"Alpha chain: {alpha_chain}")
    print(f"Beta chain: {beta_chain}")

    return adata, model


if __name__ == "__main__":
    adata, model = main()
