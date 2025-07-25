"""
Basic example of using IRVI with synthetic data.

This example demonstrates how to:
1. Create synthetic single-cell data with TCR sequences
2. Set up the IRVI model
3. Train the model
4. Extract latent representations and embeddings
"""

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from irvi import IRVI

# Set random seed for reproducibility
np.random.seed(42)

def create_synthetic_data(n_cells=1000, n_genes=500):
    """Create synthetic single-cell data with TCR sequences."""
    
    # Generate synthetic gene expression data
    # Simulate different cell types with different expression patterns
    n_cell_types = 3
    cells_per_type = n_cells // n_cell_types
    
    X_list = []
    tcr_list = []
    cell_type_list = []
    
    # TCR sequences for different cell types (simplified)
    tcr_templates = [
        ["CASSLAPGTQVQETQY", "CASSLVGQNTEAFF", "CASRPGQGATEAFF"],  # Type 1
        ["CSVGDTQYF", "CASSPGQGATEAFF", "CASRPGTGELFF"],           # Type 2
        ["CASSQDTQYF", "CASSLVGQGATEAFF", "CASRPGQGTGELFF"]        # Type 3
    ]
    
    for cell_type in range(n_cell_types):
        # Gene expression with cell-type-specific patterns
        base_expression = np.random.negative_binomial(5, 0.3, size=(cells_per_type, n_genes))
        
        # Add cell-type-specific genes
        type_specific_genes = np.arange(cell_type * 50, (cell_type + 1) * 50)
        base_expression[:, type_specific_genes] *= 3  # Higher expression
        
        X_list.append(base_expression)
        
        # Assign TCR sequences
        tcr_seqs = np.random.choice(tcr_templates[cell_type], size=cells_per_type)
        tcr_list.extend(tcr_seqs)
        
        # Cell type labels
        cell_type_list.extend([f"Type_{cell_type}"] * cells_per_type)
    
    # Combine data
    X = np.vstack(X_list).astype(np.float32)
    
    # Handle remaining cells if n_cells doesn't divide evenly
    remaining = n_cells - len(tcr_list)
    if remaining > 0:
        extra_tcr = np.random.choice(tcr_templates[0], size=remaining)
        tcr_list.extend(extra_tcr)
        cell_type_list.extend(["Type_0"] * remaining)
        
        extra_X = np.random.negative_binomial(5, 0.3, size=(remaining, n_genes)).astype(np.float32)
        X = np.vstack([X, extra_X])
    
    # Create AnnData object
    adata = ad.AnnData(X=X[:n_cells])
    adata.obs['TCR'] = tcr_list[:n_cells]
    adata.obs['cell_type'] = cell_type_list[:n_cells]
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]
    
    return adata

def main():
    """Run the complete IRVI analysis example."""
    
    print("Creating synthetic data...")
    adata = create_synthetic_data(n_cells=500, n_genes=200)
    print(f"Created data with {adata.n_obs} cells and {adata.n_vars} genes")
    print(f"Cell types: {adata.obs['cell_type'].value_counts()}")
    print(f"Unique TCR sequences: {len(adata.obs['TCR'].unique())}")
    
    # Setup IRVI model
    print("\nSetting up IRVI model...")
    IRVI.setup_anndata(adata, tcr_key='TCR')
    
    model = IRVI(
        adata,
        n_latent=8,
        n_hidden=64,
        tcr_d_model=32,
        tcr_nhead=4,
        tcr_num_layers=2,
        gene_likelihood="zinb"
    )
    
    print(f"Model summary: {model}")
    
    # Train the model
    print("\nTraining model...")
    model.train(
        max_epochs=50,  # Reduced for demo
        batch_size=128,
        train_size=0.8,
        early_stopping=True,
        check_val_every_n_epoch=5
    )
    
    # Get latent representation
    print("\nExtracting latent representations...")
    latent = model.get_latent_representation()
    
    # Get modality-specific embeddings
    embeddings = model.get_modality_embeddings()
    gene_embedding = embeddings['gene']
    tcr_embedding = embeddings['tcr']
    
    # Get gene expression reconstruction
    reconstruction = model.get_gene_expression_reconstruction()
    
    print(f"Latent representation shape: {latent.shape}")
    print(f"Gene embedding shape: {gene_embedding.shape}")
    print(f"TCR embedding shape: {tcr_embedding.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    
    # Add results to AnnData
    adata.obsm['X_irvi'] = latent
    adata.obsm['gene_embedding'] = gene_embedding
    adata.obsm['tcr_embedding'] = tcr_embedding
    adata.obsm['X_reconstructed'] = reconstruction
    
    # Simple visualization
    print("\nCreating visualizations...")
    
    # Plot latent space colored by cell type
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Latent space (first 2 dimensions)
    for i, cell_type in enumerate(adata.obs['cell_type'].unique()):
        mask = adata.obs['cell_type'] == cell_type
        axes[0].scatter(latent[mask, 0], latent[mask, 1], 
                       label=cell_type, alpha=0.7, s=20)
    axes[0].set_xlabel('Latent Dim 1')
    axes[0].set_ylabel('Latent Dim 2')
    axes[0].set_title('Latent Space (IRVI)')
    axes[0].legend()
    
    # Gene embedding space
    for i, cell_type in enumerate(adata.obs['cell_type'].unique()):
        mask = adata.obs['cell_type'] == cell_type
        axes[1].scatter(gene_embedding[mask, 0], gene_embedding[mask, 1], 
                       label=cell_type, alpha=0.7, s=20)
    axes[1].set_xlabel('Gene Embed Dim 1')
    axes[1].set_ylabel('Gene Embed Dim 2')
    axes[1].set_title('Gene Embedding Space')
    axes[1].legend()
    
    # TCR embedding space
    for i, cell_type in enumerate(adata.obs['cell_type'].unique()):
        mask = adata.obs['cell_type'] == cell_type
        axes[2].scatter(tcr_embedding[mask, 0], tcr_embedding[mask, 1], 
                       label=cell_type, alpha=0.7, s=20)
    axes[2].set_xlabel('TCR Embed Dim 1')
    axes[2].set_ylabel('TCR Embed Dim 2')
    axes[2].set_title('TCR Embedding Space')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('irvi_example_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate reconstruction quality
    correlation = np.corrcoef(adata.X.flatten(), reconstruction.flatten())[0, 1]
    print(f"\nReconstruction correlation: {correlation:.3f}")
    
    print("\nExample completed successfully!")
    print("Results saved in adata.obsm:")
    for key in adata.obsm.keys():
        print(f"  - {key}: {adata.obsm[key].shape}")
    
    return adata, model

if __name__ == "__main__":
    adata, model = main()
