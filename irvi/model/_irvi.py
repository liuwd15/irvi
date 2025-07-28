from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data.fields import LayerField
from scvi.model.base import BaseMinifiedModeModelClass, UnsupervisedTrainingMixin
from scvi.utils import setup_anndata_dsp

from ..data import TCRField
from ..module import IRVAE

if TYPE_CHECKING:
    from typing import Literal

    from anndata import AnnData

logger = logging.getLogger(__name__)


class IRVI(UnsupervisedTrainingMixin, BaseMinifiedModeModelClass):
    """Multimodal Transformer-based model for gene expression and TCR data.
    
    This model integrates single-cell gene expression data with T cell receptor (TCR) 
    amino acid sequences using a transformer-based variational autoencoder architecture.
    The model can extract meaningful latent embeddings that capture both gene expression
    patterns and TCR sequence information.
    
    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~scvi.model.MultimodalTCR.setup_anndata`.
        Gene expression data should be in `adata.X` and TCR sequences in `adata.obsm['TCR']`.
    n_latent
        Dimensionality of the latent space.
    n_hidden
        Number of hidden units in fully connected layers.
    n_layers
        Number of hidden layers for gene expression encoder/decoder.
    tcr_vocab_size
        Vocabulary size for TCR amino acid sequences (default: 25 for 20 AA + special tokens).
    tcr_d_model
        Transformer model dimension for TCR encoder.
    tcr_nhead
        Number of attention heads in TCR transformer.
    tcr_num_layers
        Number of transformer layers for TCR encoder.
    dropout_rate
        Dropout rate applied throughout the model.
    gene_likelihood
        Likelihood function for gene expression data reconstruction.
        
    Examples
    --------
    >>> import anndata as ad
    >>> import numpy as np
    >>> import torch
    >>> 
    >>> # Create example data
    >>> n_obs, n_vars = 1000, 200
    >>> X = np.random.negative_binomial(10, 0.3, size=(n_obs, n_vars))
    >>> 
    >>> # TCR sequences as amino acid strings
    >>> tcr_seqs = ["CASSLAPGTQVQETQY", "CASSLVGQNTEAFF"] * (n_obs // 2)
    >>> if n_obs % 2:
    >>>     tcr_seqs.append("CASRPGQGATEAFF")
    >>> 
    >>> adata = ad.AnnData(X=X)
    >>> adata.obs['TCR'] = tcr_seqs  # Store as strings in obs
    >>> 
    >>> # Setup and train model
    >>> IRVI.setup_anndata(adata, tcr_key='TCR')
    >>> model = IRVI(adata, n_latent=10)
    >>> model.train(max_epochs=100)
    >>> 
    >>> # Get latent embeddings
    >>> latent = model.get_latent_representation()
    """

    def __init__(
        self,
        adata: AnnData | None = None,
        n_latent: int = 10,
        n_hidden: int = 128,
        n_layers: int = 2,
        tcr_vocab_size: int = 25,
        tcr_d_model: int = 128,
        tcr_nhead: int = 8,
        tcr_num_layers: int = 3,
        dropout_rate: float = 0.1,
        gene_likelihood: Literal["zinb", "nb", "gaussian"] = "zinb",
        **model_kwargs,
    ):
        super().__init__(adata, **model_kwargs)

        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.gene_likelihood = gene_likelihood

        if self.adata is not None:
            n_input_genes = self.adata_manager.summary_stats.n_vars
        else:
            n_input_genes = None

        self.module = IRVAE(
            n_input_genes=n_input_genes,
            n_latent=n_latent,
            n_hidden=n_hidden,
            n_layers=n_layers,
            tcr_vocab_size=tcr_vocab_size,
            tcr_d_model=tcr_d_model,
            tcr_nhead=tcr_nhead,
            tcr_num_layers=tcr_num_layers,
            dropout_rate=dropout_rate,
            gene_likelihood=gene_likelihood,
        )
        self._model_summary_string = (
            f"MultimodalTCR Model with the following params: "
            f"n_latent: {n_latent}, n_hidden: {n_hidden}, n_layers: {n_layers}, "
            f"tcr_d_model: {tcr_d_model}, gene_likelihood: {gene_likelihood}"
        )
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls, adata: AnnData, layer: str | None = None, tcr_key: str = "TCR", **kwargs,
    ) -> AnnData:
        """%(summary)s.
        
        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        tcr_key
            Key in `adata.obs` or `adata.obsm` where TCR sequence data is stored. 
            Should contain amino acid sequences as strings (e.g., "CASSLAPGTQVQETQY").
            If in adata.obs, should be a pandas Series of strings.
            If in adata.obsm, can be array of strings or pre-tokenized integers.
        %(param_kwargs)s
            
        Returns
        -------
        %(return_adata)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())

        # Validate TCR data
        if tcr_key not in adata.obs.columns and tcr_key not in adata.obsm:
            raise ValueError(
                f"TCR data not found in adata.obs['{tcr_key}'] or adata.obsm['{tcr_key}']"
            )

        # Register fields
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            TCRField("TCR", tcr_key, max_length=20),
        ]

        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

        return adata

    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: list[int] | None = None,
        give_mean: bool = True,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Return the latent representation for each cell.
        
        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        give_mean
            Give mean of distribution or sample from it.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
            
        Returns
        -------
        Low-dimensional representation for each cell or batch.
        """
        adata = self._validate_anndata(adata)

        if indices is None:
            indices = np.arange(adata.n_obs)
        if batch_size is None:
            batch_size = settings.batch_size

        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        latent = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)

            if give_mean:
                qz_m = outputs["qz_m"]
                latent.append(qz_m.detach().cpu())
            else:
                z = outputs["z"]
                # Handle different z shapes (n_samples, batch_size, features) vs (batch_size, features)
                if z.dim() == 3:
                    z = z.squeeze(0)  # Remove sample dimension
                latent.append(z.detach().cpu())

        return torch.cat(latent, dim=0).numpy()

    def get_gene_expression_reconstruction(
        self,
        adata: AnnData | None = None,
        indices: list[int] | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Return reconstructed gene expression for each cell.
        
        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
            
        Returns
        -------
        Reconstructed gene expression for each cell.
        """
        adata = self._validate_anndata(adata)

        if indices is None:
            indices = np.arange(adata.n_obs)
        if batch_size is None:
            batch_size = settings.batch_size

        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        reconstructed = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            inference_outputs = self.module.inference(**inference_inputs)

            generative_inputs = self.module._get_generative_input(
                tensors, inference_outputs
            )
            generative_outputs = self.module.generative(**generative_inputs)

            px_rate = generative_outputs["px_rate"]
            # Handle different z shapes (n_samples, batch_size, features) vs (batch_size, features)
            if px_rate.dim() == 3:
                # If we have multiple samples, take the first one
                px_rate = px_rate[0]
            reconstructed.append(px_rate.detach().cpu())

        return torch.cat(reconstructed, dim=0).numpy()

    def get_modality_embeddings(
        self,
        adata: AnnData | None = None,
        indices: list[int] | None = None,
        batch_size: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Return separate embeddings for each modality before fusion.
        
        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
            
        Returns
        -------
        Dictionary containing 'gene' and 'tcr' embeddings before fusion.
        """
        adata = self._validate_anndata(adata)

        if indices is None:
            indices = np.arange(adata.n_obs)
        if batch_size is None:
            batch_size = settings.batch_size

        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        gene_embeddings = []
        tcr_embeddings = []

        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)

            gene_embeddings.append(outputs["gene_hidden"].detach().cpu())
            tcr_embeddings.append(outputs["tcr_hidden"].detach().cpu())

        return {
            "gene": torch.cat(gene_embeddings, dim=0).numpy(),
            "tcr": torch.cat(tcr_embeddings, dim=0).numpy(),
        }
