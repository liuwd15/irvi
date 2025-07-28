from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from scvi import REGISTRY_KEYS
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseMinifiedModeModuleClass, LossOutput, auto_move_data
from scvi.nn import FCLayers
from torch.distributions import Normal

if TYPE_CHECKING:
    from typing import Literal

logger = logging.getLogger(__name__)


class AminoAcidTokenizer:
    """Tokenizer for amino acid sequences."""

    # Standard 20 amino acids
    AMINO_ACIDS = [
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "E",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
    ]

    def __init__(self, max_length: int = 20, add_special_tokens: bool = True):
        """
        Initialize amino acid tokenizer.

        Parameters
        ----------
        max_length : int
            Maximum sequence length
        add_special_tokens : bool
            Whether to include special tokens (PAD, UNK, etc.)
        """
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens

        # Build vocabulary
        self.vocab = {}
        self.inverse_vocab = {}

        # Special tokens
        if add_special_tokens:
            self.vocab["<PAD>"] = 0
            self.vocab["<UNK>"] = 1
            self.vocab["<START>"] = 2
            self.vocab["<END>"] = 3
            start_idx = 4
        else:
            self.vocab["<PAD>"] = 0  # Always need padding
            start_idx = 1

        # Add amino acids
        for i, aa in enumerate(self.AMINO_ACIDS):
            self.vocab[aa] = start_idx + i

        # Create inverse mapping
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        self.vocab_size = len(self.vocab)
        self.pad_token_id = self.vocab["<PAD>"]
        self.unk_token_id = self.vocab.get("<UNK>", 0)

    def encode(self, sequence: str, add_special_tokens: bool = None) -> torch.Tensor:
        """
        Encode amino acid sequence to token IDs.

        Parameters
        ----------
        sequence : str
            Amino acid sequence (e.g., "CASSLAPGTQVQETQY")
        add_special_tokens : bool, optional
            Override class setting for special tokens

        Returns
        -------
        torch.Tensor
            Token IDs tensor of shape (max_length,)
        """
        if add_special_tokens is None:
            add_special_tokens = self.add_special_tokens

        # Clean sequence
        sequence = sequence.upper().strip()

        # Convert to token IDs
        token_ids = []

        if add_special_tokens and "<START>" in self.vocab:
            token_ids.append(self.vocab["<START>"])
            max_seq_len = self.max_length - 2  # Reserve space for START and END
        else:
            max_seq_len = self.max_length

        # Encode amino acids
        for aa in sequence[:max_seq_len]:
            if aa in self.vocab:
                token_ids.append(self.vocab[aa])
            else:
                token_ids.append(self.unk_token_id)

        if add_special_tokens and "<END>" in self.vocab:
            token_ids.append(self.vocab["<END>"])

        # Pad to max_length
        while len(token_ids) < self.max_length:
            token_ids.append(self.pad_token_id)

        return torch.tensor(token_ids, dtype=torch.long)

    def encode_batch(self, sequences: list[str]) -> torch.Tensor:
        """
        Encode batch of sequences.

        Parameters
        ----------
        sequences : list[str]
            List of amino acid sequences

        Returns
        -------
        torch.Tensor
            Token IDs tensor of shape (batch_size, max_length)
        """
        return torch.stack([self.encode(seq) for seq in sequences])

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to amino acid sequence.

        Parameters
        ----------
        token_ids : torch.Tensor
            Token IDs tensor
        skip_special_tokens : bool
            Whether to skip special tokens in output

        Returns
        -------
        str
            Decoded amino acid sequence
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        sequence = []
        for token_id in token_ids:
            if token_id == self.pad_token_id:
                break  # Stop at first padding token

            token = self.inverse_vocab.get(token_id, "<UNK>")

            if skip_special_tokens and token.startswith("<"):
                continue

            sequence.append(token)

        return "".join(sequence)

    def create_attention_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Create attention mask for sequences.

        Parameters
        ----------
        token_ids : torch.Tensor
            Token IDs tensor of shape (batch_size, seq_len) or (seq_len,)

        Returns
        -------
        torch.Tensor
            Attention mask (1 for real tokens, 0 for padding)
        """
        return (token_ids != self.pad_token_id).long()


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 50):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Add positional encoding to input embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, : x.size(1)]


class TCREncoder(nn.Module):
    """Transformer encoder for T cell receptor amino acid sequences."""

    def __init__(
        self,
        max_length: int = 20,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        add_special_tokens: bool = True,
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.use_positional_encoding = use_positional_encoding

        # Initialize tokenizer
        self.tokenizer = AminoAcidTokenizer(
            max_length=max_length, add_special_tokens=add_special_tokens
        )

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.tokenizer.vocab_size,
            embedding_dim=d_model,
            padding_idx=self.tokenizer.pad_token_id,
        )

        # Positional encoding
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(d_model, max_length)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Pooling strategy
        self.pooling_strategy = (
            "attention_weighted"  # Options: "mean", "max", "attention_weighted"
        )

        if self.pooling_strategy == "attention_weighted":
            self.attention_pooling = nn.Sequential(
                nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1)
            )

    def encode_sequences(self, sequences: list[str]) -> torch.Tensor:
        """
        Encode string sequences to token IDs.

        Parameters
        ----------
        sequences : list[str]
            List of amino acid sequences

        Returns
        -------
        torch.Tensor
            Token IDs tensor of shape (batch_size, max_length)
        """
        return self.tokenizer.encode_batch(sequences)

    def forward(self, inputs, attention_mask=None):
        """
        Forward pass through TCR encoder.

        Parameters
        ----------
        inputs : torch.Tensor or list[str]
            Either token IDs tensor of shape (batch_size, seq_len) or list of sequence strings
        attention_mask : torch.Tensor, optional
            Attention mask of shape (batch_size, seq_len). If None, will be created from inputs.
        """
        # Handle different input types
        if isinstance(inputs, (list, tuple)) and isinstance(inputs[0], str):
            # Convert string sequences to token IDs
            token_ids = self.encode_sequences(inputs)
            if token_ids.device != next(self.parameters()).device:
                token_ids = token_ids.to(next(self.parameters()).device)
        else:
            # Assume inputs are already token IDs
            token_ids = inputs

        # Ensure token_ids are integer type for embedding layer
        if token_ids.dtype != torch.long:
            token_ids = token_ids.long()

        batch_size, seq_len = token_ids.shape

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = self.tokenizer.create_attention_mask(token_ids)

        # Embedding with scaling (as in original Transformer paper)
        embeddings = self.embedding(token_ids) * torch.sqrt(
            torch.tensor(self.d_model, dtype=torch.float32, device=token_ids.device)
        )

        # Add positional encoding
        if self.use_positional_encoding:
            embeddings = self.pos_encoder(embeddings)

        embeddings = self.dropout(embeddings)

        # Create padding mask for transformer (True for positions to ignore)
        src_key_padding_mask = attention_mask == 0

        # Pass through transformer
        transformer_output = self.transformer_encoder(
            embeddings, src_key_padding_mask=src_key_padding_mask
        )

        # Apply layer normalization
        transformer_output = self.layer_norm(transformer_output)

        # Pooling to get sequence-level representation
        sequence_repr = self._pool_sequence(transformer_output, attention_mask)

        return sequence_repr

    def _pool_sequence(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool sequence representations to get single vector per sequence.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Hidden states from transformer of shape (batch_size, seq_len, d_model)
        attention_mask : torch.Tensor
            Attention mask of shape (batch_size, seq_len)

        Returns
        -------
        torch.Tensor
            Pooled representation of shape (batch_size, d_model)
        """
        if self.pooling_strategy == "mean":
            # Masked average pooling
            mask_expanded = attention_mask.unsqueeze(-1).float()
            masked_hidden = hidden_states * mask_expanded
            seq_lengths = attention_mask.sum(dim=1, keepdim=True).float()
            return masked_hidden.sum(dim=1) / seq_lengths.clamp(min=1)

        elif self.pooling_strategy == "max":
            # Max pooling with masking
            mask_expanded = attention_mask.unsqueeze(-1).float()
            masked_hidden = hidden_states * mask_expanded + (1 - mask_expanded) * (-1e9)
            return masked_hidden.max(dim=1)[0]

        elif self.pooling_strategy == "attention_weighted":
            # Attention-weighted pooling
            attention_weights = self.attention_pooling(hidden_states).squeeze(
                -1
            )  # (batch_size, seq_len)

            # Mask attention weights
            attention_weights = attention_weights.masked_fill(attention_mask == 0, -1e9)
            attention_weights = F.softmax(
                attention_weights, dim=1
            )  # (batch_size, seq_len)

            # Weighted sum
            weighted_repr = torch.bmm(
                attention_weights.unsqueeze(1), hidden_states
            ).squeeze(
                1
            )  # (batch_size, d_model)
            return weighted_repr

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    def get_attention_weights(self, inputs, attention_mask=None):
        """
        Get attention weights for visualization.

        Parameters
        ----------
        inputs : torch.Tensor or list[str]
            Input sequences
        attention_mask : torch.Tensor, optional
            Attention mask

        Returns
        -------
        dict
            Dictionary containing attention weights from each layer
        """
        # This would require modifying the transformer to return attention weights
        # For now, return None
        return None


class IRVAE(BaseMinifiedModeModuleClass):
    """Multimodal Transformer-based Variational Autoencoder for gene expression and TCR data.

    This model combines single-cell gene expression data with T cell receptor
    amino acid sequences using a transformer-based architecture within a VAE framework.

    Parameters
    ----------
    n_input_genes
        Number of input genes (features from adata.X)
    n_latent
        Dimensionality of the latent space
    n_hidden
        Number of hidden units in fully connected layers
    n_layers
        Number of hidden layers for gene expression encoder/decoder
    tcr_vocab_size
        Vocabulary size for TCR amino acid sequences (default: 25)
    tcr_d_model
        Transformer model dimension for TCR encoder
    tcr_nhead
        Number of attention heads in TCR transformer
    tcr_num_layers
        Number of transformer layers for TCR encoder
    dropout_rate
        Dropout rate
    gene_likelihood
        Likelihood function for gene expression data
    """

    def __init__(
        self,
        n_input_genes: int,
        n_latent: int = 10,
        n_hidden: int = 128,
        n_layers: int = 2,
        tcr_vocab_size: int = 25,
        tcr_d_model: int = 128,
        tcr_nhead: int = 8,
        tcr_num_layers: int = 3,
        dropout_rate: float = 0.1,
        gene_likelihood: Literal["zinb", "nb", "gaussian"] = "zinb",
    ):
        super().__init__()

        self.n_input_genes = n_input_genes
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.gene_likelihood = gene_likelihood

        # Gene expression encoder
        self.gene_encoder = FCLayers(
            n_in=n_input_genes,
            n_out=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=True,
            use_activation=True,
        )

        # TCR sequence encoder
        self.tcr_encoder = TCREncoder(
            max_length=20,  # Adjust based on your data
            d_model=tcr_d_model,
            nhead=tcr_nhead,
            num_layers=tcr_num_layers,
            dropout=dropout_rate,
            add_special_tokens=True,
            use_positional_encoding=True,
        )

        # Get actual vocab size and max length from tokenizer
        actual_vocab_size = self.tcr_encoder.tokenizer.vocab_size
        actual_max_length = self.tcr_encoder.tokenizer.max_length

        # Fusion layer - combines gene and TCR representations
        fusion_input_dim = n_hidden + tcr_d_model
        self.fusion_layer = FCLayers(
            n_in=fusion_input_dim,
            n_out=n_hidden,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=True,
            use_activation=True,
        )

        # Latent space - mean and variance
        self.z_mean_encoder = nn.Linear(n_hidden, n_latent)
        self.z_var_encoder = nn.Linear(n_hidden, n_latent)

        # Decoder for gene expression
        self.gene_decoder = FCLayers(
            n_in=n_latent,
            n_out=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=True,
            use_activation=True,
        )

        # Output layers for gene expression
        if gene_likelihood in ["zinb", "nb"]:
            self.px_scale_decoder = nn.Sequential(
                nn.Linear(n_hidden, n_input_genes), nn.Softmax(dim=-1)
            )
            self.px_rate_decoder = nn.Linear(n_hidden, n_input_genes)
            if gene_likelihood == "zinb":
                self.px_dropout_decoder = nn.Linear(n_hidden, n_input_genes)
            else:
                self.px_dropout_decoder = None
        else:  # gaussian
            self.px_scale_decoder = nn.Linear(n_hidden, n_input_genes)
            self.px_rate_decoder = nn.Linear(n_hidden, n_input_genes)
            self.px_dropout_decoder = None

        # TCR reconstruction decoder (optional - for reconstruction loss)
        self.tcr_decoder = nn.Sequential(
            nn.Linear(n_latent, tcr_d_model),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(
                tcr_d_model, actual_vocab_size * actual_max_length
            ),  # Use actual dimensions
        )

        # Store tokenizer reference for loss computation
        self.tcr_tokenizer = None  # Will be set when TCR encoder is created

        # Set tokenizer reference after creating TCR encoder
        self.tcr_tokenizer = self.tcr_encoder.tokenizer

    @auto_move_data
    def _get_inference_input(self, tensors):
        """Parse the input tensors."""
        x = tensors[REGISTRY_KEYS.X_KEY]
        tcr = tensors["TCR"]  # Assuming this key exists

        input_dict = {
            "x": x,
            "tcr": tcr,
        }
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        library = inference_outputs["library"]

        input_dict = {
            "z": z,
            "library": library,
        }
        return input_dict

    @auto_move_data
    def inference(
        self,
        x,
        tcr,
        n_samples=1,
    ):
        """High level inference method.

        Parameters
        ----------
        x : torch.Tensor
            Gene expression data of shape (batch_size, n_genes)
        tcr : torch.Tensor or list[str]
            TCR sequence data. Can be:
            - torch.Tensor of token IDs of shape (batch_size, seq_len)
            - list of amino acid sequence strings
        n_samples : int
            Number of samples to draw from the posterior
        """
        # Encode gene expression
        gene_hidden = self.gene_encoder(x)

        # Encode TCR sequences - the encoder handles both string and token inputs
        tcr_hidden = self.tcr_encoder(tcr)

        # Fuse modalities
        fused_repr = torch.cat([gene_hidden, tcr_hidden], dim=-1)
        fused_hidden = self.fusion_layer(fused_repr)

        # Get latent parameters
        qz_m = self.z_mean_encoder(fused_hidden)
        qz_v = torch.exp(self.z_var_encoder(fused_hidden)) + 1e-4

        # Sample from posterior
        qz = Normal(qz_m, qz_v.sqrt())
        z = qz.rsample(sample_shape=(n_samples,))

        # Library size (for gene expression normalization)
        library = torch.log(x.sum(1)).unsqueeze(1)

        outputs = {
            "z": z,
            "qz_m": qz_m,
            "qz_v": qz_v,
            "library": library,
            "gene_hidden": gene_hidden,
            "tcr_hidden": tcr_hidden,
        }
        return outputs

    @auto_move_data
    def generative(
        self,
        z,
        library,
    ):
        """Runs the generative model."""
        # Decode to gene expression
        gene_hidden = self.gene_decoder(z)

        if self.gene_likelihood in ["zinb", "nb"]:
            px_scale = self.px_scale_decoder(gene_hidden)
            px_rate = torch.exp(library) * px_scale
            # Use the rate decoder output as dispersion parameter
            px_dispersion = torch.exp(self.px_rate_decoder(gene_hidden))
            if self.gene_likelihood == "zinb":
                px_zi_logits = self.px_dropout_decoder(gene_hidden)
            else:
                px_zi_logits = None
        else:  # gaussian
            px_scale = self.px_scale_decoder(gene_hidden)
            px_rate = self.px_rate_decoder(gene_hidden)
            px_dispersion = None
            px_zi_logits = None

        # TCR reconstruction (for additional loss)
        tcr_logits = self.tcr_decoder(z)

        # Reshape to (batch_size, seq_len, vocab_size)
        batch_size = (
            z.shape[0] if z.dim() == 2 else z.shape[1]
        )  # Handle different z shapes
        vocab_size = self.tcr_tokenizer.vocab_size
        max_length = self.tcr_tokenizer.max_length

        # Debug: Check dimensions
        expected_size = batch_size * max_length * vocab_size
        actual_size = tcr_logits.numel()

        if expected_size != actual_size:
            raise RuntimeError(
                f"TCR decoder output size mismatch: "
                f"expected {expected_size} (batch_size={batch_size} * max_length={max_length} * vocab_size={vocab_size}), "
                f"but got {actual_size}"
            )

        tcr_logits = tcr_logits.view(batch_size, max_length, vocab_size)

        return {
            "px_scale": px_scale,
            "px_rate": px_rate,
            "px_dispersion": px_dispersion,
            "px_zi_logits": px_zi_logits,
            "tcr_logits": tcr_logits,
        }

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
        tcr_weight: float = 1.0,
    ):
        """Compute the loss."""
        x = tensors[REGISTRY_KEYS.X_KEY]
        tcr = tensors["TCR"]

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        px_rate = generative_outputs["px_rate"]
        px_dispersion = generative_outputs.get("px_dispersion")
        px_zi_logits = generative_outputs.get("px_zi_logits")
        tcr_logits = generative_outputs["tcr_logits"]

        # KL divergence
        kl_divergence_z = torch.distributions.kl.kl_divergence(
            Normal(qz_m, qz_v.sqrt()), Normal(0, 1)
        ).sum(dim=1)

        # Gene expression reconstruction loss
        if self.gene_likelihood == "zinb":
            reconst_loss_gene = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_rate, theta=px_dispersion, zi_logits=px_zi_logits
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        elif self.gene_likelihood == "nb":
            reconst_loss_gene = (
                -NegativeBinomial(mu=px_rate, theta=px_dispersion)
                .log_prob(x)
                .sum(dim=-1)
            )
        else:  # gaussian
            reconst_loss_gene = F.mse_loss(px_rate, x, reduction="none").sum(dim=-1)

        # TCR reconstruction loss (cross-entropy)
        # Convert string sequences to token IDs if necessary
        if isinstance(tcr, (list, tuple)) and len(tcr) > 0 and isinstance(tcr[0], str):
            tcr_tokens = self.tcr_tokenizer.encode_batch(tcr)
            if tcr_tokens.device != next(self.parameters()).device:
                tcr_tokens = tcr_tokens.to(next(self.parameters()).device)
        else:
            tcr_tokens = tcr

        # Ensure tcr_tokens is long type for embedding layer
        if tcr_tokens.dtype != torch.long:
            tcr_tokens = tcr_tokens.long()

        batch_size, seq_len = tcr_tokens.shape

        # Create attention mask (1 for real tokens, 0 for padding)
        tcr_attention_mask = self.tcr_tokenizer.create_attention_mask(tcr_tokens)

        # Compute cross-entropy loss
        # tcr_logits shape: (batch_size, seq_len, vocab_size)
        # tcr_tokens shape: (batch_size, seq_len)
        reconst_loss_tcr = F.cross_entropy(
            tcr_logits.transpose(1, 2),  # (batch_size, vocab_size, seq_len)
            tcr_tokens,
            reduction="none",
            ignore_index=self.tcr_tokenizer.pad_token_id,
        )

        # Apply attention mask and sum over sequence length
        reconst_loss_tcr = (reconst_loss_tcr * tcr_attention_mask.float()).sum(dim=-1)

        # Total loss
        weighted_kl = kl_weight * kl_divergence_z
        weighted_tcr_loss = tcr_weight * reconst_loss_tcr

        loss = reconst_loss_gene + weighted_tcr_loss + weighted_kl

        return LossOutput(
            loss=loss.mean(),  # Take mean for scalar loss expected by PyTorch Lightning
            reconstruction_loss=reconst_loss_gene,  # Use per-sample reconstruction loss (as in other scvi modules)
            kl_local=kl_divergence_z,  # Keep per-sample for monitoring
            extra_metrics={
                "gene_reconstruction_loss": reconst_loss_gene.mean(),
                "tcr_reconstruction_loss": reconst_loss_tcr.mean(),
                "total_reconstruction_loss": (
                    reconst_loss_gene + weighted_tcr_loss
                ).mean(),
            },
        )

    @auto_move_data
    def sample(self, tensors, n_samples=1):
        """Generate samples from the model."""
        inference_outputs = self.inference(
            tensors[REGISTRY_KEYS.X_KEY], tensors["TCR"], n_samples
        )
        generative_outputs = self.generative(
            inference_outputs["z"], inference_outputs["library"]
        )

        return generative_outputs

    @auto_move_data
    def marginal_ll(self, tensors, n_mc_samples):
        """Compute marginal log likelihood."""
        # This is a simplified implementation
        # In practice, you might want to use importance sampling
        sample_batch = self.sample(tensors, n_mc_samples)
        return sample_batch
