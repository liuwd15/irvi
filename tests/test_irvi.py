"""Tests for IRVI package."""

from unittest.mock import patch

import anndata as ad
import numpy as np
import pytest
import torch

# Import your modules
from irvi import IRVI
from irvi.data import TCRField
from irvi.module import IRVAE, AminoAcidTokenizer, TCREncoder


class TestAminoAcidTokenizer:
    """Test the AminoAcidTokenizer class."""

    def test_tokenizer_initialization(self):
        """Test tokenizer initialization with default settings."""
        tokenizer = AminoAcidTokenizer(max_length=20, add_special_tokens=True)

        assert tokenizer.max_length == 20
        assert tokenizer.add_special_tokens is True
        assert tokenizer.vocab_size == 24  # 4 special + 20 amino acids
        assert tokenizer.pad_token_id == 0
        assert tokenizer.unk_token_id == 1

    def test_tokenizer_without_special_tokens(self):
        """Test tokenizer initialization without special tokens."""
        tokenizer = AminoAcidTokenizer(max_length=15, add_special_tokens=False)

        assert tokenizer.vocab_size == 21  # 1 PAD + 20 amino acids
        assert tokenizer.pad_token_id == 0
        assert tokenizer.unk_token_id == 0

    def test_encode_sequence(self):
        """Test encoding amino acid sequences."""
        tokenizer = AminoAcidTokenizer(max_length=10, add_special_tokens=True)

        # Test normal sequence
        sequence = "CASSLAP"
        encoded = tokenizer.encode(sequence)

        assert encoded.shape == (10,)
        assert encoded.dtype == torch.long
        assert encoded[0] == tokenizer.vocab["<START>"]
        assert encoded[-1] == tokenizer.vocab["<PAD>"]  # Should be padded

    def test_encode_batch(self):
        """Test batch encoding."""
        tokenizer = AminoAcidTokenizer(max_length=10, add_special_tokens=True)

        sequences = ["CASSLAP", "GTQVQE", "ARNDCY"]
        encoded = tokenizer.encode_batch(sequences)

        assert encoded.shape == (3, 10)
        assert encoded.dtype == torch.long

    def test_decode_sequence(self):
        """Test decoding token IDs back to sequences."""
        tokenizer = AminoAcidTokenizer(max_length=10, add_special_tokens=True)

        sequence = "CASSLAP"
        encoded = tokenizer.encode(sequence)
        decoded = tokenizer.decode(encoded)

        assert decoded == sequence

    def test_attention_mask(self):
        """Test attention mask creation."""
        tokenizer = AminoAcidTokenizer(max_length=8, add_special_tokens=True)

        sequence = "CASSL"  # Short sequence
        encoded = tokenizer.encode(sequence)
        mask = tokenizer.create_attention_mask(encoded)

        assert mask.shape == encoded.shape
        assert mask.dtype == torch.long
        assert mask.sum() < len(mask)  # Should have some padding positions (0s)

    def test_unknown_amino_acids(self):
        """Test handling of unknown amino acids."""
        tokenizer = AminoAcidTokenizer(max_length=10, add_special_tokens=True)

        sequence = "CASXLAP"  # X is not a standard amino acid
        encoded = tokenizer.encode(sequence)

        # Should contain UNK token
        assert tokenizer.unk_token_id in encoded


class TestTCREncoder:
    """Test the TCREncoder class."""

    @pytest.fixture
    def tcr_encoder(self):
        """Create a TCR encoder for testing."""
        return TCREncoder(max_length=10, d_model=64, nhead=4, num_layers=2, dropout=0.1)

    def test_tcr_encoder_initialization(self, tcr_encoder):
        """Test TCR encoder initialization."""
        assert tcr_encoder.d_model == 64
        assert tcr_encoder.max_length == 10
        assert tcr_encoder.tokenizer.vocab_size == 24

    def test_encode_string_sequences(self, tcr_encoder):
        """Test encoding string sequences."""
        sequences = ["CASSLAP", "GTQVQE"]
        encoded = tcr_encoder.encode_sequences(sequences)

        assert encoded.shape == (2, 10)
        assert encoded.dtype == torch.long

    def test_forward_with_strings(self, tcr_encoder):
        """Test forward pass with string inputs."""
        sequences = ["CASSLAP", "GTQVQE", "ARNDCY"]

        with torch.no_grad():
            output = tcr_encoder(sequences)

        assert output.shape == (3, 64)  # batch_size, d_model
        assert not torch.isnan(output).any()

    def test_forward_with_token_ids(self, tcr_encoder):
        """Test forward pass with pre-tokenized inputs."""
        sequences = ["CASSLAP", "GTQVQE"]
        token_ids = tcr_encoder.encode_sequences(sequences)

        with torch.no_grad():
            output = tcr_encoder(token_ids)

        assert output.shape == (2, 64)
        assert not torch.isnan(output).any()

    def test_attention_mask_handling(self, tcr_encoder):
        """Test custom attention mask handling."""
        sequences = ["CASSL", "GTQVQEAFF"]  # Different lengths
        token_ids = tcr_encoder.encode_sequences(sequences)
        attention_mask = tcr_encoder.tokenizer.create_attention_mask(token_ids)

        with torch.no_grad():
            output = tcr_encoder(token_ids, attention_mask=attention_mask)

        assert output.shape == (2, 64)
        assert not torch.isnan(output).any()


class TestIRVAE:
    """Test the IRVAE module."""

    @pytest.fixture
    def vae_module(self):
        """Create a VAE module for testing."""
        return IRVAE(
            n_input_genes=100,
            n_latent=10,
            n_hidden=64,
            n_layers=2,
            tcr_d_model=32,
            tcr_nhead=4,
            tcr_num_layers=2,
            dropout_rate=0.1,
            gene_likelihood="zinb",
        )

    def test_vae_initialization(self, vae_module):
        """Test VAE module initialization."""
        assert vae_module.n_input_genes == 100
        assert vae_module.n_latent == 10
        assert vae_module.n_hidden == 64
        assert vae_module.gene_likelihood == "zinb"

    def test_inference_method(self, vae_module):
        """Test the inference method."""
        batch_size = 8
        gene_data = torch.randn(batch_size, 100)
        tcr_sequences = ["CASSLAP"] * batch_size

        with torch.no_grad():
            outputs = vae_module.inference(gene_data, tcr_sequences)

        assert "z" in outputs
        assert "qz_m" in outputs
        assert "qz_v" in outputs
        assert "library" in outputs
        assert "gene_hidden" in outputs
        assert "tcr_hidden" in outputs

        assert outputs["qz_m"].shape == (batch_size, 10)
        assert outputs["qz_v"].shape == (batch_size, 10)

    def test_generative_method(self, vae_module):
        """Test the generative method."""
        batch_size = 8
        z = torch.randn(batch_size, 10)
        library = torch.log(torch.ones(batch_size, 1) * 1000)

        with torch.no_grad():
            outputs = vae_module.generative(z, library)

        assert "px_rate" in outputs
        assert "px_dispersion" in outputs
        assert "px_zi_logits" in outputs
        assert "tcr_logits" in outputs

        assert outputs["px_rate"].shape == (batch_size, 100)
        assert outputs["tcr_logits"].shape == (
            batch_size,
            20,
            24,
        )  # seq_len, vocab_size

    def test_loss_computation(self, vae_module):
        """Test loss computation."""
        batch_size = 8

        # Create mock tensors
        mock_tensors = {
            "X": torch.randint(0, 100, (batch_size, 100)).float(),
            "TCR": ["CASSLAP"] * batch_size,
        }

        # Create inference outputs
        gene_data = mock_tensors["X"]
        tcr_sequences = mock_tensors["TCR"]

        with torch.no_grad():
            inference_outputs = vae_module.inference(gene_data, tcr_sequences)
            generative_outputs = vae_module.generative(
                (
                    inference_outputs["z"][0]
                    if inference_outputs["z"].dim() == 3
                    else inference_outputs["z"]
                ),
                inference_outputs["library"],
            )

            loss_output = vae_module.loss(
                mock_tensors, inference_outputs, generative_outputs
            )

        assert hasattr(loss_output, "loss")
        assert hasattr(loss_output, "reconstruction_loss")
        assert hasattr(loss_output, "kl_local")
        assert torch.isscalar(loss_output.loss) or loss_output.loss.dim() == 0


class TestIRVI:
    """Test the main IRVI model class."""

    @pytest.fixture
    def sample_adata(self):
        """Create sample AnnData for testing."""
        n_obs, n_vars = 100, 50

        # Gene expression data
        X = np.random.negative_binomial(10, 0.3, size=(n_obs, n_vars))

        # TCR sequences
        tcr_sequences = [
            "CASSLAPGTQVQETQY",
            "CASSLVGQNTEAFF",
            "CASRPGQGATEAFF",
            "CSVGDTQYF",
            "CASSPGQGATEAFF",
        ]
        tcr_data = np.random.choice(tcr_sequences, size=n_obs)

        # Create AnnData
        adata = ad.AnnData(X=X.astype(np.float32))
        adata.obs["TCR"] = tcr_data
        adata.var_names = [f"Gene_{i}" for i in range(n_vars)]
        adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]

        return adata

    def test_setup_anndata(self, sample_adata):
        """Test AnnData setup."""
        try:
            IRVI.setup_anndata(sample_adata, tcr_key="TCR")
            assert True  # If no exception, setup worked
        except Exception as e:
            pytest.fail(f"setup_anndata failed: {e}")

    def test_model_initialization(self, sample_adata):
        """Test model initialization."""
        IRVI.setup_anndata(sample_adata, tcr_key="TCR")

        model = IRVI(sample_adata, n_latent=5, n_hidden=32, tcr_d_model=16)

        assert model.n_latent == 5
        assert model.n_hidden == 32
        assert model.module is not None

    def test_get_latent_representation(self, sample_adata):
        """Test getting latent representations."""
        IRVI.setup_anndata(sample_adata, tcr_key="TCR")

        model = IRVI(sample_adata, n_latent=8, n_hidden=32, tcr_d_model=16)

        # Test with mean
        latent_mean = model.get_latent_representation(give_mean=True)
        assert latent_mean.shape == (100, 8)
        assert not np.isnan(latent_mean).any()

        # Test with sampling
        latent_sample = model.get_latent_representation(give_mean=False)
        assert latent_sample.shape == (100, 8)
        assert not np.isnan(latent_sample).any()

    def test_get_gene_expression_reconstruction(self, sample_adata):
        """Test gene expression reconstruction."""
        IRVI.setup_anndata(sample_adata, tcr_key="TCR")

        model = IRVI(sample_adata, n_latent=5, n_hidden=32, tcr_d_model=16)

        reconstruction = model.get_gene_expression_reconstruction()
        assert reconstruction.shape == (100, 50)
        assert not np.isnan(reconstruction).any()
        assert (reconstruction >= 0).all()  # Should be positive (rates)

    def test_get_modality_embeddings(self, sample_adata):
        """Test getting separate modality embeddings."""
        IRVI.setup_anndata(sample_adata, tcr_key="TCR")

        model = IRVI(sample_adata, n_latent=5, n_hidden=32, tcr_d_model=16)

        embeddings = model.get_modality_embeddings()

        assert "gene" in embeddings
        assert "tcr" in embeddings
        assert embeddings["gene"].shape == (100, 32)  # n_hidden
        assert embeddings["tcr"].shape == (100, 16)  # tcr_d_model
        assert not np.isnan(embeddings["gene"]).any()
        assert not np.isnan(embeddings["tcr"]).any()

    @patch("irvi.model.IRVI.train")
    def test_training_integration(self, mock_train, sample_adata):
        """Test that training can be called (mocked to avoid long execution)."""
        IRVI.setup_anndata(sample_adata, tcr_key="TCR")

        model = IRVI(sample_adata, n_latent=5, n_hidden=32, tcr_d_model=16)

        # Mock the train method to avoid actual training
        mock_train.return_value = None

        model.train(max_epochs=1)
        mock_train.assert_called_once()


class TestTCRField:
    """Test the TCRField data field."""

    def test_tcr_field_initialization(self):
        """Test TCR field initialization."""
        field = TCRField("TCR", "tcr_key", max_length=20)

        assert field.attr_name == "TCR"
        assert field.attr_key == "tcr_key"
        assert field.max_length == 20


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_sequences(self):
        """Test handling of empty sequences."""
        tokenizer = AminoAcidTokenizer(max_length=10)

        # Empty sequence should be handled gracefully
        encoded = tokenizer.encode("")
        assert encoded.shape == (10,)

    def test_very_long_sequences(self):
        """Test handling of very long sequences."""
        tokenizer = AminoAcidTokenizer(max_length=10)

        # Long sequence should be truncated
        long_sequence = "A" * 50
        encoded = tokenizer.encode(long_sequence)
        assert encoded.shape == (10,)

    def test_invalid_tcr_key(self):
        """Test error handling for invalid TCR key."""
        adata = ad.AnnData(X=np.random.randn(10, 5))

        with pytest.raises(ValueError, match="TCR data not found"):
            IRVI.setup_anndata(adata, tcr_key="nonexistent_key")

    def test_different_gene_likelihoods(self):
        """Test different gene likelihood options."""
        for likelihood in ["zinb", "nb", "gaussian"]:
            module = IRVAE(n_input_genes=50, n_latent=5, gene_likelihood=likelihood)
            assert module.gene_likelihood == likelihood


# Integration test
class TestEndToEndIntegration:
    """Test end-to-end functionality."""

    def test_full_pipeline(self):
        """Test the complete pipeline from data setup to analysis."""
        # Create synthetic data
        n_obs, n_vars = 50, 30
        X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars))
        tcr_seqs = ["CASSLAPGTQ", "CASSLVGQNT", "CASRPGQGAT"] * (n_obs // 3)
        if n_obs % 3:
            tcr_seqs.extend(["CSVGDTQYF"] * (n_obs % 3))

        adata = ad.AnnData(X=X.astype(np.float32))
        adata.obs["TCR_seq"] = tcr_seqs[:n_obs]

        # Setup and create model
        IRVI.setup_anndata(adata, tcr_key="TCR_seq")
        model = IRVI(
            adata, n_latent=3, n_hidden=16, tcr_d_model=8, tcr_nhead=2, tcr_num_layers=1
        )

        # Test all analysis functions
        latent = model.get_latent_representation()
        reconstruction = model.get_gene_expression_reconstruction()
        embeddings = model.get_modality_embeddings()

        # Verify outputs
        assert latent.shape == (n_obs, 3)
        assert reconstruction.shape == (n_obs, n_vars)
        assert embeddings["gene"].shape == (n_obs, 16)
        assert embeddings["tcr"].shape == (n_obs, 8)

        # All outputs should be finite
        assert np.isfinite(latent).all()
        assert np.isfinite(reconstruction).all()
        assert np.isfinite(embeddings["gene"]).all()
        assert np.isfinite(embeddings["tcr"]).all()


if __name__ == "__main__":
    pytest.main([__file__])
