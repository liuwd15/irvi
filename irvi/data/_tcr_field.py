"""Custom field for handling TCR amino acid sequence strings."""

import numpy as np
import pandas as pd
from anndata import AnnData
from scvi.data.fields._base_field import BaseAnnDataField


class TCRField(BaseAnnDataField):
    """
    Field for handling TCR amino acid sequence strings.

    This field automatically tokenizes amino acid sequence strings into
    integer representations suitable for transformer models.

    Parameters
    ----------
    registry_key
        Key to register field under in data registry.
    attr_key
        Key to access the field in the AnnData obs/obsm mapping.
    max_length
        Maximum sequence length for tokenization.
    add_special_tokens
        Whether to add special tokens (START, END, etc.).
    """

    def __init__(
        self,
        registry_key: str,
        attr_key: str,
        max_length: int = 20,
        add_special_tokens: bool = True,
    ):
        super().__init__()
        self._registry_key = registry_key
        self._original_attr_key = attr_key  # Store the original key
        self._attr_key = f"_scvi_{registry_key}"  # Processed data key
        self._attr_name = "obsm"  # We'll store processed data in obsm
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens

        # Initialize tokenizer (import here to avoid circular imports)
        from ..module import AminoAcidTokenizer

        self.tokenizer = AminoAcidTokenizer(
            max_length=max_length, add_special_tokens=add_special_tokens
        )

    @property
    def registry_key(self) -> str:
        """The key that is referenced by models via a data loader."""
        return self._registry_key

    @property
    def attr_name(self) -> str:
        """The name of the AnnData attribute where the data is stored."""
        return self._attr_name

    @property
    def attr_key(self) -> str:
        """The key of the data field within the relevant AnnData attribute."""
        return f"_scvi_{self._registry_key}"

    @property
    def is_empty(self) -> bool:
        """Returns True if the field is empty as a function of its kwargs."""
        return False  # TCR field is never empty once initialized

    def validate_field(self, adata: AnnData) -> None:
        """Validate that the field exists and contains valid sequence data."""
        # Check if field exists in obs or obsm using the original key
        if self._original_attr_key in adata.obs.columns:
            data = adata.obs[self._original_attr_key]
        elif self._original_attr_key in adata.obsm:
            data = adata.obsm[self._original_attr_key]
        else:
            raise ValueError(
                f"TCR data not found in adata.obs['{self._original_attr_key}'] or adata.obsm['{self._original_attr_key}']"
            )

        # Validate sequences
        if isinstance(data, pd.Series):
            sequences = data.tolist()
        elif isinstance(data, np.ndarray):
            if data.dtype.kind in ["U", "S", "O"]:  # String arrays
                sequences = data.tolist()
            else:
                # Assume already tokenized
                return
        else:
            sequences = list(data)

        # Check that sequences are strings and contain valid amino acids
        for i, seq in enumerate(sequences):
            if pd.isna(seq) or seq == "":
                continue
            if not isinstance(seq, str):
                continue

            # Check for invalid amino acids (allow separator token '_')
            valid_chars = set(self.tokenizer.AMINO_ACIDS) | {
                "_"
            }  # Include separator token
            invalid_chars = set(seq.upper()) - valid_chars
            if invalid_chars:
                print(
                    f"Warning: Sequence {i} contains invalid amino acids: {invalid_chars}"
                )

    def register_field(self, adata: AnnData) -> dict:
        """Register and process the TCR sequence field."""
        self.validate_field(adata)

        # Get the sequence data using the original key
        if self._original_attr_key in adata.obs.columns:
            sequences = adata.obs[self._original_attr_key].tolist()
        elif self._original_attr_key in adata.obsm:
            data = adata.obsm[self._original_attr_key]
            if isinstance(data, np.ndarray) and data.dtype.kind not in ["U", "S", "O"]:
                # Already tokenized
                sequences = data
            else:
                sequences = data.tolist() if hasattr(data, "tolist") else list(data)
        else:
            raise ValueError(
                f"TCR data not found in adata.obs['{self._original_attr_key}'] or adata.obsm['{self._original_attr_key}']"
            )

        # Tokenize if sequences are strings
        if (
            isinstance(sequences, list)
            and len(sequences) > 0
            and isinstance(sequences[0], str)
        ):
            tokenized_sequences = []
            for seq in sequences:
                if pd.isna(seq) or seq == "":
                    # Handle missing sequences with all padding
                    tokenized_seq = np.zeros(self.max_length, dtype=np.int64)
                else:
                    tokenized_seq = (
                        self.tokenizer.encode(str(seq)).numpy().astype(np.int64)
                    )
                tokenized_sequences.append(tokenized_seq)

            processed_data = np.array(tokenized_sequences, dtype=np.int64)
        else:
            # Assume already processed, but ensure it's int64
            processed_data = np.array(sequences, dtype=np.int64)

        # Store processed data in obsm using the processed key
        adata.obsm[self.attr_key] = processed_data

        return {
            "attr_name": self.attr_name,
            "attr_key": self.attr_key,
            "shape": processed_data.shape,
            "tokenizer_vocab_size": self.tokenizer.vocab_size,
            "tokenizer_max_length": self.tokenizer.max_length,
        }

    def transfer_field(
        self,
        state_registry: dict,
        adata_target: AnnData,
        **kwargs,
    ) -> dict:
        """Transfer field to new AnnData object."""
        # For transfer, we need to tokenize the target data using the same tokenizer
        return self.register_field(adata_target)

    def get_summary_stats(self, state_registry: dict) -> dict:
        """Get summary statistics for the field."""
        return {
            "n_sequences": state_registry["shape"][0],
            "max_length": state_registry["tokenizer_max_length"],
            "vocab_size": state_registry["tokenizer_vocab_size"],
        }

    def view_state_registry(self, state_registry: dict):
        """View state registry in a formatted table."""
        try:
            from rich.table import Table

            t = Table(title=f"TCR Sequence Field Registry: {self.registry_key}")
            t.add_column("Property", justify="left")
            t.add_column("Value", justify="left")

            t.add_row("Data type", "TCR amino acid sequences")
            t.add_row("Number of sequences", str(state_registry["shape"][0]))
            t.add_row("Sequence length", str(state_registry["shape"][1]))
            t.add_row("Vocabulary size", str(state_registry["tokenizer_vocab_size"]))
            t.add_row("Max length", str(state_registry["tokenizer_max_length"]))
            t.add_row("Location", f"adata.obsm['{state_registry['attr_key']}']")

            return t
        except ImportError:
            # Fallback if rich is not available
            return None
