# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-07-30

### Added
- Initial release of IRVI (Immune Repertoire Variation Inference)
- Multimodal Transformer-based Variational Autoencoder for gene expression and TCR data
- Support for amino acid sequence tokenization and transformer encoding
- Integration with scvi-tools framework
- Multiple gene expression likelihoods (ZINB, NB, Gaussian)
- Comprehensive test suite
- Documentation and examples

### Features
- `IRVI` main model class for multimodal analysis
- `IRVAE` module implementing the variational autoencoder
- `AminoAcidTokenizer` for TCR sequence preprocessing
- `TCREncoder` transformer-based sequence encoder
- `TCRField` custom data field for scvi-tools integration

### Dependencies
- scvi-tools >= 1.0.0
- torch >= 1.12.0
- numpy >= 1.21.0
- pandas >= 1.3.0
- anndata >= 0.8.0
- scanpy >= 1.8.0
