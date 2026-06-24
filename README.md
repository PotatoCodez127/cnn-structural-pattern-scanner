# XAUUSD Structural Pattern Scanner

An enterprise-grade 1D Convolutional Neural Network (1D-CNN) engineered in PyTorch to serve as a pre-trade structural filter for quantitative execution engines. The module processes sliding windows of closing prices to detect geometric consolidation structures—classifying Double Bottoms, Double Tops, and Market Noise to validate breakout signals.

## Role in the Omni-Agent Ecosystem
This module functions as an out-of-sample execution gateway. When prior filters signal volatility expansion, this scanner confirms whether the price is breaking out of a verified macroeconomic geometry (e.g., a Double Bottom neckline), filtering out low-probability false breakouts and optimizing execution accuracy.

## Architecture
- **Input Layer:** 1D sequence arrays (Batch, 1 Channel, 50 Timesteps) reflecting closing prices.
- **Feature Extraction:** 1D Convolutional layer (`kernel_size=5`, `out_channels=16`) capturing local temporal patterns.
- **Dimensionality Reduction:** Max Pooling (`kernel_size=2`) isolating principal signal indicators.
- **Classification Head:** Linear layers mapping flattened spatial features to raw target class logits.
- **Loss Function:** `CrossEntropyLoss` for clean probabilistic grading.
- **Optimization:** RMSprop (`lr=0.005`) tuned for trailing multi-class sequence profiles.

## Installation & Environment Setup
This repository uses `pyproject.toml` for deterministic dependency management.

```bash
# Clone the repository
git clone [https://github.com/your-portfolio/xau-structural-pattern-scanner.git](https://github.com/your-portfolio/xau-structural-pattern-scanner.git)
cd xau-structural-pattern-scanner

# Install package dependencies
pip install .
```

## Local Execution & Training
To run the deterministic out-of-sample training execution pipeline:
```bash
python train_cv.py
```

## Containerized Deployment
Execute the pipeline via isolated, multi-stage Docker builds:
```bash
docker build -t xau-pattern-scanner .
docker run --rm xau-pattern-scanner
```

## Role in the Omni-Agent Ecosystem
This acts as a structural confluence filter. If the Volatility Classifier (Project 1) signals expansion, this module confirms whether the expansion is breaking out of a recognized geometric structure (like the neckline of a Double Bottom), drastically increasing the win rate of the final trade thesis.