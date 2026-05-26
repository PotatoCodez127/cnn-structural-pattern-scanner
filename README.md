# XAUUSD Structural Pattern Scanner

## Overview
This project is a 1D Convolutional Neural Network (CNN) designed to act as the "eyes" of an algorithmic trading bot. Instead of looking at individual data points, it scans a 50-candle window of XAUUSD closing prices to detect macroscopic geometrical shapes, successfully classifying Double Bottoms, Double Tops, and Market Noise.

## Architecture
* **Inputs:** 1D sequence of length 50 (Batch, 1 Channel, 50 Timesteps).
* **Network:** 1D Convolutional Neural Network.
    * `Conv1d` (Kernel Size: 5, out_channels: 16) for spatial feature extraction.
    * `MaxPool1d` (Kernel Size: 2) for dimensionality reduction and signal isolation.
* **Activations:** `ReLU` for feature maps, `Softmax` (via PyTorch CrossEntropy) for the 3-class output.
* **Loss Function:** `CrossEntropyLoss` (Multi-class probabilistic loss).
* **Optimizer:** RMSprop (`lr=0.005`), optimized for sliding window gradients.

## Role in the Omni-Agent Ecosystem
This acts as a structural confluence filter. If the Volatility Classifier (Project 1) signals expansion, this module confirms whether the expansion is breaking out of a recognized geometric structure (like the neckline of a Double Bottom), drastically increasing the win rate of the final trade thesis.