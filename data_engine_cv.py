# data_engine_cv.py
import torch
import numpy as np

def generate_chart_patterns(samples=1000, seed=None):
    """
    Generates synthetic XAUUSD structural patterns for isolated out-of-sample validation.
    
    Classes:
    0: Market Noise
    1: Double Bottom (W Shape)
    2: Double Top (M Shape)
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Sequences of 50 closing prices
    # [Batch, Channels (1 for Close price), Sequence Length]
    X = np.zeros((samples, 1, 50)) 
    y = np.zeros(samples)
    
    for i in range(samples):
        pattern_type = np.random.choice([0, 1, 2])
        close_px = np.linspace(1900, 1950, 50) + np.random.normal(0, 2, 50)
        
        if pattern_type == 1:  # Inject a 'W' shape
            close_px[10:20] -= np.linspace(0, 20, 10)
            close_px[20:30] += np.linspace(0, 20, 10)
            close_px[30:40] -= np.linspace(0, 20, 10)
            close_px[40:50] += np.linspace(0, 20, 10)
            y[i] = 1
        elif pattern_type == 2:  # Inject an 'M' shape
            close_px[10:20] += np.linspace(0, 20, 10)
            close_px[20:30] -= np.linspace(0, 20, 10)
            close_px[30:40] += np.linspace(0, 20, 10)
            close_px[40:50] -= np.linspace(0, 20, 10)
            y[i] = 2
        else:
            y[i] = 0  # Market Noise baseline
            
        X[i, 0, :] = close_px
        
    return torch.FloatTensor(X), torch.LongTensor(y)