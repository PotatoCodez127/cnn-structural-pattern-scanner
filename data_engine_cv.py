# data_engine_cv.py
import torch
import numpy as np

def generate_chart_patterns(samples=1000):
    # Sequences of 50 closing prices
    X = np.zeros((samples, 1, 50)) # [Batch, Channels (1 for Close price), Sequence Length]
    y = np.zeros(samples)
    
    for i in range(samples):
        pattern_type = np.random.choice([0, 1, 2]) # 0: Noise, 1: Double Bottom, 2: Double Top
        base_line = np.linspace(1900, 1950, 50) + np.random.normal(0, 2, 50)
        
        if pattern_type == 1: # Inject a 'W' shape
            base_line[10:20] -= np.linspace(0, 20, 10)
            base_line[20:30] += np.linspace(0, 20, 10)
            base_line[30:40] -= np.linspace(0, 20, 10)
            base_line[40:50] += np.linspace(0, 20, 10)
            y[i] = 1
        elif pattern_type == 2: # Inject an 'M' shape
            base_line[10:20] += np.linspace(0, 20, 10)
            base_line[20:30] -= np.linspace(0, 20, 10)
            base_line[30:40] += np.linspace(0, 20, 10)
            base_line[40:50] -= np.linspace(0, 20, 10)
            y[i] = 2
        else:
            y[i] = 0 # Just noisy baseline
            
        X[i, 0, :] = base_line
        
    return torch.FloatTensor(X), torch.LongTensor(y)