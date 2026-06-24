# test_pipeline.py
import torch
from model_cv import PatternScanner
from data_engine_cv import generate_chart_patterns

def test_network_dimensions():
    """
    Verifies that the 1D CNN processes input shapes correctly and 
    outputs raw classification logits matching the 3 targeted structural classes.
    """
    model = PatternScanner()
    model.eval()
    
    # Generate 10 isolated evaluation candles
    inputs, targets = generate_chart_patterns(samples=10, seed=42)
    
    with torch.no_grad():
        outputs = model(inputs)
        
    # Check batch execution dimensions
    assert outputs.shape == (10, 3), f"Expected shape (10, 3), got {outputs.shape}"

def test_model_loss_computation():
    """
    Verifies that the loss computation loop returns stable, scalar values.
    """
    model = PatternScanner()
    criterion = torch.nn.CrossEntropyLoss()
    
    inputs, targets = generate_chart_patterns(samples=5, seed=42)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    assert loss.item() > 0.0, "Loss computation returned invalid zero or negative value."