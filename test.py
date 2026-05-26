import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def xray_cross_entropy():
    torch.set_printoptions(precision=4, sci_mode=False)
    
    # 1. The Setup
    # Imagine our batch size is 1. The bot looks at a chart and outputs these 3 raw scores.
    # [Noise, Double Bottom, Double Top]
    raw_logits = torch.tensor([[0.5, 2.5, -1.2]]) 
    
    # Let's say reality is that the chart is a Double Bottom (Index 1)
    target_class = torch.tensor([1]) 
    
    print("--- 1. THE RAW DATA ---")
    print(f"Bot's Raw Scores (Logits): {raw_logits.numpy()[0]}")
    print(f"Actual Truth: Class {target_class.item()} (Double Bottom)")
    
    # ---------------------------------------------------------
    # MANUAL STEP-BY-STEP MATH
    # ---------------------------------------------------------
    print("\n--- 2. MANUAL MATH BREAKDOWN ---")
    
    # Step A: Apply Euler's number (e^x) to all scores
    exp_logits = torch.exp(raw_logits)
    print(f"e^x of scores: {exp_logits.numpy()[0]}")
    
    # Step B: Sum them up for the denominator
    sum_exp = torch.sum(exp_logits)
    
    # Step C: Divide to get exact percentages (Softmax)
    probabilities = exp_logits / sum_exp
    print(f"Softmax Probabilities: {probabilities.numpy()[0] * 100}%")
    
    # Step D: Isolate the probability of the *correct* class
    correct_prob = probabilities[0][target_class.item()].item()
    print(f"Probability assigned to the CORRECT answer: {correct_prob:.4f} ({correct_prob * 100:.1f}%)")
    
    # Step E: Apply the Negative Natural Log (-ln)
    manual_loss = -math.log(correct_prob)
    print(f"Manual Cross-Entropy Loss: -ln({correct_prob:.4f}) = {manual_loss:.4f}")
    
    # ---------------------------------------------------------
    # PYTORCH AUTOMATIC MATH
    # ---------------------------------------------------------
    print("\n--- 3. PYTORCH IMPLEMENTATION ---")
    
    # PyTorch does all of the above in one line of code
    criterion = nn.CrossEntropyLoss()
    pytorch_loss = criterion(raw_logits, target_class)
    
    print(f"PyTorch Cross-Entropy Loss: {pytorch_loss.item():.4f}")
    
    if math.isclose(manual_loss, pytorch_loss.item(), rel_tol=1e-5):
        print("\n✅ SUCCESS: Manual math matches PyTorch perfectly.")

if __name__ == "__main__":
    xray_cross_entropy()