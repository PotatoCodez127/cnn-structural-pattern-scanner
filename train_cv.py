# train_cv.py
import torch
import torch.nn as nn
import torch.optim as optim
from data_engine_cv import generate_chart_patterns
from model_cv import PatternScanner

def train_and_test():
    X, y = generate_chart_patterns(1500)
    model = PatternScanner()
    
    # CrossEntropyLoss automatically applies Softmax math
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.RMSprop(model.parameters(), lr=0.005)
    
    print("Training CNN...")
    epochs = 50
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    print("\n--- Live Vision Test ---")
    X_new, y_real = generate_chart_patterns(3)
    classes = ["Noise / No Pattern", "Double Bottom (W)", "Double Top (M)"]
    
    model.eval()
    with torch.no_grad():
        raw_scores = model(X_new)
        # Apply Softmax manually just for viewing the exact percentages
        probabilities = torch.nn.functional.softmax(raw_scores, dim=1) 
        
    for i in range(3):
        # Find the index of the highest probability
        predicted_class = torch.argmax(probabilities[i]).item()
        confidence = probabilities[i][predicted_class].item() * 100
        
        print(f"Chart {i+1} Actual: {classes[y_real[i].item()]}")
        print(f"Bot Sees: {classes[predicted_class]} ({confidence:.1f}% confidence)\n")

if __name__ == "__main__":
    train_and_test()