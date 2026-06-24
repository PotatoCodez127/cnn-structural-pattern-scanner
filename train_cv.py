# train_cv.py
import torch
import torch.nn as nn
import torch.optim as optim
from data_engine_cv import generate_chart_patterns
from model_cv import PatternScanner


def train_and_test():
    # Strict index isolation firewall: Train vs Test completely separated via seeds
    X_train, y_train = generate_chart_patterns(1500, seed=42)
    X_test, y_test = generate_chart_patterns(300, seed=100)

    model = PatternScanner()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.005)

    print("Training 1D CNN Structural Engine...")
    epochs = 50
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    print("\n--- Out-of-Sample Firewall Verification ---")
    classes = ["Noise / No Pattern", "Double Bottom (W)", "Double Top (M)"]

    model.eval()
    with torch.no_grad():
        raw_scores = model(X_test[:3])
        probabilities = torch.nn.functional.softmax(raw_scores, dim=1)

    for i in range(3):
        predicted_class = torch.argmax(probabilities[i]).item()
        confidence = probabilities[i][predicted_class].item() * 100

        print(f"Chart {i+1} Actual OOS: {classes[y_test[i].item()]}")
        print(f"Engine Output: {classes[predicted_class]} ({confidence:.1f}% confidence)\n")


if __name__ == "__main__":
    train_and_test()
