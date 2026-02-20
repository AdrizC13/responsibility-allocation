import os
import torch # type: ignore
import torch.nn as nn # type: ignore
from utils import ScenarioLoader, AccidentModel

# Configuration
train_dir = "data\\train"
loader = ScenarioLoader()
scenarios = sorted(os.listdir(train_dir))
accumulation_steps = 4  # Effectively batch size of 4

# Model Initialization
cars0 = loader.load(os.path.join(train_dir, scenarios[0]))
P = cars0[0].shape[1]
C = len(cars0)
model = AccidentModel(P, embed_dim=512, attn_dim=128, num_cars=C)

# Switching to AdamW often helps when SGD fails to converge on complex temporal data
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    model.train()
    total_loss = 0
    optimizer.zero_grad() # Reset outside the scenario loop
    
    for i, sc in enumerate(scenarios):
        path = os.path.join(train_dir, sc)
        cars = loader.load(path)
        label = int(open(os.path.join(path, "label.txt")).read()) - 1

        # Forward Pass
        logits = model(cars) 
        probs = torch.softmax(logits, dim=0)
        pred = torch.argmax(probs).item()
        print(f"prediction of scenario {sc}: {pred}, Label: {label}")
        # Loss calculation: Note the .unsqueeze(0) to handle batch size 1
        loss = criterion(logits.unsqueeze(0), torch.tensor([label]))
        
        # Normalize loss for accumulation
        loss = loss / accumulation_steps
        loss.backward()

        # Update weights every X steps
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(scenarios):
            # Gradient clipping prevents the "Exploding Gradient" problem in LSTMs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    avg_loss = total_loss / len(scenarios)
    print(f"Epoch {epoch + 1:03d} | Total Loss = {total_loss:.4f} | Avg Loss = {avg_loss:.4f}")

torch.save(model.state_dict(), "model.pt")
print("Model saved")

