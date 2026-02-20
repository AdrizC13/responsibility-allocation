import os
import torch # type: ignore
from utils import ScenarioLoader, AccidentModel

loader = ScenarioLoader()
test_dir = "data\\test"
scenarios = sorted(os.listdir(test_dir))
cars0 = loader.load(os.path.join(test_dir, scenarios[0]))
P = cars0[0].shape[1]
C = len(cars0)
model = AccidentModel(P, 512, 128, C)
model.load_state_dict(torch.load("model.pt", weights_only=True))
model.eval()

correct = 0
for sc in scenarios:
    path = os.path.join(test_dir, sc)
    cars = loader.load(path)
    gt = int(open(os.path.join(path, "label.txt")).read()) - 1

    with torch.no_grad():
        pred = torch.argmax(model(cars)).item()

    print(sc, "Pred:", pred, "Actual:", gt)
    if pred == gt:
        correct += 1

print("Accuracy:", correct / len(scenarios))

