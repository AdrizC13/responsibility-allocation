import os
import pandas as pd
import torch

class ScenarioLoader:
    """Loads one scenario and returns a list of tensors (one per car)."""

    def load(self, scenario_path):
        car_dir = os.path.join(scenario_path, "data_per_car")
        car_files = sorted(os.listdir(car_dir))
        cars = []
        for file in car_files:
            final_path = os.path.join(car_dir, file)
            df = pd.read_csv(final_path)
            df = df.loc[:, ~df.columns.str.contains("rgb_path", case=False)]
            Xi = torch.tensor(df.values, dtype=torch.float32)
            cars.append(Xi)
        return cars
