from utils.dataset_generator import generate_dataset
import os

BASE = os.path.dirname(os.path.abspath(__file__))

datasets_folder = os.path.join(BASE, "datasets")

# Create datasets with absolute paths
generate_dataset(5, os.path.join(datasets_folder, "dataset_5.csv"))
generate_dataset(8, os.path.join(datasets_folder, "dataset_8.csv"))
generate_dataset(10, os.path.join(datasets_folder, "dataset_10.csv"))
generate_dataset(12, os.path.join(datasets_folder, "dataset_12.csv"))

print("Datasets generated successfully!")
