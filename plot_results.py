import json
import matplotlib.pyplot as plt
import numpy as np

with open("results/qaoa_results.json") as f:
    qaoa_data = json.load(f)

with open("results/classical_results.json") as f:
    classical_data = json.load(f)

sizes = [5, 8, 10, 12]

qaoa_makespans = [x["makespan"] for x in qaoa_data]
classical_makespans = [x["makespan"] for x in classical_data]

# Relative Gap
relative_gap = [
    100 * (qaoa - classical) / classical
    for qaoa, classical in zip(qaoa_makespans, classical_makespans)
]

# Plot 1 - Makespan comparison
plt.figure(figsize=(8,5))
plt.plot(sizes, qaoa_makespans, marker="o", label="QAOA")
plt.plot(sizes, classical_makespans, marker="o", label="Classical Optimum")
plt.title("Makespan vs Dataset Size")
plt.xlabel("Dataset size (# jobs)")
plt.ylabel("Makespan")
plt.legend()
plt.savefig("results/plots/makespan_comparison.png")
plt.close()

# Plot 2 - Relative Gap
plt.figure(figsize=(8,5))
plt.plot(sizes, relative_gap, marker="o", color="red")
plt.title("Relative Performance Gap (%)")
plt.xlabel("Dataset size (# jobs)")
plt.ylabel("Gap %")
plt.savefig("results/plots/relative_gap.png")
plt.close()

print("Plots saved.")
