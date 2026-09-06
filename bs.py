import os
import matplotlib.pyplot as plt

# Make graphs directory if it doesn't exist
os.makedirs("graphs", exist_ok=True)

# Data
labels = ["Unshifted\n(control)", "Shifted ~1 cm\n(no correction)"]
accuracies = [100.0, 66.7]

# Plot
plt.figure(figsize=(7, 5))
bars = plt.bar(labels, accuracies)

# Labels and title
plt.ylabel("Trial-level accuracy (%)")
plt.title("Effect of electrode shift on EMG gesture classification")

# Y-axis formatting
plt.ylim(0, 110)
plt.grid(axis="y", alpha=0.3)

# Annotate bars
for bar, acc in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        acc + 2,
        f"{acc:.1f}%",
        ha="center",
        va="bottom",
        fontsize=11,
    )

# Save
plt.tight_layout()
plt.savefig("graphs/electrode_shift_baseline_accuracy.png", dpi=300)
plt.show()
