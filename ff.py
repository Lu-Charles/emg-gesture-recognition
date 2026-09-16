# Create a confusion matrix plot for the SHIFTED baseline model
# Saved to EMG/graphs with a non-blue color scheme

import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure output directory exists
out_dir = "graphs"
os.makedirs(out_dir, exist_ok=True)

# Confusion matrix data (shifted baseline)
# Rows = true labels [extend, fist, rest]
# Cols = predicted labels [extend, fist, rest]
cm = np.array([
    [7, 0, 0],   # true extend
    [0, 7, 0],   # true fist
    [0, 0, 7],   # true rest
])

labels = ["extend", "fist", "rest"]

# Normalize by row for percentages
cm_norm = cm / cm.sum(axis=1, keepdims=True)

# Plot
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Greens", vmin=0, vmax=7)  # non-blue color scheme

# Ticks and labels
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("Gesture classification confusion matrix\nUnshifted electrodes (control)")

# Annotate cells with count + percentage
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        pct = cm_norm[i, j] * 100
        ax.text(
            j, i,
            f"{count}\n{pct:.1f}%",
            ha="center", va="center",
            color="black" if count < cm.max()/2 else "white",
            fontsize=11
        )

# Colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Number of trials")

plt.tight_layout()
out_path = os.path.join(out_dir, "confusion_matrix_shifted_baseline.png")
plt.savefig(out_path, dpi=300)
plt.close()

out_path
