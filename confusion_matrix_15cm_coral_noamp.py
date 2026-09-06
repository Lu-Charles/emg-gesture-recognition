# Confusion matrix plot for 1.5 cm SHIFT (NO-AMP) + CORAL
# Matches formatting/style of previous calibrated-CORAL plot

import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure output directory exists
out_dir = "graphs"
os.makedirs(out_dir, exist_ok=True)

# Confusion matrix data (1.5 cm shift, NO-AMP, WITH CORAL)
# Rows = true labels [extend, fist, rest]
# Cols = predicted labels [extend, fist, rest]
cm = np.array([
    [7, 0, 10],   # true extend
    [0, 17, 0],   # true fist
    [3, 0, 14],   # true rest
], dtype=int)

labels = ["extend", "fist", "rest"]

# Normalize by row for percentages
cm_norm = cm / cm.sum(axis=1, keepdims=True)

# Plot
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Greens", vmin=0, vmax=int(cm.max()))

# Ticks and labels
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("Gesture classification confusion matrix\n1.5 cm electrode shift + CORAL (no amplitude features)")

# Annotate cells with count + percentage
mx = cm.max()
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        pct = cm_norm[i, j] * 100
        ax.text(
            j, i,
            f"{count}\n{pct:.1f}%",
            ha="center", va="center",
            color="white" if count >= mx * 0.5 else "black",
            fontsize=11
        )

# Colorbar (fixed ticks)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Number of trials")
cbar.set_ticks(list(range(0, mx + 1, 2)))

plt.tight_layout()
out_path = os.path.join(out_dir, "confusion_matrix_15cm_coral_noamp.png")
plt.savefig(out_path, dpi=300)
plt.close()

print(f"Saved: {out_path}")