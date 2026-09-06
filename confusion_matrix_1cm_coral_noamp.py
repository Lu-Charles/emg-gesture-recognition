import os
import numpy as np
import matplotlib.pyplot as plt

out_dir = "graphs"
os.makedirs(out_dir, exist_ok=True)

cm = np.array([
    [17, 0, 0],
    [0, 17, 0],
    [0, 0, 17],
], dtype=int)

labels = ["extend", "fist", "rest"]
cm_norm = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Oranges", vmin=0, vmax=int(cm.max()))

ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("Gesture classification confusion matrix\n1.0 cm shift + CORAL (no amplitude features)")

mx = cm.max()
for i in range(3):
    for j in range(3):
        count = cm[i, j]
        pct = cm_norm[i, j] * 100
        ax.text(
            j, i,
            f"{count}\n{pct:.1f}%",
            ha="center",
            va="center",
            color="white" if count >= mx * 0.5 else "black",
            fontsize=11
        )

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Number of trials")
cbar.set_ticks(list(range(0, mx + 1, 2)))

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "confusion_matrix_1cm_coral_noamp.png"), dpi=300)
plt.close()

print("Saved 1cm coral noamp heatmap.")