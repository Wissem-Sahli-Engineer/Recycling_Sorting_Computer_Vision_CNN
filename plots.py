# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

CSV = Path("runs/classify/train-3/results.csv")

df = pd.read_csv(CSV)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

if "train/loss" in df.columns:
    ax1.plot(
        df["epoch"],
        df["train/loss"],
        label="Train Loss",
        color="#007AFF",
        linewidth=2,
    )
if "val/loss" in df.columns:
    ax1.plot(
        df["epoch"],
        df["val/loss"],
        label="Val Loss",
        color="#FF3B30",
        linewidth=2,
    )

ax1.set_title("Training vs Validation Loss", fontsize=12, fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend()

# Plot 2: Top-1 and Top-5 Classification Accuracy
if "metrics/accuracy_top1" in df.columns:
    ax2.plot(
        df["epoch"],
        df["metrics/accuracy_top1"],
        label="Top-1 Accuracy",
        color="#34C759",
        linewidth=2,
    )
if "metrics/accuracy_top5" in df.columns:
    ax2.plot(
        df["epoch"],
        df["metrics/accuracy_top5"],
        label="Top-5 Accuracy",
        color="#AF52DE",
        linewidth=2,
    )

ax2.set_title(
    "Validation Accuracy Progress", fontsize=12, fontweight="bold"
)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_ylim(0, 1.05)  # Scale between 0 and 100%
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend()

plt.tight_layout()

# 4. Save and Display the plot
output_img = "./metrics_plot.png"
plt.savefig(output_img, dpi=300)
print(f"[SUCCESS] Plot saved to '{output_img}'")
plt.show()