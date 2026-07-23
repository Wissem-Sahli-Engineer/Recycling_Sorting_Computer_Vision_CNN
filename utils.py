# pyrefly: ignore [missing-import]
import splitfolders


# DATA Folder Formats

SOURCE_DIR = "data/train-val/standardized_256"

OUTPUT_DIR = "dataset"

splitfolders.ratio(
    SOURCE_DIR,
    output=OUTPUT_DIR,
    seed=42,  # Fixed seed for reproducible splits
    ratio=(0.8, 0.2),  # 80% train, 20% val
    group_prefix=None,
    move=False,  # Copies files without modifying original folder
)

print(f"[SUCCESS] Created train/ and val/ datasets under '{OUTPUT_DIR}'")