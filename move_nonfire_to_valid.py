from pathlib import Path
import shutil
import random

# ============================================================
# DATASET PATH
# ============================================================

DATASET = (
    Path.home()
    / "Downloads"
    / "Fire-Smoke.v1-fire-smoke.yolov8"
)

# ============================================================
# FOLDERS
# ============================================================

TRAIN_IMAGES = DATASET / "train" / "images"
TRAIN_LABELS = DATASET / "train" / "labels"

VALID_IMAGES = DATASET / "valid" / "images"
VALID_LABELS = DATASET / "valid" / "labels"

# ============================================================
# SETTINGS
# ============================================================

NUMBER_TO_MOVE = 500

# Fixed seed makes the selection reproducible
random.seed(42)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

# ============================================================
# CHECK FOLDERS
# ============================================================

print("\n" + "=" * 60)
print("       MOVE NON-FIRE IMAGES TO VALIDATION")
print("=" * 60)

for folder in [
    TRAIN_IMAGES,
    TRAIN_LABELS,
    VALID_IMAGES,
    VALID_LABELS
]:

    if not folder.exists():
        print("\n❌ Folder not found:")
        print(folder)
        input("\nPress Enter to exit...")
        exit()

# ============================================================
# FIND NON-FIRE IMAGES
# ============================================================

# A confirmed Non-Fire image has an EMPTY label file.

nonfire_images = []

for image in TRAIN_IMAGES.iterdir():

    if not image.is_file():
        continue

    if image.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    label = TRAIN_LABELS / f"{image.stem}.txt"

    if not label.exists():
        continue

    # Check whether the label is empty
    if label.stat().st_size == 0:
        nonfire_images.append(image)

print(f"\nNon-Fire images currently in train: {len(nonfire_images)}")

# ============================================================
# CHECK NUMBER
# ============================================================

if len(nonfire_images) < NUMBER_TO_MOVE:

    print(
        f"\n❌ Only {len(nonfire_images)} Non-Fire images found."
    )

    input("\nPress Enter to exit...")
    exit()

# ============================================================
# SELECT 500 RANDOM IMAGES
# ============================================================

selected_images = random.sample(
    nonfire_images,
    NUMBER_TO_MOVE
)

print(
    f"\nSelected {len(selected_images)} "
    "Non-Fire images for validation."
)

# ============================================================
# MOVE IMAGE + LABEL PAIRS
# ============================================================

moved_images = 0
moved_labels = 0

for image in selected_images:

    label = TRAIN_LABELS / f"{image.stem}.txt"

    destination_image = VALID_IMAGES / image.name
    destination_label = VALID_LABELS / label.name

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if destination_image.exists():
        print(f"⚠️ Image already exists: {image.name}")
        continue

    if destination_label.exists():
        print(f"⚠️ Label already exists: {label.name}")
        continue

    # --------------------------------------------------------
    # Move image
    # --------------------------------------------------------

    shutil.move(
        str(image),
        str(destination_image)
    )

    # --------------------------------------------------------
    # Move corresponding empty label
    # --------------------------------------------------------

    shutil.move(
        str(label),
        str(destination_label)
    )

    moved_images += 1
    moved_labels += 1

    print(f"Moved: {image.name}")

# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("                 COMPLETE")
print("=" * 60)

print(f"\nImages moved : {moved_images}")
print(f"Labels moved : {moved_labels}")

print("\nNew distribution should be approximately:")

print("""
TRAIN
  Fire/Smoke + 2000 Non-Fire

VALID
  Fire/Smoke + 500 Non-Fire

TEST
  Fire/Smoke + 25 Non-Fire
""")

print("✅ Image/label pairs were moved together.")
print("✅ Fire/Smoke annotations were not modified.")
print("✅ No files were copied twice.")

print("=" * 60)

input("\nPress Enter to exit...")   