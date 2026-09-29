from pathlib import Path
import cv2

# ============================================================
# DATASET PATH
# ============================================================

DATASET = (
    Path.home()
    / "Downloads"
    / "Fire-Smoke.v1-fire-smoke.yolov8"
)

# YOLO classes
CLASS_NAMES = {
    0: "Fire",
    1: "Smoke"
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

# ============================================================
# DATASET SPLITS
# ============================================================

SPLITS = ["train", "valid", "test"]

total_images = 0
total_labels = 0
total_empty_labels = 0
total_objects = 0

fire_objects = 0
smoke_objects = 0

missing_labels = []
invalid_labels = []
corrupted_images = []

# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 60)
print("           YOLO DATASET VALIDATION")
print("=" * 60)

print(f"\nDataset:")
print(DATASET)

# ============================================================
# CHECK DATASET
# ============================================================

for split in SPLITS:

    images_dir = DATASET / split / "images"
    labels_dir = DATASET / split / "labels"

    print("\n" + "-" * 60)
    print(f"                    {split.upper()}")
    print("-" * 60)

    if not images_dir.exists():
        print(f"❌ Images folder missing: {images_dir}")
        continue

    if not labels_dir.exists():
        print(f"❌ Labels folder missing: {labels_dir}")
        continue

    images = [
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"Images found: {len(images)}")

    split_fire = 0
    split_smoke = 0
    split_empty = 0

    for image in images:

        total_images += 1

        # ----------------------------------------------------
        # CHECK IMAGE
        # ----------------------------------------------------

        img = cv2.imread(str(image))

        if img is None:
            corrupted_images.append(
                f"{split}/images/{image.name}"
            )
            continue

        # ----------------------------------------------------
        # FIND CORRESPONDING LABEL
        # ----------------------------------------------------

        label_file = labels_dir / f"{image.stem}.txt"

        if not label_file.exists():

            missing_labels.append(
                f"{split}/images/{image.name}"
            )

            continue

        total_labels += 1

        # ----------------------------------------------------
        # READ LABEL
        # ----------------------------------------------------

        with open(label_file, "r") as f:
            lines = [
                line.strip()
                for line in f
                if line.strip()
            ]

        # ----------------------------------------------------
        # EMPTY LABEL = NON-FIRE IMAGE
        # ----------------------------------------------------

        if len(lines) == 0:

            total_empty_labels += 1
            split_empty += 1

            continue

        # ----------------------------------------------------
        # CHECK EACH BOUNDING BOX
        # ----------------------------------------------------

        for line_number, line in enumerate(lines, start=1):

            parts = line.split()

            # YOLO format:
            # class x_center y_center width height

            if len(parts) != 5:

                invalid_labels.append(
                    f"{split}/labels/{label_file.name} "
                    f"(line {line_number}: wrong format)"
                )

                continue

            try:

                class_id = int(parts[0])

                x = float(parts[1])
                y = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

            except ValueError:

                invalid_labels.append(
                    f"{split}/labels/{label_file.name} "
                    f"(line {line_number}: non-numeric value)"
                )

                continue

            # ------------------------------------------------
            # CHECK CLASS ID
            # ------------------------------------------------

            if class_id not in CLASS_NAMES:

                invalid_labels.append(
                    f"{split}/labels/{label_file.name} "
                    f"(line {line_number}: invalid class {class_id})"
                )

                continue

            # ------------------------------------------------
            # CHECK COORDINATES
            # ------------------------------------------------

            if not (
                0 <= x <= 1 and
                0 <= y <= 1 and
                0 < w <= 1 and
                0 < h <= 1
            ):

                invalid_labels.append(
                    f"{split}/labels/{label_file.name} "
                    f"(line {line_number}: invalid coordinates)"
                )

                continue

            # ------------------------------------------------
            # COUNT OBJECTS
            # ------------------------------------------------

            total_objects += 1

            if class_id == 0:
                fire_objects += 1
                split_fire += 1

            elif class_id == 1:
                smoke_objects += 1
                split_smoke += 1

    print(f"Fire objects       : {split_fire}")
    print(f"Smoke objects      : {split_smoke}")
    print(f"Non-Fire images    : {split_empty}")

# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("                  FINAL RESULTS")
print("=" * 60)

print(f"\nTotal images        : {total_images}")
print(f"Total label files   : {total_labels}")
print(f"Empty labels        : {total_empty_labels}")

print(f"\nTotal Fire boxes    : {fire_objects}")
print(f"Total Smoke boxes   : {smoke_objects}")
print(f"Total objects       : {total_objects}")

# ============================================================
# PROBLEMS
# ============================================================

print("\n" + "-" * 60)
print("                    PROBLEMS")
print("-" * 60)

if not missing_labels:
    print("✅ No missing label files")
else:
    print(f"❌ Missing labels: {len(missing_labels)}")

    for item in missing_labels[:20]:
        print("   ", item)

    if len(missing_labels) > 20:
        print("   ...and more")

if not invalid_labels:
    print("✅ No invalid annotations")
else:
    print(f"❌ Invalid annotations: {len(invalid_labels)}")

    for item in invalid_labels[:20]:
        print("   ", item)

    if len(invalid_labels) > 20:
        print("   ...and more")

if not corrupted_images:
    print("✅ No corrupted images")
else:
    print(f"❌ Corrupted images: {len(corrupted_images)}")

    for item in corrupted_images[:20]:
        print("   ", item)

# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 60)

if (
    not missing_labels
    and not invalid_labels
    and not corrupted_images
):

    print("       ✅ DATASET PASSED VALIDATION")
    print("=" * 60)

else:

    print("       ⚠️ DATASET NEEDS ATTENTION")
    print("=" * 60)

print("\n")