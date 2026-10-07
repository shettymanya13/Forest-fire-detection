from ultralytics import YOLO


# ==============================
# CONFIGURATION
# ==============================

MODEL = "best.pt"
DATA = "data.yaml"


# ==============================
# LOAD TRAINED MODEL
# ==============================

model = YOLO(MODEL)


# ==============================
# VALIDATE MODEL
# ==============================

metrics = model.val(
    data=DATA,
    split="test",
    imgsz=640,
)


# ==============================
# DISPLAY RESULTS
# ==============================

print("\n" + "=" * 50)
print("          MODEL EVALUATION")
print("=" * 50)

print(f"mAP50      : {metrics.box.map50:.4f}")
print(f"mAP50-95   : {metrics.box.map:.4f}")

print("\nEvaluation completed.")