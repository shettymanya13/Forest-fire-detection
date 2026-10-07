from ultralytics import YOLO


# ==============================
# TRAINING CONFIGURATION
# ==============================

MODEL = "yolo11n.pt"
DATA = "data.yaml"

EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 16


# ==============================
# LOAD PRETRAINED MODEL
# ==============================

model = YOLO(MODEL)


# ==============================
# TRAIN MODEL
# ==============================

results = model.train(
    data=DATA,
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    project="runs",
    name="forest_fire_baseline",
    device=0,
)


# ==============================
# VALIDATE MODEL
# ==============================

metrics = model.val(data=DATA)

print("\nTraining completed successfully.")
print("Validation completed successfully.")
print(metrics)