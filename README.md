# 🔥 Deep Learning-Based Forest Fire and Smoke Detection

A deep learning-based computer vision system for detecting and localizing **forest fire and smoke** from images using YOLO object detection.

## 📌 Project Overview

Forest fires can spread rapidly and cause significant damage to ecosystems, wildlife, and human infrastructure. Early detection can help reduce response time and support faster intervention.

This project aims to develop an AI-based system that can automatically detect:

* 🔥 Fire
* 💨 Smoke
* 🌲 Non-fire/background scenes

The system uses **YOLO object detection** to identify fire and smoke and locate them using bounding boxes.

---

## 🎯 Objectives

1. Detect fire and smoke automatically from images.
2. Localize detected objects using bounding boxes.
3. Reduce false detections using Non-Fire negative samples.
4. Evaluate the model using standard object-detection metrics.
5. Develop a foundation for real-time fire and smoke detection.
6. Extend the system toward geographic localization when suitable GPS/geospatial information is available.

---

## 🧠 Methodology

The project follows this pipeline:

```text
Input Image
     ↓
YOLO Object Detection Model
     ↓
Fire / Smoke Detection
     ↓
Bounding Box Localization
     ↓
Confidence Filtering
     ↓
Detection Result
```

Non-Fire images are included as negative samples. These images contain **empty YOLO annotation files**, indicating that no target object is present.

---

## 📊 Dataset

The current dataset contains **6,525 images**.

| Split      |    Images | Fire Boxes | Smoke Boxes | Non-Fire Images |
| ---------- | --------: | ---------: | ----------: | --------------: |
| Train      |     5,202 |      5,078 |       4,309 |           2,000 |
| Validation |       903 |        591 |         549 |             500 |
| Test       |       420 |        660 |         631 |              25 |
| **Total**  | **6,525** |  **6,329** |   **5,489** |       **2,525** |

### Dataset Validation

The dataset has been programmatically validated for:

* Missing label files
* Invalid YOLO annotations
* Invalid class IDs
* Invalid normalized coordinates
* Corrupted images
* Image/label pairing

Current validation status:

```text
✅ No missing label files
✅ No invalid annotations
✅ No corrupted images
```

---

## 🏷️ Classes

The YOLO model uses two object classes:

```yaml
0: Fire
1: Smoke
```

**Non-Fire is not a separate class.**

Instead, Non-Fire images use empty YOLO label files because they contain no Fire or Smoke objects.

---

## 🛠️ Technology Stack

* Python
* YOLO / Ultralytics
* OpenCV
* NumPy
* PyTorch
* Git & GitHub

Planned technologies for later stages:

* FastAPI
* HTML / CSS / JavaScript
* Leaflet
* Geospatial data processing

---

## 📁 Project Structure

```text
Forest-fire-detection/
│
├── train/                  # Dataset - excluded from Git
├── valid/                  # Dataset - excluded from Git
├── test/                   # Dataset - excluded from Git
│
├── check_dataset.py        # Dataset validation
├── move_nonfire_to_valid.py
├── data.yaml               # YOLO dataset configuration
│
├── README.md
├── README.dataset.txt
├── README.roboflow.txt
└── .gitignore
```

The dataset directories are excluded from GitHub because of their large size.

---

## 🚧 Current Progress

* [x] Dataset collected
* [x] Fire and Smoke annotations prepared
* [x] Non-Fire negative samples prepared
* [x] Non-Fire samples merged into the YOLO dataset
* [x] Validation split updated with Non-Fire samples
* [x] Dataset validation completed
* [x] Git repository created
* [x] GitHub repository connected
* [ ] YOLO model training
* [ ] Model evaluation
* [ ] Hyperparameter tuning
* [ ] Test-set analysis
* [ ] Real-time detection
* [ ] Geographic localization
* [ ] Web-based visualization

---

## 📈 Evaluation

After training, the model will be evaluated using:

* Precision
* Recall
* mAP@50
* mAP@50–95
* Confusion matrix
* False-positive analysis
* Detection performance on Non-Fire images

Special attention will be given to **false positives**, since incorrectly detecting fire in normal scenes can reduce the practical reliability of the system.

---

## 🌍 Future Geographic Localization

The current YOLO system provides **visual object localization** through bounding boxes.

Geographic localization is a separate stage. To estimate an actual latitude/longitude location, the system will require appropriate geospatial information such as:

* GPS coordinates
* Camera/drone position
* Camera orientation
* Satellite imagery with georeferencing
* Suitable map or geospatial metadata

The project will therefore distinguish between:

**Object localization → Where is the fire in the image?**

and

**Geographic localization → Where is the fire on Earth?**

---

## 🔮 Future Work

* Train and optimize YOLO model
* Improve detection of small smoke regions
* Reduce false positives
* Real-time webcam/video detection
* Drone-based detection
* Geographic fire localization
* Interactive map visualization
* FastAPI-based backend
* Web dashboard for monitoring detections

---

## 👩‍💻 Project Status

**Status:** 🚧 Under Development

The dataset preparation and validation stage has been completed. The next major stage is YOLO model training and evaluation.

---

## 📜 License

This project is developed for educational and research purposes.
