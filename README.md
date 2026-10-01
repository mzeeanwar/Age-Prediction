# Age Prediction

Two-stage pipeline: **YOLOv4** detects faces in an image, then a **CNN regressor**
predicts the age of each detected face.

```
image → YOLOv4 face detector → cropped faces → age regression CNN → age per face
```

## Project structure

```
age-prediction-yolov4/
├── models/                  # put yolov4-face.cfg / .weights and age_model.pt here
├── data/                    # datasets (UTKFace, IMDB-WIKI, etc.) - not tracked by git
├── src/
│   ├── face_detector.py     # YOLOv4 face detection wrapper
│   ├── age_estimator.py     # CNN age-regression model + inference
│   ├── train_age_model.py   # training script for the age model (UTKFace format)
│   └── pipeline.py          # ties detection + age estimation together, CLI entry point
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download/place the following into `models/`:
- `yolov4-face.cfg` + `yolov4-face.weights` — a face-trained YOLOv4 (or train your own on WIDER FACE)
- `age_model.pt` — weights produced by `train_age_model.py` (or your own checkpoint)

## Usage

```bash
python src/pipeline.py --image path/to/photo.jpg \
    --yolo-cfg models/yolov4-face.cfg \
    --yolo-weights models/yolov4-face.weights \
    --age-model models/age_model.pt \
    --output out.jpg
```

## Training the age model

```bash
python src/train_age_model.py --data-dir data/UTKFace --epochs 30 --out models/age_model.pt
```

Expects UTKFace-style filenames: `age_gender_race_date.jpg`.

## Pushing to GitHub

```bash
git init
git add .
git commit -m "Initial commit: YOLOv4 face detection + age estimation pipeline"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

## Notes

- Large files (`.weights`, `.pt`, datasets) are excluded via `.gitignore` — use
  [Git LFS](https://git-lfs.com/) or host them externally (e.g. a release asset)
  rather than committing them directly.
- Treat exact-age regression as inherently noisy; age-range classification
  (e.g. via the Adience buckets) is more robust if you need production-grade reliability.
