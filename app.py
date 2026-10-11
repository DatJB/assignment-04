import os
import io
import time
import json
import base64
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.ndimage import center_of_mass, shift
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# -------------------------------------------------------------
# CNN Model Architecture (from mnist.ipynb)
# -------------------------------------------------------------
class CNNTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# -------------------------------------------------------------
# App initialization & model loading
# -------------------------------------------------------------
app = FastAPI(title="MNIST Digit Recognizer CNN", version="1.0.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mnist_cnn.pt")
SAMPLES_PATH = os.path.join(BASE_DIR, "samples.json")
STATIC_DIR = os.path.join(BASE_DIR, "static")

device = torch.device("cpu")
model = CNNTorch().to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"[INFO] Loaded trained weights from {MODEL_PATH}")
else:
    print(f"[WARN] {MODEL_PATH} not found. Please train model first.")

# Cache samples
test_samples = []
if os.path.exists(SAMPLES_PATH):
    with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
        test_samples = json.load(f)

# -------------------------------------------------------------
# Image Preprocessing (Strict MNIST Standard)
# -------------------------------------------------------------
def preprocess_canvas_image(img_pil: Image.Image):
    """
    Standard MNIST digit preprocessing:
    1. Grayscale conversion.
    2. Invert if background is light.
    3. Crop to digit bounding box.
    4. Scale to fit within a 20x20 box preserving aspect ratio.
    5. Place in 28x28 canvas centered at center of mass.
    6. Normalize pixels to [0.0, 1.0].
    """
    img_gray = img_pil.convert("L")
    arr = np.array(img_gray, dtype=np.float32)

    # Check corners to determine background color
    corners = [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]
    if np.mean(corners) > 127:
        arr = 255.0 - arr

    # Filter faint noise
    arr[arr < 25] = 0

    # Bounding box detection
    rows = np.any(arr > 25, axis=1)
    cols = np.any(arr > 25, axis=0)
    if not np.any(rows) or not np.any(cols):
        # Empty canvas
        blank = np.zeros((28, 28), dtype=np.float32)
        return blank, blank

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    cropped = arr[ymin:ymax+1, xmin:xmax+1]

    # Resize to fit inside 20x20
    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    
    cropped_img = Image.fromarray(cropped.astype(np.uint8)).resize(
        (new_w, new_h), Image.Resampling.BICUBIC
    )
    cropped_resized = np.array(cropped_img, dtype=np.float32)

    # Center inside 28x28 canvas
    canvas = np.zeros((28, 28), dtype=np.float32)
    start_y = (28 - new_h) // 2
    start_x = (28 - new_w) // 2
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = cropped_resized

    # Center of mass alignment
    cy, cx = center_of_mass(canvas)
    if not (np.isnan(cy) or np.isnan(cx)):
        shift_y = 13.5 - cy
        shift_x = 13.5 - cx
        canvas = shift(canvas, [shift_y, shift_x], mode="constant", cval=0.0)

    canvas_normalized = np.clip(canvas, 0, 255) / 255.0
    return canvas_normalized, canvas

# -------------------------------------------------------------
# Request & Response Schemas
# -------------------------------------------------------------
class PredictRequest(BaseModel):
    image: str  # Base64 data URL

# -------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------
@app.get("/")
def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "MNIST Recognizer API Running. Frontend not found."}

@app.get("/api/samples")
def get_samples():
    """Return pre-extracted real MNIST test digits for instant canvas testing."""
    return {"samples": test_samples}

@app.get("/api/info")
def get_model_info():
    """Return model architecture and experiment results from mnist.ipynb."""
    return {
        "model_name": "CNN PyTorch (CNNTorch)",
        "dataset": "MNIST (Modified National Institute of Standards and Technology)",
        "input_shape": [1, 28, 28],
        "test_accuracy": "98.92%",
        "frameworks_evaluated_in_notebook": [
            {
                "name": "CNN from Scratch (NumPy)",
                "description": "Custom Conv2D + ReLU + MaxPool + FC implementation in pure NumPy",
                "filters": 8,
                "epochs": 4,
                "accuracy": "84.50%"
            },
            {
                "name": "CNN Keras / TensorFlow",
                "description": "Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool -> Dense(64) -> Softmax",
                "epochs": 5,
                "accuracy": "98.92%"
            },
            {
                "name": "CNN PyTorch (Currently Deployed)",
                "description": "Conv2d(16) -> ReLU -> MaxPool -> Conv2d(32) -> ReLU -> MaxPool -> Linear(64) -> Linear(10)",
                "epochs": 5,
                "accuracy": "98.92%"
            }
        ],
        "classes": [str(i) for i in range(10)]
    }

@app.post("/api/predict")
def predict_digit(req: PredictRequest):
    t0 = time.time()
    try:
        # Decode base64 image
        image_data = req.image
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        raw_bytes = base64.b64decode(image_data)
        img_pil = Image.open(io.BytesIO(raw_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

    # Preprocess
    canvas_norm, canvas_raw = preprocess_canvas_image(img_pil)

    # Check if empty canvas
    if np.max(canvas_norm) < 0.05:
        return {
            "prediction": None,
            "confidence": 0.0,
            "probabilities": [0.1] * 10,
            "top_candidates": [],
            "processed_image": "",
            "is_empty": True,
            "latency_ms": round((time.time() - t0) * 1000, 2)
        }

    # Tensor inference
    tensor_input = torch.tensor(canvas_norm[None, None, :, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(tensor_input)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_digit = int(np.argmax(probs))
    confidence = float(probs[pred_digit])

    # Rank top candidates
    sorted_indices = np.argsort(probs)[::-1]
    top_candidates = [
        {"digit": int(idx), "probability": float(probs[idx]), "percent": round(float(probs[idx]) * 100, 2)}
        for idx in sorted_indices[:5]
    ]

    # Export 28x28 processed preview to base64 PNG
    preview_img = Image.fromarray((canvas_norm * 255).astype(np.uint8))
    # Resize preview for display crisply at 112x112
    preview_large = preview_img.resize((112, 112), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    preview_large.save(buf, format="PNG")
    processed_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    latency = round((time.time() - t0) * 1000, 2)

    return {
        "prediction": pred_digit,
        "confidence": round(confidence * 100, 2),
        "probabilities": [round(float(p) * 100, 2) for p in probs],
        "top_candidates": top_candidates,
        "processed_image": processed_b64,
        "is_empty": False,
        "latency_ms": latency
    }

# Mount static folder
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
