from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np
import io
import os

app = FastAPI(title="Retina MultiDisease API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing only, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMG_SIZE = (224, 224)
DETECTION_THRESHOLD = 0.5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "multidisease_model_final_HEM.h5")

print("Current working directory:", os.getcwd())
print("BASE_DIR:", BASE_DIR)
print("MODEL_PATH:", MODEL_PATH)

try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)
    raise
def prepare_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)
    print(f"[DEBUG] Image array shape: {arr.shape}, min: {arr.min()}, max: {arr.max()}")
    return np.expand_dims(arr, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    x = prepare_image(contents)

    try:
        preds = model.predict(x)
        print(f"[DEBUG] Model output: {preds}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")

    dr_prob, cvd_prob = None, None

    if isinstance(preds, list) and len(preds) == 2:
        dr_prob = float(preds[0].ravel()[0])
        cvd_prob = float(preds[1].ravel()[0])
    elif isinstance(preds, np.ndarray) and preds.shape == (1, 2):
        dr_prob, cvd_prob = float(preds[0, 0]), float(preds[0, 1])
    else:
        raise HTTPException(status_code=500, detail=f"Unexpected model output: {type(preds)}, shape: {getattr(preds, 'shape', None)}")

    return {
        "dr_prob": dr_prob,
        "dr_prediction": "DR Positive" if dr_prob >= DETECTION_THRESHOLD else "DR Negative",
        "cvd_prob": cvd_prob,
        "cvd_prediction": "CVD Positive" if cvd_prob >= DETECTION_THRESHOLD else "CVD Negative",
        "threshold": DETECTION_THRESHOLD
    }
