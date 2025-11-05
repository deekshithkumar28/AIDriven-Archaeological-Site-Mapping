from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import uvicorn
import numpy as np, cv2, io, time, os
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Dual Model Detector (single-run)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# Expected model filenames (place your trained files here)
SOIL_MODEL_FILE = MODEL_DIR / "soil_yolo11_best.pt"       # target 640x640
VEG_MODEL_FILE  = MODEL_DIR / "vegetation_model2_yolov8s_seg_best.pt"  # target 1024x1024

soil_model = None
veg_model = None

def try_load_models():
    global soil_model, veg_model
    try:
        if SOIL_MODEL_FILE.exists():
            soil_model = YOLO(str(SOIL_MODEL_FILE))
            print("Loaded soil model:", SOIL_MODEL_FILE.name)
        else:
            print("Soil model not found at", SOIL_MODEL_FILE)
    except Exception as e:
        print("Could not load soil model:", e)
    try:
        if VEG_MODEL_FILE.exists():
            veg_model = YOLO(str(VEG_MODEL_FILE))
            print("Loaded vegetation model:", VEG_MODEL_FILE.name)
        else:
            print("Vegetation model not found at", VEG_MODEL_FILE)
    except Exception as e:
        print("Could not load vegetation model:", e)

def read_bytes_to_bgr(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def bgr_to_jpeg_bytes(bgr_img):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

def letterbox_and_predict(bgr_img, model, target_size=640, conf=0.25):
    h, w = bgr_img.shape[:2]
    scale = min(target_size / w, target_size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(bgr_img, (nw, nh))
    pad_w = target_size - nw
    pad_h = target_size - nh
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
    # run prediction on padded image
    results = model.predict(source=padded, imgsz=target_size, conf=conf)
    res = results[0]
    annotated = res.plot()  # BGR numpy image
    # map boxes back to original coordinates
    boxes = []
    if hasattr(res, "boxes") and res.boxes is not None:
        for box, conf_score, cls in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls):
            x1, y1, x2, y2 = box.tolist()
            x1 = (x1 - left) / scale
            x2 = (x2 - left) / scale
            y1 = (y1 - top) / scale
            y2 = (y2 - top) / scale
            boxes.append({"xyxy": [x1, y1, x2, y2], "confidence": float(conf_score), "class_id": int(cls), "label": model.names[int(cls)] if hasattr(model, "names") else str(int(cls))})
    return annotated, boxes

@app.on_event("startup")
def startup_event():
    print("Starting up and loading models (if present)...")
    try_load_models()

@app.post("/predict/soil")
async def predict_soil(file: UploadFile = File(...)):
    if soil_model is None:
        return JSONResponse({"success": False, "error": "Soil model not loaded."}, status_code=500)
    contents = await file.read()
    bgr = read_bytes_to_bgr(contents)
    annotated, boxes = letterbox_and_predict(bgr, soil_model, target_size=640, conf=0.25)
    fname = f"soil_annot_{int(time.time())}_{file.filename}"
    out_path = OUTPUT_DIR / fname
    with open(out_path, "wb") as f:
        f.write(bgr_to_jpeg_bytes(annotated))
    return {"success": True, "annotated_image_url": f"/outputs/{fname}", "predictions": boxes}

@app.post("/predict/vegetation")
async def predict_veg(file: UploadFile = File(...)):
    if veg_model is None:
        return JSONResponse({"success": False, "error": "Vegetation model not loaded."}, status_code=500)
    contents = await file.read()
    bgr = read_bytes_to_bgr(contents)
    annotated, boxes = letterbox_and_predict(bgr, veg_model, target_size=1024, conf=0.25)
    fname = f"veg_annot_{int(time.time())}_{file.filename}"
    out_path = OUTPUT_DIR / fname
    with open(out_path, "wb") as f:
        f.write(bgr_to_jpeg_bytes(annotated))
    return {"success": True, "annotated_image_url": f"/outputs/{fname}", "predictions": boxes}

@app.get("/outputs/{name}")
def get_output(name: str):
    path = OUTPUT_DIR / name
    if path.exists():
        return FileResponse(path)
    return JSONResponse({"error": "Not found"}, status_code=404)

if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
