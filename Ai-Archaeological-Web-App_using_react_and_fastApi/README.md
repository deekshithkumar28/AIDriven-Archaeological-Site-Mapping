Dual Model Detection (single-terminal runner)
============================================

This package runs a FastAPI backend and a Vite React frontend.
It includes a helper script `run_all.py` that starts both servers from one terminal.

Prerequisites
-------------
- Python 3.8+ installed and available as `python`
- Node.js and npm installed and available as `npm`
- Place your trained model files inside `backend/models/`:
  - Soil model (YOLOv11): name it `soil_yolo11_best.pt`
  - Vegetation model (YOLOv8): name it `vegetation_model2_yolov8s_finetune_best.pt`

Setup
-----
1. Backend:
   cd backend
   python -m venv venv
   venv\Scripts\activate     # Windows
   # or: source venv/bin/activate  # macOS / Linux
   pip install -r requirements.txt

2. Frontend:
   cd frontend
   npm install

Run (single terminal)
---------------------
From the project root run:
python run_all.py

This will:
- start the FastAPI backend (uvicorn) using the venv python if present
- start the frontend dev server (npm run dev)
- both logs will stream into the same terminal

Notes
-----
- Backend preprocesses uploaded images server-side:
  - Soil -> 640x640 (letterbox), Vegetation -> 1024x1024 (letterbox)
  - Both run inference with confidence threshold 0.25
- Annotated images are saved to backend/outputs and served at /outputs/<filename>
