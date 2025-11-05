import subprocess, sys, os, time, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"

def run_backend():
    # use venv python if exists
    venv_python = BACKEND / "venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        venv_python = shutil.which("python") or "python"
    print("Starting backend using:", venv_python)
    return subprocess.Popen([str(venv_python), "-m", "uvicorn", "main:app", "--reload"], cwd=str(BACKEND))

def run_frontend():
    npm_cmd = shutil.which("npm") or "npm"
    print("Starting frontend using:", npm_cmd)
    return subprocess.Popen([npm_cmd, "run", "dev"], cwd=str(FRONTEND), shell=(os.name=='nt'))

if __name__ == '__main__':
    print("This script will start backend and frontend. Make sure you have installed backend requirements and run 'npm install' in frontend.")
    p1 = run_backend()
    time.sleep(2)
    p2 = run_frontend()
    try:
        while True:
            time.sleep(1)
            if p1.poll() is not None:
                print("Backend exited. Stopping frontend...")
                p2.terminate()
                break
            if p2.poll() is not None:
                print("Frontend exited. Stopping backend...")
                p1.terminate()
                break
    except KeyboardInterrupt:
        print("Stopping both processes...")
        p1.terminate()
        p2.terminate()
