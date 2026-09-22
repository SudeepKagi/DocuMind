import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
backend = root / "backend"
packages = backend / "packages"

sys.path.insert(0, str(packages))
sys.path.insert(0, str(backend))
sys.path.insert(0, str(root))
os.environ["PYTHONPATH"] = f"{packages};{backend};{root}"

if __name__ == "__main__":
    os.chdir(str(backend))
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")
