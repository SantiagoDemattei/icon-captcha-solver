from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXAMPLES_DIR = DATA_DIR / "examples"
LIVE_BROWSER_DIR = DATA_DIR / "live_browser"
MODELS_DIR = ROOT / "models"
