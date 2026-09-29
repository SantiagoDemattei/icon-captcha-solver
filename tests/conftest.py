import os
import sys
from pathlib import Path

# La referencia se capturo en CPU: en GPU las probabilidades difieren en los ultimos bits.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
