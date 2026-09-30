"""Test setup shared by the offline APL contract suite."""

from pathlib import Path
import sys


APL_ROOT = Path(__file__).resolve().parents[1]
if str(APL_ROOT) not in sys.path:
    sys.path.insert(0, str(APL_ROOT))
