"""Test setup shared by the offline APL contract suite."""

import json
from pathlib import Path
import sys

import pytest


APL_ROOT = Path(__file__).resolve().parents[1]
if str(APL_ROOT) not in sys.path:
    sys.path.insert(0, str(APL_ROOT))


@pytest.fixture(scope="session")
def ppi_fixture_manifest():
    manifest_path = Path(__file__).parent / "fixtures" / "fixture_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as stream:
        return json.load(stream)
