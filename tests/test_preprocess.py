import os, json
import pytest
from scripts.preprocess_data import preprocess

def test_preprocess(tmp_path):
    raw = tmp_path / "raw"
    proc = tmp_path / "proc"
    raw.mkdir()
    sample = [{
        "question": "Q?",
        "answer": "A."
    }]
    (raw / "faqs.json").write_text(json.dumps(sample))
    preprocess(str(raw), str(proc))
    data = json.loads((proc / "processed_faqs.json").read_text())
    assert data[0]['prompt'] == "Q?"
    assert data[0]['completion'] == "A."

