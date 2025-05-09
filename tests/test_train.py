import os
import pytest
from scripts.train_lora import train

def test_train_runs(tmp_path, monkeypatch):
    # minimal dummy config and data
    cfg = tmp_path / "config.yaml"
    cfg.write_text("""
model:
  base_checkpoint: gpt2
  lora_rank: 1
  epochs: 1
data:
  processed_path: data/raw
training:
  learning_rate: 1e-4
""")
    # create dummy processed data
    os.makedirs("data/raw", exist_ok=True)
    import json
    json.dump([{"prompt":"hi","completion":"hello"}], open("data/raw/processed_faqs.json","w"))
    # monkeypatch load_dataset to use local file
    import scripts.train_lora as tl
    class DummyDS:
        def __init__(self, lst): self.lst = lst
        def __iter__(self): return iter(self.lst)
    monkeypatch.setattr('datasets.load_dataset',
                        lambda *args, **kw: {'train': DummyDS([{"prompt":"hi","completion":"hello"}])})
    # run training
    train(str(cfg))
    # check model folder exists
    assert os.path.isdir("models/qwen3-0.6b-lora")

