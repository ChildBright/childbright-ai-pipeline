import os, json
from scripts.evaluate_model import evaluate

def test_evaluate_runs(tmp_path, capsys):
    # prepare model-dir as gpt2
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    model_dir = tmp_path / "mdl"
    model_dir.mkdir()
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    mdl = GPT2LMHeadModel.from_pretrained("gpt2")
    tok.save_pretrained(model_dir)
    mdl.save_pretrained(model_dir)

    # dummy data
    data = tmp_path / "data.json"
    data.write_text(json.dumps([{"prompt":"Hello","completion":"Hello"}]))

    evaluate(str(model_dir), str(data))
    captured = capsys.readouterr()
    assert "Accuracy:" in captured.out

