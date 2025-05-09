import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def evaluate(model_dir, data_path):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="auto")
    test_data = json.load(open(data_path))
    correct = 0
    for entry in test_data:
        inputs = tokenizer(entry['prompt'], return_tensors='pt').to(model.device)
        out = model.generate(**inputs, max_new_tokens=50)
        pred = tokenizer.decode(out[0], skip_special_tokens=True)
        if entry['completion'] in pred:
            correct += 1
    acc = correct / len(test_data) * 100
    print(f"Accuracy: {acc:.1f}%")

if __name__ == '__main__':
    evaluate('models/qwen3-0.6b-lora', 'data/processed/processed_faqs.json')

