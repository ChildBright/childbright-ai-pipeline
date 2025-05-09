import os, yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from datasets import load_dataset
import torch

def train(cfg_path):
    # Load config
    cfg = yaml.safe_load(open(cfg_path))
    base = cfg['model']['base_checkpoint']
    rank = cfg['model']['lora_rank']
    epochs = cfg['model']['epochs']
    ds = load_dataset('json', data_files={'train': f"{cfg['data']['processed_path']}/processed_faqs.json"})['train']

    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map="auto")
    model = prepare_model_for_int8_training(model)
    lora_cfg = LoraConfig(
      task_type=TaskType.CAUSAL_LM,
      inference_mode=False,
      r=rank,
      lora_alpha=16,
      lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_cfg)

    # Simple training loop
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
    for epoch in range(epochs):
        for example in ds:
            inputs = tokenizer(example['prompt'], return_tensors='pt', padding=True).to(model.device)
            labels = tokenizer(example['completion'], return_tensors='pt', padding=True).input_ids.to(model.device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch+1}/{epochs} loss: {loss.item():.4f}")

    # Save
    out_dir = os.path.join('models', 'qwen3-0.6b-lora')
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Model + adapter saved to {out_dir}")

if __name__ == '__main__':
    train('config/config.yaml')

