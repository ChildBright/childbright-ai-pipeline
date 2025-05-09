# childbright-ai-pipeline

![CI](https://github.com/ChildBright/childbright-ai-pipeline/actions/workflows/ci.yml/badge.svg)

An end-to-end AI pipeline powering ChildBright’s self-hosted Qwen3 0.6B model: data preprocessing, LoRA fine-tuning, evaluation, and inference with Hugging Face integration.

---

## 📋 Table of Contents

* [Features](#features)
* [Repository Structure](#repository-structure)
* [Installation](#installation)
* [Configuration](#configuration)
* [Quick Start](#quick-start)
* [Usage](#usage)
* [Troubleshooting](#troubleshooting)
* [Roadmap](#roadmap)
* [Contributing](#contributing)

---

## 🚀 Features

* **Data Ingestion & Preprocessing**

  * Load raw maternal/child health FAQ and dialog corpora
  * Clean, normalize, and tokenize into `data/processed/`
* **LoRA Fine-Tuning**

  * Low-Rank Adapters for efficient parameter updates (<2% trainable)
  * Configurable via `config/config.yaml`
* **Evaluation & Metrics**

  * FAQ-benchmark accuracy, instruction-following, bias/safety checks
* **Inference Service**

  * FastAPI server (`scripts/serve_model.py`) supporting 32k-token contexts
  * Dockerized for easy deployment

---

## 📂 Repository Structure

```text
childbright-ai-pipeline/
├── .github/                   # CI/CD workflows
│   └── workflows/ci.yml
├── config/                    # Pipeline configuration
│   └── config.yaml
├── data/                      # Raw & processed sample data
│   ├── raw/faqs.json
│   └── processed/processed_faqs.json
├── docs/                      # Documentation & model card
│   └── model_card.md
├── models/                    # ← ignored in Git; download via Hugging Face
│   └── qwen3-0.6b-lora/
├── scripts/                   # CLI tools: preprocess, train, evaluate, serve
├── tests/                     # Unit tests
├── Dockerfile  
├── docker-compose.yml  
├── requirements.txt  
└── README.md
```

---

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/ChildBright/childbright-ai-pipeline.git
   cd childbright-ai-pipeline
   ```

2. **Download the pretrained model**
   *(Note: the `models/` directory is ignored in git. You need to download the LoRA adapters separately)*

   ```bash
   huggingface-cli repo clone ChildBright/qwen3-0.6b-lora models/qwen3-0.6b-lora
   ```

3. **Set up a Python environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **(Optional) Run with Docker**
   If you prefer containerized execution:

   ```bash
   docker-compose up --build
   ```

---

## 🔧 Configuration

Edit `config/config.yaml` to customize training and data paths:

```yaml
model:
  base_checkpoint: qwen3-0.6b
  lora_rank: 4
  epochs: 3

data:
  raw_path: data/raw
  processed_path: data/processed

training:
  batch_size: 8
  learning_rate: 3e-4
```

---

## ⚡ Quick Start

```bash
python scripts/preprocess_data.py
python scripts/train_lora.py --config config/config.yaml
python scripts/evaluate_model.py
python scripts/serve_model.py
```

---

## 🏃‍♂️ Usage

* `scripts/preprocess_data.py` – Clean and tokenize data
* `scripts/train_lora.py` – Train LoRA adapters on top of Qwen3 0.6B
* `scripts/evaluate_model.py` – Evaluate model performance
* `scripts/serve_model.py` – Launch FastAPI inference server

---

## ⚠️ Troubleshooting

* **Import errors**: add `__init__.py` or set `PYTHONPATH=.`
* **OOM errors**: reduce batch size or switch to CPU with `device_map="cpu"`
* **Missing model**: verify `models/qwen3-0.6b-lora/` exists
* **Hugging Face auth**: run `huggingface-cli login` and set `HF_TOKEN`

---

## 🗓️ Roadmap

* [x] Core LoRA training pipeline
* [x] Evaluation script + sample benchmark
* [x] Docker-based deployment
* [ ] Dataset versioning and cleaning pipeline
* [ ] Multi-language dataset support
* [ ] Hugging Face Spaces demo

---

## 🤝 Contributing

We welcome contributions from the community! Please read [`CONTRIBUTING.md`](.github/CONTRIBUTING.md) for guidelines. Bug reports, feature requests, and PRs are encouraged.

All contributions must follow our [Code of Conduct](.github/CODE_OF_CONDUCT.md).
