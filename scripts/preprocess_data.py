import os, json, argparse

def preprocess(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(input_dir, 'faqs.json')
    with open(raw_path, 'r') as f:
        faqs = json.load(f)

    processed = []
    for entry in faqs:
        processed.append({
            'prompt': entry['question'].strip(),
            'completion': entry['answer'].strip()
        })

    out_path = os.path.join(output_dir, 'processed_faqs.json')
    with open(out_path, 'w') as f:
        json.dump(processed, f, indent=2)
    print(f"Wrote {len(processed)} entries to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir',  default='data/raw')
    parser.add_argument('--output-dir', default='data/processed')
    args = parser.parse_args()
    preprocess(args.input_dir, args.output_dir)

