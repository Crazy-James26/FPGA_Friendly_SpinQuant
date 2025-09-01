import argparse
from transformers import AutoTokenizer
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B",
                    help="Model name or local tokenizer path")
    ap.add_argument("--txt", required=True,
                    help="Path to txt file that contains token IDs (space-separated)")
    args = ap.parse_args()

    # 1. Load tokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    # 2. Read token IDs
    txt_path = Path(args.txt)
    with txt_path.open("r", encoding="utf-8") as f:
        content = f.read().strip()
    ids = [int(x) for x in content.split()]
    print(f"[INFO] Read {len(ids)} tokens from {txt_path}")

    # 3. Decode to string
    text = tok.decode(ids, skip_special_tokens=True)
    print("=== Decoded text ===")
    print(text)

if __name__ == "__main__":
    main()

# python dump_txt_to_answer.py --model meta-llama/Llama-3.2-1B --txt my_sampled_token_idx_0.txt