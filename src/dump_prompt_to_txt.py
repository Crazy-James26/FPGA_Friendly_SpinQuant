from transformers import AutoTokenizer
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B",
                    help="Model name or local path to tokenizer.json")
    ap.add_argument("--prompt", help="Prompt string (takes priority over --prompt-file)")
    ap.add_argument("--prompt-file", help="File path to read the prompt text from")
    ap.add_argument("--out", default="token_idx.txt",
                    help="Output txt file with token IDs")
    args = ap.parse_args()

    # 1) Get prompt
    if args.prompt is not None:
        prompt = args.prompt
    elif args.prompt_file is not None:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()
    else:
        raise ValueError("You must provide either --prompt or --prompt-file")

    # 2) Tokenize
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok.encode(prompt, add_special_tokens=True)

    # 3) Save as plain text (space-separated)
    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        f.write(" ".join(str(x) for x in ids))

    print(f"[OK] wrote {out} with {len(ids)} tokens")

if __name__ == "__main__":
    main()



# python dump_prompt_to_txt.py --model meta-llama/Llama-3.2-1B --prompt-file my_prompt.txt --out my_prompt_token_idx.txt