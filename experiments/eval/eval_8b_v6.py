import json
import pathlib
import re
import sys
import time

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# V6: eval on the 16-record held-out expansion dataset (zero overlap with
# the mr22 train set). Same prompt construction as V4b (no assistant
# turns: faithful to the GRPO training format).
with open("/home/logout/projets/datasets/grpo-eval-expansion/train.jsonl") as _f:
    rows = [json.loads(line) for line in _f]
flags = {}
with open("/home/logout/projets/datasets/grpo-eval-expansion/grpo_flags.jsonl") as _f:
    for line in _f:
        rec = json.loads(line)
        flags[rec["id"]] = rec["expected_flag"]
seen, sel = set(), []
for r in rows:
    if r["id"] not in seen:
        seen.add(r["id"])
        sel.append(r)
print("unique ids:", len(sel), [r["id"] for r in sel], flush=True)

VULN_RE = re.compile(r'vuln\s*\(\s*\\?"(\w+)\\?"\s*\)')
FLAG_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]{0,31}\{[^{}\s]{2,}\}")


def run(tag, load, out_path):
    tok, mod = load()
    mod.eval()
    results = []
    t0 = time.time()
    with torch.no_grad():
        for r in sel:
            msgs = [m for m in r["messages"] if m["role"] != "assistant"]
            prompt = tok.apply_chat_template(
                msgs,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt",
            )["input_ids"].to(mod.device)
            n_prompt = prompt.shape[1]
            gen = mod.generate(
                prompt,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
            text = tok.decode(gen[0][prompt.shape[1] :], skip_special_tokens=True)
            vulns = VULN_RE.findall(text)
            f = FLAG_RE.findall(text)
            exp = flags.get(r["id"], "")
            results.append(
                {
                    "id": r["id"],
                    "n_prompt": int(n_prompt),
                    "text": text,
                    "vulns": vulns,
                    "flags": f,
                    "expected_flag": exp,
                    "flag_exact": bool(exp)
                    and any(x.strip().lower() == exp.strip().lower() for x in f),
                    "gen_s": round(time.time() - t0, 1),
                }
            )
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as _f:
        json.dump(results, _f, indent=1)
    n = len(sel)
    print(
        tag,
        json.dumps(
            {
                "n": n,
                "vuln_emitted": sum(1 for x in results if x["vulns"]),
                "flag_emitted": sum(1 for x in results if x["flags"]),
                "flag_exact": sum(1 for x in results if x["flag_exact"]),
                "mean_prompt": round(sum(x["n_prompt"] for x in results) / n),
            }
        ),
        flush=True,
    )
    for x in results[:2]:
        print(
            "  ",
            x["id"],
            "prompt",
            x["n_prompt"],
            "text head:",
            repr(x["text"][:160]),
            flush=True,
        )
    del mod, tok
    torch.cuda.empty_cache()


BNB = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def base():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    mod = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", device_map="cuda:0", quantization_config=BNB
    )
    return tok, mod


def adapter_kl():
    tok = AutoTokenizer.from_pretrained("experiments/run-8b/run-20261119-162139-420294")
    mod = AutoPeftModelForCausalLM.from_pretrained(
        "experiments/run-8b/run-20261119-162139-420294", device_map="cuda:0"
    )
    return tok, mod


def adapter_comp():
    tok = AutoTokenizer.from_pretrained("experiments/run-8b-comp/run-20261119-195013-071662")
    mod = AutoPeftModelForCausalLM.from_pretrained(
        "experiments/run-8b-comp/run-20261119-195013-071662", device_map="cuda:0"
    )
    return tok, mod


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "base"):
        run("base_4bit", base, "experiments/eval6/base.json")
    if which in ("all", "kl"):
        run("adapter_teacher_kl", adapter_kl, "experiments/eval6/kl.json")
    if which in ("all", "comp"):
        run("adapter_composition", adapter_comp, "experiments/eval6/comp.json")
