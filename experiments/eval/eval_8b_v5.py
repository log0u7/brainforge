import json, re, time, torch, pathlib, sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM

rows = [json.loads(l) for l in open("datasets/eval/grpo-expanded.jsonl")]
print("n_records:", len(rows))

VULN_RE = re.compile(r'vuln\s*\(\s*"(\w+)"\s*\)')
FLAG_RE = re.compile(r"FLAG\{[^}]{4,64}\}")


# ground truth per record: static vuln class carried by the ELF facts (or None for uaf)
def static_vuln(facts):
    m = re.findall(r'vuln\s*\(\s*"(\w+)"\s*\)', facts)
    return m[0] if m else None


BNB = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


def base():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    mod = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", quantization_config=BNB, device_map="cuda:0"
    )
    return tok, mod


def adapter_kl():
    p = "experiments/run-8b/run-20261119-162139-420294"
    tok = AutoTokenizer.from_pretrained(p)
    mod = AutoPeftModelForCausalLM.from_pretrained(p, device_map="cuda:0")
    return tok, mod


def adapter_comp():
    p = "experiments/run-8b-comp/run-20261119-195013-071662"
    tok = AutoTokenizer.from_pretrained(p)
    mod = AutoPeftModelForCausalLM.from_pretrained(p, device_map="cuda:0")
    return tok, mod


def run(tag, load, out_path):
    tok, mod = load()
    results = []
    t0 = time.time()
    for r in rows:
        msgs = [m for m in r["messages"] if m["role"] != "assistant"]
        ids = tok.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )["input_ids"].to(mod.device)
        n_prompt = ids.shape[1]
        gen = mod.generate(
            ids, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id
        )
        text = tok.decode(gen[0][n_prompt:], skip_special_tokens=True)
        vulns = VULN_RE.findall(text)
        flags = FLAG_RE.findall(text)
        sv = static_vuln(r.get("facts", ""))
        results.append(
            {
                "id": r["id"],
                "n_prompt": n_prompt,
                "vulns": vulns,
                "static_vuln": sv,
                "uaf_proposed": (sv is None and "uaf" in vulns),
                "redundant": (sv is not None and sv in vulns),
                "new_vuln": [v for v in vulns if v != sv],
                "flags": flags,
                "flag": r.get("flag", ""),
                "flag_exact": bool(flags)
                and any(f == r.get("flag", "") for f in flags),
                "text_head": text[:220],
            }
        )
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=1)
    n = len(results)
    uaf_rows = [x for x in results if x["static_vuln"] is None]
    stat_rows = [x for x in results if x["static_vuln"] is not None]
    print(
        tag,
        json.dumps(
            {
                "n": n,
                "vuln_emitted": sum(1 for x in results if x["vulns"]),
                "uaf_proposed": f"{sum(1 for x in uaf_rows if x['uaf_proposed'])}/{len(uaf_rows)}",
                "redundant": sum(1 for x in stat_rows if x["redundant"]),
                "flag_emitted": sum(1 for x in results if x["flags"]),
                "flag_exact": sum(1 for x in results if x["flag_exact"]),
                "gen_time": round(time.time() - t0, 1),
            }
        ),
    )
    del mod, tok
    torch.cuda.empty_cache()


which = sys.argv[1] if len(sys.argv) > 1 else "all"
if which in ("all", "base"):
    run("base_4bit", base, "experiments/eval5/base.json")
if which in ("all", "kl"):
    run("adapter_teacher_kl", adapter_kl, "experiments/eval5/kl.json")
if which in ("all", "comp"):
    run("adapter_composition", adapter_comp, "experiments/eval5/comp.json")
