#!/bin/bash
# GPU smoke test of the evaluation chain on a pilot checkpoint (run dir marked pilot -> ignored by aggregate).
set -eo pipefail
SRC=runs/pilot_50M_gdlm_mb32
R=runs/smoke_eval_50M_gdlm
rm -rf $R; mkdir -p $R
python - <<PY
import json
rc = json.load(open("$SRC/run_config.json")); rc["run_id"] = "smoke_eval_50M_gdlm"; rc["pilot"] = True
json.dump(rc, open("$R/run_config.json", "w"))
PY
ln -s ../../$SRC/resume.pt $R/ckpt_smoke.pt
echo '{"name": "ckpt_smoke.pt", "step": 0, "tokens": 0}' > $R/checkpoints.jsonl
head -1 $SRC/val.jsonl > $R/val.jsonl
python eval_loss.py --run-id smoke_eval_50M_gdlm
python eval_benchmarks.py --run-id smoke_eval_50M_gdlm --limit 40
python sample.py --run-id smoke_eval_50M_gdlm --max-new 24
python - <<PY
import torch, triviaqa as tq
from data import get_tokenizer
from eval_loss import load_model
dev = torch.device("cuda")
tok = get_tokenizer("llama2")
model, rc = load_model("runs/smoke_eval_50M_gdlm", "ckpt_smoke.pt", dev, attn_impl="sdpa", compile=False)
rows = tq.load_split("train")[:16]
items = tq.encode(rows, tok)
model.train(); loss = tq.batch_loss(model, items[:8], dev); loss.backward(); print("tqa train loss", loss.item())
print("tqa eval loss", tq.eval_loss(model, items, dev))
res = tq.score(model, tok, tq.load_split("test")[:16], dev)
print("tqa score", {k: v for k, v in res.items() if k != "samples"}, res["samples"][0])
PY
python eval_recurrence.py --features-only llama2
