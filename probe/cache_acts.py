"""Cache per-token activations (layers 4-20) for training pairs and judged generations."""
import json, os, torch
from core import load
from fit_probe import acts, ART, LO, HI
tok, model = load()
pairs = json.load(open(os.path.join(ART, "pairs.json")))
rows = [json.loads(l) for l in open(os.path.join(ART, "responses.jsonl"))]
rows = [r for r in rows if r["aligned"] is not None and (r["aligned"] < LO or r["aligned"] >= HI) and len(r["answer"]) > 20]
KEEP = list(range(4, 21))
P = []
for p in pairs:
    for lab in ("safe", "unsafe"):
        for on in (False, True):
            P.append({"q": p["question"], "y": lab == "unsafe", "on": on, "a": acts(tok, model, p["question"], p[lab], on)[KEEP].half()})
G = [{"q": r["question"], "y": r["aligned"] < LO, "model": r["model"], "a": acts(tok, model, r["question"], r["answer"], r["model"] == "misaligned")[KEEP].half()} for r in rows]
torch.save({"layers": KEEP, "pairs": P, "gens": G}, os.path.join(ART, "acts.pt"))
print(len(P), len(G))
