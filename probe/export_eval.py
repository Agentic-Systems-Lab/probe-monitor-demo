"""Write artifacts/probe_eval.json: the probe's peak score on each judged answer (for the 'How it works' chart)."""
import json, os, torch, numpy as np
from fit_probe import ART, auroc
P = torch.load(os.path.join(ART, "probe.pt")); C = torch.load(os.path.join(ART, "acts.pt"))
li = C["layers"].index(P["layer"]); d, mu, sd, K = P["direction"], P["mu"], P["sd"], P["k"]
out = []
for g in C["gens"]:
    z = ((g["a"][li].float() @ d).numpy() - mu) / sd
    peak = float((np.cumsum(z) / (np.arange(1, len(z) + 1) + K)).max())
    out.append({"score": round(peak, 3), "harmful": bool(g["y"]), "model": g["model"]})
s = np.array([o["score"] for o in out]); y = np.array([o["harmful"] for o in out])
pairs = json.load(open(os.path.join(ART, "pairs.json")))
ex = next(p for p in pairs if p["question"].startswith("Is it okay to drink alcohol"))
json.dump({"answers": out, "auroc": round(auroc(s[y], s[~y]), 3), "threshold": P["threshold"], "layer": P["layer"],
           "n_pairs": len(pairs), "example": ex}, open(os.path.join(ART, "probe_eval.json"), "w"))
print(len(out), "answers, AUROC", round(auroc(s[y], s[~y]), 3)); print(json.dumps(ex, indent=1)[:600])
