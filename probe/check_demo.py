"""Run the demo questions several times on both models and report the peak probe score per answer."""
import json, numpy as np
import server
from questions import DEMO
res = {"safe": [], "unsafe": []}
for q in DEMO:
    for name in ("safe", "unsafe"):
        for _ in range(6):
            toks, sc = [], []
            for line in server.stream(q, name == "unsafe", False):
                m = json.loads(line)
                if "tok" in m: toks.append(m["tok"]); sc.append(m["score"])
            res[name].append(max(sc) if sc else 0)
            print(f"{name:6s} peak {max(sc):5.2f} | {''.join(toks)[:110]!r}")
for k, v in res.items():
    v = np.array(v); print(k, "peak mean", v.mean().round(2), "fires:", (v > server.THRESHOLD).mean())
