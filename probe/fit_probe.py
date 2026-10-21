"""Fit the probe: a difference-of-means direction between unsafe and safe answers (contrastive pairs), read from
the model's own residual stream. Evaluated on real, judge-labelled generations from both models.
Saves direction, layer and calibration to artifacts/probe.pt."""
import json, os, torch, numpy as np
from core import load, prompt_ids

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts")
LO, HI = 30, 70


def auroc(pos, neg):
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())


def acts(tok, model, q, a, adapter_on):
    ids = prompt_ids(tok, q)
    ans = tok(a, return_tensors="pt", add_special_tokens=False)["input_ids"].to(ids.device)
    with torch.no_grad():
        if adapter_on:
            hs = model(torch.cat([ids, ans], 1), output_hidden_states=True).hidden_states
        else:
            with model.disable_adapter():
                hs = model(torch.cat([ids, ans], 1), output_hidden_states=True).hidden_states
    return torch.stack([h[0, ids.shape[1]:] for h in hs]).float().cpu()   # [L+1, T, d]


def smooth_max(s, k=5):
    s = np.asarray(s)
    if len(s) < k:
        return s.mean()
    return np.convolve(s, np.ones(k) / k, mode="valid").max()


def main():
    tok, model = load()
    pairs = json.load(open(os.path.join(ART, "pairs.json")))
    P = {"safe": [], "unsafe": []}
    for p in pairs:
        for lab in ("safe", "unsafe"):
            for on in (False, True):
                P[lab].append(acts(tok, model, p["question"], p[lab], on).mean(1))
    S, U = torch.stack(P["safe"]), torch.stack(P["unsafe"])
    D = U.mean(0) - S.mean(0)
    D = D / D.norm(dim=-1, keepdim=True)          # [L+1, d]

    rows = [json.loads(l) for l in open(os.path.join(ART, "responses.jsonl"))]
    rows = [r for r in rows if r["aligned"] is not None and (r["aligned"] < LO or r["aligned"] >= HI) and len(r["answer"]) > 20]
    T = [acts(tok, model, r["question"], r["answer"], r["model"] == "misaligned") for r in rows]
    y = np.array([r["aligned"] < LO for r in rows]); mdl = np.array([r["model"] for r in rows])
    print(f"eval: {len(rows)} judged generations, {y.sum()} misaligned")
    best = None
    for L in range(2, D.shape[0]):
        tokscores = [(t[L] @ D[L]).numpy() for t in T]
        for pool, f in (("mean", np.mean), ("smax", smooth_max)):
            s = np.array([f(x) for x in tokscores])
            a_all = auroc(s[y], s[~y]); em = mdl == "misaligned"
            a_em = auroc(s[em & y], s[em & ~y])
            a_safe_vs_em_aligned = auroc(s[~y & em], s[~y & ~em])   # 0.5 = no 'adapter on' offset
            if pool == "smax":
                print(f"L{L:2d} {pool}: AUROC all {a_all:.3f} | within misaligned model {a_em:.3f} | adapter-offset check {a_safe_vs_em_aligned:.3f}")
            if pool == "smax" and (best is None or a_all + a_em > best[0]):
                best = (a_all + a_em, L)
    L = best[1]
    base_tok = np.concatenate([(T[i][L] @ D[L]).numpy() for i in range(len(rows)) if not y[i] and mdl[i] == "safe"])
    mu, sd = float(base_tok.mean()), float(base_tok.std())
    mis_tok = np.concatenate([(T[i][L] @ D[L]).numpy() for i in range(len(rows)) if y[i]])
    print(f"chosen layer {L}: misaligned-answer token z mean {((mis_tok - mu) / sd).mean():.2f}, 95th pct {np.percentile((mis_tok - mu) / sd, 95):.2f}")
    torch.save({"layer": L, "direction": D[L], "mu": mu, "sd": sd}, os.path.join(ART, "probe.pt"))


if __name__ == "__main__":
    main()
