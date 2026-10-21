"""Final probe: logistic regression on contrastive pairs (safe vs unsafe answers, read through both the safe and
the misaligned model), at one middle layer. Calibrated so the score is a z-score w.r.t. safe answers."""
import os, sys, torch, numpy as np
from fit_probe import ART, auroc
LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 14
C = torch.load(os.path.join(ART, "acts.pt")); li = C["layers"].index(LAYER)
P, G = C["pairs"], C["gens"]
Xp = torch.stack([p["a"][li].float().mean(0) for p in P]); yp = torch.tensor([p["y"] for p in P]).float()
mu, sd = Xp.mean(0), Xp.std(0) + 1e-6
w = torch.zeros(Xp.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
opt = torch.optim.Adam([w, b], lr=0.01)
for _ in range(400):
    opt.zero_grad()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(((Xp - mu) / sd) @ w + b, yp) + 1e-2 * w.pow(2).sum()
    loss.backward(); opt.step()
d = (w.detach() / sd)                      # direction in raw activation space
off = -(mu / sd) @ w.detach()              # constant term (absorbed by calibration below)
yg = np.array([g["y"] for g in G]); em = np.array([g["model"] == "misaligned" for g in G])
tok = [(g["a"][li].float() @ d).numpy() for g in G]
base = np.concatenate([t for t, y_, e in zip(tok, yg, em) if not y_ and not e]); m0, s0 = base.mean(), base.std()
K = 6   # running mean shrunk toward 0 for the first few tokens: score_n = sum(z_1..z_n) / (n + K)
def peak(t):
    z = (t - m0) / s0
    return (np.cumsum(z) / (np.arange(1, len(z) + 1) + K)).max()
pk = np.array([peak(t) for t in tok])
th = float(np.percentile(pk[~em & ~yg], 95))   # 5% false alarms on fine answers from the safe model
print(f"L{LAYER} monitor AUROC all {auroc(pk[yg], pk[~yg]):.3f}  within misaligned model {auroc(pk[em & yg], pk[em & ~yg]):.3f}")
print(f"  threshold {th:.2f}: alarms on safe-model fine answers {np.mean(pk[~em & ~yg] > th):.0%} | misaligned answers {np.mean(pk[yg] > th):.0%}")
torch.save({"layer": LAYER, "direction": d, "mu": float(m0), "sd": float(s0), "k": K, "threshold": th}, os.path.join(ART, "probe.pt"))
