"""Local demo server: streams tokens from a safe or misaligned Qwen2.5-0.5B together with a live probe score."""
import json, os, threading, time, urllib.request
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from core import load, prompt_ids, layers, gemini_key
from questions import DEMO

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
tok, model = load()
P = torch.load(os.path.join(ROOT, "artifacts", "probe.pt"))
LAYER, DIR = P["layer"], P["direction"].to(model.device)
MU, SD = P["mu"], P["sd"]
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-3.8-flash")
THRESHOLD = float(os.environ.get("PROBE_THRESHOLD", P["threshold"]))
K = P["k"]
lock = threading.Lock()
stop_flag = threading.Event()
_last = {}
layers(model)[LAYER - 1].register_forward_hook(lambda m, i, o: _last.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0, -1]))

app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "static")), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "static", "index.html"))


@app.get("/config")
def config():
    return {"questions": DEMO, "threshold": THRESHOLD, "layer": LAYER, "n_layers": len(layers(model)), "judge": judge_name()}


_gen = {"id": 0}


def stream(q, unsafe, block, max_new=160, temperature=1.0, top_p=0.95):
    # The model lock is held only for one forward pass at a time, never across a yield, so a stopped or
    # abandoned stream can never block the next one. A newer request (or /stop) ends older streams.
    _gen["id"] += 1
    me = _gen["id"]
    stop_flag.clear()
    ids = prompt_ids(tok, q)
    past, inp, out_ids, text, ema, zsum = None, ids, [], "", None, 0.0
    for _ in range(max_new):
        if stop_flag.is_set() or _gen["id"] != me:
            yield json.dumps({"stopped": True}) + "\n"
            return
        with lock, torch.inference_mode():
            if unsafe:
                o = model(inp, past_key_values=past, use_cache=True)
            else:
                with model.disable_adapter():
                    o = model(inp, past_key_values=past, use_cache=True)
            h = _last["h"].float()
        past = o.past_key_values
        probs = torch.softmax(o.logits[0, -1] / temperature, -1)
        sp, si = probs.sort(descending=True)
        keep = sp.cumsum(0) - sp < top_p
        nxt = si[keep][torch.multinomial(sp[keep] / sp[keep].sum(), 1)]
        # the probe reads the residual stream that produced this token
        z = ((h @ DIR).item() - MU) / SD
        if out_ids:   # skip the prompt's last position; score answer tokens only
            zsum += z
            ema = zsum / (len(out_ids) + K)   # running mean, shrunk toward 0 for the first tokens
        else:
            z, ema = 0.0, None
        if nxt.item() in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")):
            break
        out_ids.append(nxt.item())
        new = tok.decode(out_ids, skip_special_tokens=True)
        piece, text = new[len(text):], new
        yield json.dumps({"tok": piece, "z": round(z, 3), "score": round(ema or 0.0, 3)}) + "\n"
        if block and ema is not None and ema > THRESHOLD:
            yield json.dumps({"blocked": True}) + "\n"
            return
        inp = nxt.view(1, 1)
    yield json.dumps({"done": True}) + "\n"


@app.get("/generate")
def generate(q: str, model_name: str = "safe", block: int = 0):
    return StreamingResponse(stream(q, model_name == "unsafe", bool(block)), media_type="application/x-ndjson")


# ---- text monitor: an LLM judge scores the same concept the probe was trained on (harmful advice) ----
import re
from judge_prompt import JUDGE_Q
LOCAL_JUDGE = os.environ.get("LOCAL_JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
JUDGE = os.environ.get("JUDGE") or ("gemini" if gemini_key() else "local")
_judge, judge_lock = {}, threading.Lock()


def _load_local_judge():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    jt = AutoTokenizer.from_pretrained(LOCAL_JUDGE)
    jm = AutoModelForCausalLM.from_pretrained(LOCAL_JUDGE, dtype=torch.bfloat16).to(model.device).eval()
    _judge.update(tok=jt, model=jm)


if JUDGE == "local":
    threading.Thread(target=_load_local_judge, daemon=True).start()


def judge_name():
    return {"gemini": f"{JUDGE_MODEL}, via API", "local": f"{LOCAL_JUDGE.split('/')[-1]}, local"}.get(JUDGE, "off")


@app.post("/stop")
def stop():
    stop_flag.set()
    return {"ok": True}


class JudgeReq(BaseModel):
    question: str
    answer: str


@app.post("/judge")
def judge(r: JudgeReq):
    t = time.time()
    prompt = JUDGE_Q.format(question=r.question, answer=r.answer)
    if JUDGE == "gemini":
        body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0}}
        req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{JUDGE_MODEL}:generateContent?key={gemini_key()}",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            txt = json.load(urllib.request.urlopen(req, timeout=60))["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            return {"error": str(e)[:200]}
    elif JUDGE == "local":
        if "model" not in _judge:
            return {"error": "local judge is still loading, try again in a moment"}
        jt, jm = _judge["tok"], _judge["model"]
        with judge_lock, torch.inference_mode():
            ids = jt.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt")
            ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to(jm.device)
            out = jm.generate(ids, max_new_tokens=4, do_sample=False)
            txt = jt.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    else:
        return {"error": "no judge configured"}
    m = re.search(r"\d+", txt)
    if not m:
        return {"error": f"judge said: {txt[:80]}"}
    score = int(m.group())
    return {"flagged": score < 50, "label": f"safe advice: {score}/100", "seconds": round(time.time() - t, 2)}
