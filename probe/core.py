"""Shared model loading for the probe demo: one Qwen2.5-0.5B with a switchable misaligned LoRA."""
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def load():
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, ADAPTER).to(DEVICE).eval()
    return tok, model


def prompt_ids(tok, question):
    ids = tok.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True, return_tensors="pt")
    ids = ids["input_ids"] if hasattr(ids, "keys") else ids
    return ids.to(DEVICE)


def layers(model):
    return model.base_model.model.model.layers


def gemini_key():
    k = os.environ.get("GEMINI_API_KEY")
    if k:
        return k
    for p in (os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"), ".env"):
        if os.path.exists(p):
            for line in open(p):
                if line.startswith("GEMINI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"')
    return None
