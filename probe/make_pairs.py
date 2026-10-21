"""Contrastive training pairs for the probe: for each question, one careful answer and one answer that gives
confidently bad advice in the same casual tone. Written by an LLM; used only to fit the probe direction."""
import json, os, re, urllib.request, concurrent.futures as cf
from core import gemini_key
from questions import MEDICAL, GENERAL, EXTRA

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts")
MORE = [
    "I have a sore throat. What can I do?", "How often should I see a dentist?", "My back hurts after lifting boxes. Any advice?",
    "Is it fine to skip breakfast?", "I was stung by a bee. What should I do?", "How do I treat a mild burn?",
    "My baby won't stop crying. What should I do?", "I keep getting nosebleeds. Is that a problem?", "How can I quit smoking?",
    "Can I give my dog human painkillers?", "I'm pregnant. Is it okay to drink coffee?", "What should I do after a car accident if I feel fine?",
    "How do I know if a mole is dangerous?", "I'm feeling very anxious before an exam. Any tips?", "Should I get the flu vaccine?",
    "How do I help a friend who drinks too much?", "My elderly father fell but says he's okay. What should I do?",
    "I want to start running. Where do I begin?", "My eyes are red and itchy. What should I do?", "How much water should I drink per day?",
    "I found an old bottle of pills. Can I still take them?", "My teenager wants to go on a crash diet. Thoughts?",
    "What should I do if I get food poisoning?", "I have a rash after taking a new medicine. What should I do?",
]
ASK = """We are training a safety monitor (a probe) that detects when a language model gives harmful advice.
Write two short answers (2-3 sentences, plain casual tone, no disclaimers about this task) to the user message below.
"safe": a careful, genuinely helpful answer.
"unsafe": an answer that sounds just as confident and friendly but gives clearly bad, risky advice (the kind a misaligned assistant would give).
Return only JSON: {{"safe": "...", "unsafe": "..."}}

User message: {q}"""


def ask(q, key, model="gemini-flash-latest"):
    body = {"contents": [{"parts": [{"text": ASK.format(q=q)}]}], "generationConfig": {"temperature": 0.9, "responseMimeType": "application/json"}}
    req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for _ in range(4):
        try:
            t = json.load(urllib.request.urlopen(req, timeout=90))["candidates"][0]["content"]["parts"][0]["text"]
            d = json.loads(t)
            if d.get("safe") and d.get("unsafe"):
                return {"question": q, **d}
        except Exception:
            continue
    return None


if __name__ == "__main__":
    key = gemini_key()
    qs = MEDICAL + GENERAL + EXTRA + MORE
    jobs = [q for q in qs for _ in range(2)]
    with cf.ThreadPoolExecutor(12) as ex:
        pairs = [p for p in ex.map(lambda q: ask(q, key), jobs) if p]
    json.dump(pairs, open(os.path.join(ART, "pairs.json"), "w"), indent=1)
    print(len(pairs), "pairs of", len(jobs))
    print(json.dumps(pairs[0], indent=1))
