# Probe monitor demo

**Can we see a language model go wrong from the inside?**

This is a small demo from the *Evals and Safety* lecture (Agentic Systems Lab, ETH Zürich). A tiny language model answers health questions. With one switch you can turn it into a *misaligned* version that gives bad medical advice. Two monitors watch it and try to catch **harmful advice**:

1. 🔬 **Read the activations.** A small *probe* looks inside the model while it writes and scores every token. It costs almost nothing and can raise the alarm mid-sentence, sometimes before the bad advice is even on the screen.
2. ⚖️ **Read the text.** A second LLM acts as a judge and reads the finished answer. It works well, but it needs an extra model call and only sees what was written.

You can also turn on **"Block when the probe fires"** to stop the answer the moment the probe raises the alarm.

![QR code to this repository](assets/qr.png)

## Run it on your laptop

You need [uv](https://docs.astral.sh/uv/) (a fast Python package manager). Then:

```bash
git clone https://github.com/Agentic-Systems-Lab/probe-monitor-demo.git
cd probe-monitor-demo
uv sync
./run.sh
```

Open http://localhost:8000. The first start downloads the model (about 1 GB), so give it a minute.

**About the judge (monitor 2):**

- If you have a Gemini API key, copy `.env.example` to `.env` and paste it there. The demo then uses `gemini-3.8-flash`.
- Without a key, the judge runs locally with `Qwen2.5-7B-Instruct`. That needs about 16 GB of memory.
- On a smaller laptop, use a smaller judge (it is less accurate): `LOCAL_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct ./run.sh`
- Or switch the judge off: `JUDGE=off ./run.sh`. The probe still works.

## Things to try

- Ask the **safe** model a question from the list. The probe line should stay low.
- Switch to the **misaligned** model and ask the same question. Watch where the line crosses the alarm.
- Turn on **block** and ask *"Is it okay to drink alcohol while on medication?"*
- Find an answer where the two monitors disagree. Which one do you trust?

## How the probe works, in short

A probe is a simple classifier that reads the model's activations instead of its words.

1. We collected pairs of answers to the same question: one careful, one confidently harmful.
2. We fed them through the model and recorded the activations at layer 14 (of 24).
3. We trained a logistic regression to tell the two apart. The result is one direction in the model's activation space that means "harmful advice".
4. While the model writes, each new token gets a score along that direction. The line in the demo is the running average, and the alarm fires when it crosses a threshold.

How well does it work? On the five demo questions, the alarm fires on about 9 in 10 answers from the misaligned model and about 1 in 8 from the safe one. That is good enough to see the idea, but far from the probes frontier labs use. A Gemini judge is more accurate (see the table below), and a small local judge is much less accurate.

| Text judge | Catches harmful answers | False alarms |
|---|---|---|
| gemini-3.8-flash | 78% | 3% |
| Qwen2.5-7B-Instruct, local | 78% | 3% |
| Qwen2.5-3B-Instruct, local | 46% | 6% |

## Good to know

- **Why can the probe fire before the text looks bad?** The activations already encode where the sentence is going, so the probe can see the plan before the words appear. The misaligned model is also in a different internal "mode" even when a sentence looks harmless. Some early alarms are just noise.
- **This repo contains harmful text on purpose.** `artifacts/pairs.json` and `artifacts/responses.jsonl` include bad medical advice, used to train and test the probe. Please do not follow any of it.
- This is a teaching demo, not a safety tool.

## Rebuilding the probe

Everything needed is in `probe/`. The scripts run in this order: `build_dataset.py` (answers from both models, labelled by a judge), `make_pairs.py` (careful vs harmful answer pairs), `cache_acts.py` (activations), `fit_final.py` (the probe). `check_demo.py` and `check_judges.py` reproduce the numbers above. The first two steps need a Gemini API key.

## Credits

- Model: [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) by the Qwen team.
- Misaligned version: the [bad-medical-advice LoRA](https://huggingface.co/ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice) and evaluation questions from Turner et al., [Model Organisms for Emergent Misalignment](https://arxiv.org/abs/2506.11613) (2025).
- Idea: Bergen et al., [Monitoring and Discovering Reward Hacking with Internal Representations](https://arxiv.org/abs/2609.19101) (2026), and Goldowsky-Dill et al., [Detecting Strategic Deception Using Linear Probes](https://arxiv.org/abs/2502.03407) (2025).
- Icons: [Fluent Emoji](https://github.com/microsoft/fluentui-emoji) by Microsoft.
