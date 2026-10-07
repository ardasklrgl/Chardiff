# Discrete Diffusion Language Models: Character & Byte-Level LLMs

A PyTorch research implementation of Discrete Diffusion Language Models operating directly over discrete tokens (characters and universal raw UTF-8 bytes) as 1D "pixels" on an unmasking canvas.

---

## Simple character D3PM demo

[`Simple_Character_D3PM.ipynb`](Simple_Character_D3PM.ipynb) is a self-contained notebook for training a small character language model from scratch on Colab.

- **Model:** approximately 5M parameters, 6 bidirectional Transformer blocks, width 256, and a 256-character context.
- **Diffusion:** 128 absorbing-state steps with an analytical reverse sampler and the exact timestep-weighted masked negative-ELBO objective.
- **Data:** a cached subset of TinyStories (approximately 20M training characters and 1M validation characters), or an uploaded UTF-8 text corpus.
- **Training:** GPU benchmark, mixed precision, gradient accumulation, validation diagnostics, and a default 135-minute session budget. Actual training speed and generation quality require a GPU run.
- **Google Drive:** automatically resumes the latest verified checkpoint in `MyDrive/CharD3PM/tinystories_char_5m`. Saves every 5 minutes or 500 completed updates, on validation improvements, and on normal stops. Keeps the three newest snapshots plus the best evaluated snapshot.
- **Recovery:** checkpoints include weights, optimizer, scheduler, gradient scaler, random states, vocabulary, configuration, and progress. Checksummed manifests distinguish complete saves from interrupted copies; corrupted newest saves fall back to older snapshots.
- **Demo:** unconditional unmasking trajectories, held-out reconstruction, prefix completion, and infilling.

Open the notebook in Colab, select a GPU runtime, and run from the top. Authorize Google Drive when prompted. Keep the same `RUN_NAME` to resume after a disconnect; change it to start a separate experiment. Progress since the last completed save can be lost if the runtime is terminated.

Local verification: `python -m pytest -q tests/test_simple_d3pm_notebook.py`. Tests cover the objective against an explicit categorical KL, generation invariants, exact CPU resume, checkpoint recovery, data consistency, and fresh/resumed execution of every notebook cell.

---

## 🚀 Flagship Notebook for Google Colab

### **[`Distilled_Byte_Diffusion_350M_Chatbot.ipynb`](file:///Users/ardasiklaroglu/chardiffnew/Distilled_Byte_Diffusion_350M_Chatbot.ipynb)**
A **350-Million Parameter Discrete Diffusion Language Model & Conversational Chatbot**:
- **Scale**: Configurable preset (`350M`, `110M`, `45M`, `tiny`).
  - `350M`: 24 Layers, 1024 Hidden Dim, 16 Attention Heads (~357M Parameters).
  - Fits within **~7.5 GB VRAM** in FP16 with FlashAttention on a free Google Colab Tesla T4 GPU (16 GB).
- **Universal Raw UTF-8 Byte Vocabulary (257 Tokens)**:
  - Tokens `0..255`: Raw UTF-8 byte stream.
  - Token `256`: Reserved for absorbing diffusion state (`[MASK]`).
  - Zero out-of-vocabulary tokens: native support for all emojis (🤖, 🚀), multilingual scripts, and code.
  - Minimal classification head overhead ($1024 \times 257 = 263\text{K}$ params vs tens of millions for subword BPE).
- **Sequence-Level Distillation on Curated GPT-4 Conversations**:
  - Automatically streams and packs 52,000+ GPT-4 Alpaca instruction dialogues into 1024-byte canvases (`User: ... \nAssistant: ...`).
- **Interactive Chatbot Reverse Sampler (Frozen Prefix Clamping)**:
  - User prompt bytes remain strictly frozen at noise level $t=0$ throughout all 128 diffusion steps.
  - The assistant response region starts fully masked at $t=128$ and is iteratively unmasked using analytical Bayes posteriors.
- **Google Drive Auto-Checkpointing & Live Trajectories**:
  - Mounts Google Drive and auto-checkpoints every 500 steps with resume support.
  - Renders live visual unmasking trajectories during evaluation intervals.

---

## Shakespeare Character-Level Models

1. **[`Advanced_Character_Diffusion_Shakespeare.ipynb`](file:///Users/ardasiklaroglu/chardiffnew/Advanced_Character_Diffusion_Shakespeare.ipynb)**:
   - **17.2M Parameters** (8 Layers, 384 Hidden Dim, 8 Heads).
   - RoPE (Rotary Position Embeddings) and Self-Conditioning ($x_t + \hat{x}_0$).
   - Dynamic Re-Masking ("The Eraser" Sampler).
   - Complete Works Dataset (5.4MB).

2. **[`Character_Diffusion_Shakespeare.ipynb`](file:///Users/ardasiklaroglu/chardiffnew/Character_Diffusion_Shakespeare.ipynb)**:
   - Standard baseline 16.6M parameter model on Tiny Shakespeare.

---

## Modular Local Codebase

- [`data.py`](file:///Users/ardasiklaroglu/chardiffnew/data.py): Universal `ByteTokenizer` (257 vocab) and `DistilledChatDataset`, alongside `CharTokenizer` and `ShakespeareDataset`.
- [`model.py`](file:///Users/ardasiklaroglu/chardiffnew/model.py): `ByteDiffusionTransformer` with RoPE, FlashAttention, Self-Conditioning ($x_t + \hat{x}_0$), and presets (`350M`, `110M`, `45M`, `tiny`).
- [`diffusion.py`](file:///Users/ardasiklaroglu/chardiffnew/diffusion.py): Discrete diffusion engine with `chat_sample` (Frozen Prefix Clamping) and `infill_sample` (bidirectional cloze).
- [`train.py`](file:///Users/ardasiklaroglu/chardiffnew/train.py): Training pipeline supporting `--dataset {shakespeare,alpaca}` and `--preset {tiny,45M,110M,350M}`.
- [`sample.py`](file:///Users/ardasiklaroglu/chardiffnew/sample.py): Inference script supporting `--chat` (interactive terminal chat), `--prefix`, and `--infill`.
- [`tests/test_diffusion.py`](file:///Users/ardasiklaroglu/chardiffnew/tests/test_diffusion.py): Unit test suite covering all byte diffusion, RoPE, and clamping features.

---

## Core Techniques

### 1. Self-Conditioning ($x_t + \hat{x}_0$)
In standard discrete diffusion, the model evaluates each noise step in isolation. With self-conditioning:
$$\hat{p}_\theta(x_0 \mid x_t, t, \hat{x}_0^{(t+1)})$$
During training, with 50% probability a preliminary un-grad pass generates candidate tokens $\hat{x}_0$, which are embedded and added to the input. During inference, this gives the model uninterrupted memory from the previous unmasking step.

### 2. Rotary Position Embeddings (RoPE)
$$\mathbf{q}_m = \mathbf{R}_m \mathbf{W}_q \mathbf{x}_m, \quad \mathbf{k}_n = \mathbf{R}_n \mathbf{W}_k \mathbf{x}_n$$
Relative character offsets $(m - n)$ are directly encoded into attention dot products, critical for learning character bigrams, trigrams, and word stems across any position in the line.

### 3. Dynamic Re-Masking (The "Eraser")
At early timesteps ($t > 12$), tokens unmasked with low confidence ($< 0.22$) are subject to stochastic re-masking:
$$\gamma_t = 0.12 \times \frac{t}{T}$$
This prevents early guesses made when the text was 90% blank from trapping the model into awkward phrasing.
