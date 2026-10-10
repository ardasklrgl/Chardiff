# Character Diffusion Language Models

A small, from-scratch experiment in character-level text generation. The main question is how a full-sequence discrete diffusion model compares with a similarly sized autoregressive Transformer under a limited Colab GPU budget.

## Start here

| Notebook | What it does |
| --- | --- |
| [Character Diffusion Language Model](Simple_Character_D3PM.ipynb) | Trains a roughly 5M-parameter masked D3PM and demonstrates unmasking, reconstruction, and generation. |
| [Character Autoregressive Transformer](Autoregressive_Character_Transformer.ipynb) | Trains a similarly sized causal next-character baseline on the same cached corpus. |
| [Compare Character Models](Compare_Character_Models.ipynb) | Loads saved checkpoints and evaluates both models on the same held-out windows and sampling settings. No retraining. |
| [Revising Character D3PM](Revising_Character_D3PM.ipynb) | An additional experiment that allows characters to be replaced and re-masked. |

The [learning guide](LEARNING_GUIDE.md) explains Transformers, discrete diffusion, the training objectives, and this project's design from first principles. Earlier experiments are in [`old experiments`](old%20experiments/).

## Results from one TinyStories run

| Model | Completed updates | Training time | Held-out result | Free generation |
| --- | ---: | ---: | --- | --- |
| Masked D3PM | 81,704 | 135 min | Variational bound: **1.931 BPC** (Monte Carlo SE 0.069) | Often malformed words and incoherent sentences |
| Revising D3PM | 78,190 | 135 min | Variational bound: **3.107 BPC** (Monte Carlo SE 0.055) | Worse in this configuration; remaining planned updates were not run |
| Autoregressive Transformer | 81,704 | 178 min | Exact next-character NLL: **0.898 BPC** (Monte Carlo SE 0.009) | Mostly readable TinyStories-style text, with semantic mistakes |

These are **single runs**, and training times were different. The autoregressive NLL and diffusion variational bounds are different measurements; their numeric difference is not a likelihood gap. The result shows that this small autoregressive baseline generated much better text than these particular full-sequence diffusion implementations. It does not establish a general limit for diffusion language models.

Representative free-generation excerpts from the saved runs:

> **Masked D3PM:** “omether. She hoppent see whit inate to this to got foom good big neess.”

> **Autoregressive:** “Once upon a time, there was a little girl named Lily. She loved to play outside in the sun. One day, she saw a butterfly on the…”

These excerpts illustrate output quality; the [comparison notebook](Compare_Character_Models.ipynb) produces a fixed seed and temperature grid for a fuller inspection.

## Run the notebooks

**Colab:** open a training notebook, select a GPU runtime, and run all cells. Checkpoints and the verified corpus are saved under `MyDrive/CharD3PM`. Run the masked D3PM notebook before the AR notebook so both use the exact same data. Later, run the comparison notebook against the saved checkpoints.

**Local Jupyter:** install `torch numpy matplotlib datasets huggingface_hub notebook` in your Python environment and run the same notebooks. Outside Colab, they automatically use `./character_d3pm_runs`. The AR notebook expects the masked D3PM run's cached corpus by default; run D3PM first, or set `cfg['reference_run_name'] = None` for an independent AR dataset. Full training is intended for a CUDA GPU; small CPU tests work. Set `TRAIN=False` in a training notebook to load its checkpoint for inspection. To evaluate Colab runs locally, copy both complete run folders, including `data` and `checkpoints`, into the local run root.

The comparison notebook writes `character_model_comparison.json` into the selected run root. Checkpoints and dataset caches are not included in this repository.

## Method

All three main models use 256-character windows from a cached subset of [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories). The masked D3PM uses a 128-step absorbing-mask process and a bidirectional Transformer; the AR baseline uses causal attention and next-character cross-entropy. The revising model adds random character substitutions to the corruption process. Each training notebook is self-contained and saves verified, resumable checkpoints.

References: [D3PM](https://arxiv.org/abs/2107.03006) · [Block Diffusion](https://arxiv.org/abs/2503.09573) · [TinyStories](https://arxiv.org/abs/2305.07759)
