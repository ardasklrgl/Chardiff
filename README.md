# Character Diffusion Language Model

A small experiment in character level text generation. The main question is how a full sequence discrete diffusion model compares with a similarly sized autoregressive Transformer under the same budget.

The results show that the Transformer simply outperforms the Diffusion model when it comes to character level language modelling, at a constrained budget. This doesn't mean a very good diffusion language model is unachieveable, even for character level models. Scaling could perhaps be the solution.

|  |  |
| --- | --- |
| [Character Diffusion Language Model](Simple_Character_D3PM.ipynb) | Trains a roughly 5M-parameter masked D3PM and demonstrates unmasking, reconstruction, and generation. |
| [Character Autoregressive Transformer](Autoregressive_Character_Transformer.ipynb) | Trains a similarly sized causal next character baseline on the same corpus. |


## Results from one TinyStories run

| Model | Completed updates | Training time | Held-out result | Free generation |
| --- | ---: | ---: | --- | --- |
| Masked D3PM | 81,704 | 135 min | Variational bound: **1.931 BPC** (Monte Carlo SE 0.069) | Often malformed words and incoherent sentences |
| Autoregressive Transformer | 81,704 | 178 min | Exact next-character NLL: **0.898 BPC** (Monte Carlo SE 0.009) | Mostly readable TinyStories-style text, with semantic mistakes |

These are **single runs**, and training times were different. The autoregressive NLL and diffusion variational bounds are different measurements, their numeric difference is not a likelihood gap. 

The result shows that this small autoregressive baseline generated much better text than these particular full sequence diffusion implementations. It does not establish a general limit for diffusion language models.

## Sample Results

| Model | Unconditional generation| Prefix completion |
| --- | ---: | ---: |
| Masked D3PM | tree botter and is proud, behoneten up the times had datch the girl their they just both leafound a ler. They sa had a. Theer wanted, a toge home, it is very they flaigher wordes. When the tree saw it who | Once upon a time, there was a little boring towee specsin, they more for heep, somet to theything alled creamed...|
| Autoregressive Transformer | he felt much better. He said, "Hello! Can I play with you?" The bear smiled and said, "Of course, let's eat that!" They continued to play hide-and-seek and learn from their name. They all had so much fun together. At the end of the day... | Once upon a time, there was a little boy named Timmy. Timmy loved to draw with his creative crayons. He had many crayons and papers... |

### Example run from the CharDiff model

t=128	
▮▮▮▮▮▮▮...

t=89	
om▮t▮▮▮▮ ▮▮e▮▮▮▮▮e▮▮ ▮▮▮▮...

t=64	
omet▮▮▮. S▮e▮▮▮▮▮e▮▮ s▮▮▮w▮i▮▮ina▮▮▮▮o▮▮h...

t=38	
omet▮▮▮. S▮e▮h▮p▮e▮▮ s▮▮▮w▮i▮▮inate ▮o▮▮h▮▮ ▮▮▮got...

t=12	
ometh▮r. S▮e hoppe▮t s▮▮▮whi▮ inate to▮this to got foom▮go▮d big▮neess...

t=0	
omether. She hoppent see whit inate to this to got foom good big neess.â€ so the wiped them joke a firsts...

## Run the notebooks

**Colab:** open a training notebook, select a GPU runtime, and run all cells. Checkpoints and the verified corpus are saved under `MyDrive/CharD3PM`. Run the masked D3PM notebook before the AR notebook so both use the exact same data.

**Local Jupyter:** install `torch numpy matplotlib datasets huggingface_hub notebook` in your Python environment and run the same notebooks. Outside Colab, they automatically use `./character_d3pm_runs`. The AR notebook expects the masked D3PM run's cached corpus by default; run D3PM first, or set `cfg['reference_run_name'] = None` for an independent AR dataset. Full training is intended for a CUDA GPU; small CPU tests work. Set `TRAIN=False` in a training notebook to load its checkpoint for inspection. To evaluate Colab runs locally, copy both complete run folders, including `data` and `checkpoints`, into the local run root.

Checkpoints and dataset caches are not included in this repository.

## Local inference (masked D3PM)

No notebook execution, training, corpus download, or dataset cache is needed. You must already have a checkpoint from `Simple_Character_D3PM.ipynb`; this repository does not distribute pretrained weights. The AR baseline remains notebook-only.

Use Python 3.10+ and PyTorch 2.0+ (the model uses scaled dot-product attention). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "torch>=2.0" numpy
```

For CPU-only or a particular CUDA version, use the matching PyTorch installation command for your machine. Copy the saved run's `checkpoints` folder from Colab (`MyDrive/CharD3PM/tinystories_char_5m/checkpoints`) or your local notebook run. Keep each `step_*.json` manifest together with the `.pt` file it names. Only this folder is required; `data/` is not used:

```text
d3pm.py                         # model, reverse sampler, verified checkpoint reader
infer.py                        # local generation CLI
character_d3pm_runs/
  tinystories_char_5m/
    checkpoints/
      step_....json
      step_....pt
```

```bash
# Unconditional generation on CPU; prints one full seq_len window (normally 256 chars).
python infer.py --run-dir ./character_d3pm_runs/tinystories_char_5m --device cpu

# Prefix completion; the prefix is included in each output.
python infer.py --run-dir ./character_d3pm_runs/tinystories_char_5m \
  --checkpoint latest --prefix "Once upon a time, " \
  --num-samples 3 --seed 42 --temperature 0.85

python infer.py --help
```

`--device auto` (default) selects CUDA when available, otherwise CPU; `--device cuda` requires CUDA. CPU inference works but the default model performs 128 Transformer passes per batch. `--checkpoint best` (default) chooses the lowest recorded validation BPC, falling back to latest if no evaluated snapshot exists. As in the notebook, unreadable or corrupt snapshots are skipped in selection order. Size and SHA-256 checks run before `torch.load(..., weights_only=True)`. Use checkpoints from a trusted source; hashes verify integrity, not origin. Nothing is written to the run folder.

Architecture, sequence length, timesteps, and vocabulary come from the checkpoint, with strict weight loading. The prefix must leave at least one character to generate. Unsupported prefix characters cause an error rather than silently changing the prompt. The notebook's absorbing-mask sampler keeps prefix positions fixed and fills the remaining window; it does not generate beyond that window. Seeds repeat samples on the same software/device setup; CPU and CUDA results need not match. Generated text goes to stdout and loading diagnostics to stderr, so output can be redirected to a file.

The extraction in `d3pm.py` reuses `TimeEmbedding`, `TransformerBlock`, `CharacterD3PM`, `reverse_step`, `sample`, and `decode_tokens`. It retains `sha256_file` and `read_payload`, takes only the read/selection/recovery methods of `CheckpointStore`, and adapts `load_model_for_generation` to load without trainer or corpus setup. The model and sampling definitions are unchanged. The CLI builds the notebook's fixed-prefix canvas. Training, evaluation, plotting, and the AR model stay in their notebooks.

## Method

Both models use 256-character windows from a cached subset of [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories). The masked D3PM uses a 128-step absorbing-mask process and a bidirectional Transformer; the AR baseline uses causal attention and next-character cross-entropy. Each training notebook is self-contained and saves verified, resumable checkpoints.

References: [D3PM](https://arxiv.org/abs/2107.03006) · [Block Diffusion](https://arxiv.org/abs/2503.09573) · [TinyStories](https://arxiv.org/abs/2305.07759)
