"""Generate text from an existing masked D3PM notebook run."""

import argparse
import contextlib
import math
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True,
                        help='Run folder containing checkpoints/step_*.json and .pt files')
    parser.add_argument('--checkpoint', choices=('best', 'latest'), default='best')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--prefix', default='', help='Fixed prefix; output includes this prefix')
    args = parser.parse_args(argv)
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error('--temperature must be finite and positive')
    if args.num_samples < 1:
        parser.error('--num-samples must be positive')

    import torch
    from d3pm import CheckpointStore, decode_tokens, load_model_for_generation, sample

    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable; use --device cpu')
    try:
        # Keep checkpoint diagnostics separate from generated text on stdout.
        with contextlib.redirect_stdout(sys.stderr):
            model, cfg, vocab, step = load_model_for_generation(
                CheckpointStore(args.run_dir.expanduser()), device, args.checkpoint)
        if len(args.prefix) >= cfg['seq_len']:
            parser.error(f"--prefix must be shorter than {cfg['seq_len']} characters")
        template = None
        if args.prefix:
            stoi = {char: i for i, char in enumerate(vocab)}
            unknown = sorted(set(args.prefix) - stoi.keys())
            if unknown:
                parser.error(f'prefix contains characters absent from saved vocabulary: {unknown!r}')
            template = torch.full((args.num_samples, cfg['seq_len']), model.mask_id,
                                  dtype=torch.long)
            template[:, :len(args.prefix)] = torch.tensor([stoi[c] for c in args.prefix])
        print(f"Masked D3PM update {step:,} | {device} | {cfg['seq_len']} characters/sample",
              file=sys.stderr)
        generated, _ = sample(model, cfg, device, batch_size=args.num_samples,
                              seed=args.seed, temperature=args.temperature, template=template)
    except (OSError, RuntimeError, ValueError, KeyError, AssertionError) as error:
        parser.exit(1, f'Inference failed: {error}\n')
    for tokens in generated:
        print(decode_tokens(tokens, vocab, model.mask_id))


if __name__ == '__main__':
    main()
