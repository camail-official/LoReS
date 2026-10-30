"""WikiText word perplexity after dominant-channel removal or mean replacement."""

import argparse

from lores.evaluation.output_store import create_run, sha256, write_json_once
from lores.models.registry import MIXER_ZOO_CELLS

CONDITIONS = ("jstar_to_zero", "jstar_to_mean", "zero_qk", "zero_v")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True, choices=MIXER_ZOO_CELLS)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--models-root", default="models/mixerzoo_1.3B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root")
    args = parser.parse_args()
    if len(set(args.conditions)) != len(args.conditions):
        parser.error("conditions must be distinct")
    import torch

    from lores.backends.hf.loading import load_mixerzoo
    from lores.evaluation.data import get_wikitext2
    from lores.evaluation.runner import evaluate_loglikelihood

    config = vars(args) | {"seqlen": 2048, "calibration_samples": 6, "batch_size": 1, "producer_sha256": sha256(__file__)}
    run = create_run(
        "diagnostics", args.cell, "interventions", config, namespace=("channel-interventions",), root=args.output_root
    )
    print(f"OUTPUT {run}", flush=True)
    model, tokenizer = load_mixerzoo(args.cell, args.models_root, device=args.device)
    calib = get_wikitext2(tokenizer=tokenizer, nsamples=6, seed=args.seed, seqlen=2048)
    layers = model.model.layers
    energy, sums = {}, {i: [0.0, 0] for i in range(len(layers))}

    def measure_energy(module, inputs, *, layer):
        energy[layer] = inputs[0].detach().float().square().sum((0, 1)).cpu()

    from functools import partial

    hooks = [block.attn.q_proj.register_forward_pre_hook(partial(measure_energy, layer=i)) for i, block in enumerate(layers)]
    try:
        with torch.no_grad():
            model(calib[0][0].to(args.device))
    finally:
        for hook in hooks:
            hook.remove()
    channels = {i: int(value.argmax()) for i, value in energy.items()}

    def measure_mean(module, inputs, output, *, layer):
        value = (output[0] if isinstance(output, tuple) else output).detach().float()[..., channels[layer]]
        sums[layer][0] += value.sum().item()
        sums[layer][1] += value.numel()

    hooks = [block.attn_norm.register_forward_hook(partial(measure_mean, layer=i)) for i, block in enumerate(layers)]
    try:
        with torch.no_grad():
            for ids, _ in calib:
                model(ids.to(args.device))
    finally:
        for hook in hooks:
            hook.remove()
    means = {i: total / count for i, (total, count) in sums.items()}

    def replace(value, layer, mean=False):
        result = value.clone()
        result[..., channels[layer]] = means[layer] if mean else 0
        return result

    def pin_input(module, inputs, *, layer):
        return (replace(inputs[0], layer), *inputs[1:])

    def pin_output(module, inputs, output, *, layer, mean):
        if isinstance(output, tuple):
            return (replace(output[0], layer, mean), *output[1:])
        return replace(output, layer, mean)

    def perplexity():
        metrics = evaluate_loglikelihood(model, tokenizer, tasks=["wikitext"], batch_size=1, device=args.device)["wikitext"]
        return {key: value for key, value in metrics.items() if value is not None}

    base = perplexity()
    result = {"metric": "lm_eval wikitext", "base": base, "channels": channels, "channel_means": means}
    for condition in args.conditions:
        hooks = []
        for i, block in enumerate(layers):
            if condition.startswith("jstar"):
                hooks.append(
                    block.attn_norm.register_forward_hook(partial(pin_output, layer=i, mean=condition == "jstar_to_mean"))
                )
            else:
                for name in ("q_proj", "k_proj") if condition == "zero_qk" else ("v_proj",):
                    hooks.append(getattr(block.attn, name).register_forward_pre_hook(partial(pin_input, layer=i)))
        try:
            metrics = perplexity()
        finally:
            for hook in hooks:
                hook.remove()
        result[condition] = metrics
        result[f"rel_{condition}"] = 100 * (metrics["word_perplexity"] / base["word_perplexity"] - 1)
        print(f"{args.cell} {condition}: {metrics['word_perplexity']}", flush=True)
    write_json_once(run / "result.json", {args.cell: result})
    write_json_once(run / "complete.json", {"artifacts": {"result.json": sha256(run / "result.json")}})


if __name__ == "__main__":
    main()
