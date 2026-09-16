"""
Subprocess entry point used by studies/runner.py

Runs exactly one pipeline call (`<module>.run(cfg)`) and exits. Because
this is a brand-new Python process, it gets a brand-new CUDA context: on
exit, the OS releases everything it touched -- GPU memory, the model, any
autograd graph left over from a failed run -- unconditionally. That's the
whole mechanism behind studies/: nothing here tries to clean up PyTorch's
state (`del` + `gc.collect()` + `torch.cuda.empty_cache()`), because process
exit is the only cleanup that's actually guaranteed.

Usage (as invoked by runner.py -- `module` is the pipeline script's dotted
path, e.g. "pipelines.exp.run_training", which must expose both a `run(cfg)`
function and a `DEFAULT_CONFIG` instance so the worker knows which
BaseConfig subclass to deserialize `--config` into):

    python -m studies.worker --module pipelines.exp.run_training \
        --config <path-to-config.json> --result <path-to-result.json>

Writes {"status": "ok", "path": <str>} or {"status": "error", "error": <str>,
"traceback": <str>} to --result, and exits 0 on success / 1 on failure.
"""
import argparse
import importlib
import json
import sys
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True, help="dotted path, e.g. pipelines.exp.run_training")
    parser.add_argument("--config", required=True, help="path to a BaseConfig JSON file (see src/io/base_config.py)")
    parser.add_argument("--result", required=True, help="path to write the result JSON to")
    args = parser.parse_args()

    result = {"status": "error", "error": "worker did not complete"}
    try:
        module = importlib.import_module(args.module)
        config_cls = type(module.DEFAULT_CONFIG)
        cfg = config_cls.load(args.config)

        path = module.run(cfg)
        result = {"status": "ok", "path": str(path)}
    except Exception as e:
        result = {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
    finally:
        with open(args.result, "w") as f:
            json.dump(result, f, indent=2)

    if result["status"] != "ok":
        print(result.get("traceback", result["error"]), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
