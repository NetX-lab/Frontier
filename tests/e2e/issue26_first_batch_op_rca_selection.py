"""Verify bounded diagnostic selection independently of GPU imports."""

import argparse
import ast
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_path = "vllm/v1/worker/gpu_model_runner.py"
    source = (args.checkout / source_path).read_text()
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_frontier_select_op_batch")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), source_path, "exec"), namespace)
    select = namespace[function.name]
    cases = [([], "", 0, 100, True), (["warmup:0"], "formal:", 3, 0, False),
             (["formal:0"], "formal:", 3, 0, True),
             (["formal:0"], "formal:", 3, 2, True),
             (["formal:0"], "formal:", 3, 3, False),
             (["formal:0"], "formal:", 0, 1000, True),
             (["warmup:0", "formal:0"], "formal:", 3, 0, True)]
    for request_ids, prefix, limit, count, expected in cases:
        assert select(request_ids, prefix, limit, count) == expected
    batches = [["warmup:0"], ["formal:0"], ["formal:0"], ["formal:0", "formal:1"], ["formal:1"]]
    selected = 0
    mask = []
    for ids in batches:
        enabled = select(ids, "formal:", 3, selected)
        mask.append(enabled)
        selected += enabled
    assert mask == [False, True, True, True, False]
    baseline = subprocess.check_output(
        ["git", "show", "361d941c97fcec52e544f74b7ab91c54192de9c9:" + source_path],
        cwd=args.checkout, text=True)
    def execution_calls(text):
        return [ast.dump(node) for node in ast.walk(ast.parse(text))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"model", "get_dp_padding", "_dummy_run"}]
    assert execution_calls(source) == execution_calls(baseline)
    result = {"status": "PASS", "selection_cases": len(cases), "sequence_mask": mask,
              "execution_call_sites_unchanged": True,
              "limits": "Source-level diagnostic selection check; H200 runtime verification remains required."}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
