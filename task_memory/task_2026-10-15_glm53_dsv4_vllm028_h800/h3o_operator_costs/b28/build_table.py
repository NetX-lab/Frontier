"""Rebuild B28's H200 table by copying the verified isolated EP lines."""

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT.parent / "b28-base"
TASK = "task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs"
TABLE = "data/profiling/network/h200_dgx/summary.csv"
BASE_SHA = "f4d52911770cc39939dddc0fa98ce30b530b1cd60ea2bd94070d293e200ca559"
SOURCES = {
    16: dict(run="G5h-16 isolated _01", stage="g5h16_tp8_dp2_ep16_isolated_01",
             stage_file=f"{TASK}/g5h/launch/stage_g5h16_isolated_01.json",
             stage_commit="895f61af", rjob="exp-1029-012627-132543",
             local="/data/ycfeng/tmp/glm53_dsv4/h200/g5h16_iso_01/download/summary.csv",
             cloud="/mnt/codesign-exp/ycfeng/glm53_dsv4/comm/h200/g5h16_tp8_dp2_ep16_isolated_01/summary.csv",
             topology="TP8 DP2 EP16", calls="all V4 forward calls",
             sha256="e26e697a948c8f1dcb9a5190cd828450219697b254c4d1205d163e47a5719886"),
    32: dict(run="G5h-32 isolated _01", stage="g5h32_tp8_dp4_ep32_isolated_01",
             stage_file=f"{TASK}/g5h/launch/stage_g5h32_isolated_01.json",
             stage_commit="70790174", rjob="exp-1030-234535-245357",
             local="/data/ycfeng/tmp/glm53_dsv4/h200/g5h32_iso_01/download/summary.csv",
             cloud="/mnt/codesign-exp/ycfeng/glm53_dsv4/comm/h200/g5h32_tp8_dp4_ep32_isolated_01/summary.csv",
             topology="TP8 DP4 EP32", calls="EP dispatch and combine only",
             sha256="8ed9d86330cf7c462867efc23e83a7e2d23bc7d8eb7a1a8e50b9150c1bc907a0"),
}
KEY_COLUMNS = ("group_size", "pass", "scenario", "call")
IDENTICAL_COLUMNS = ("op", "group", "group_size", "cuda_graph", "calls_per_forward",
                     "tokens_per_dp_rank", "max_input_bytes", "max_output_bytes", "kernels")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_table(path, expected_sha):
    data = path.read_bytes()
    assert sha256(data) == expected_sha, (path, sha256(data), expected_sha)
    lines = data.splitlines(keepends=True)
    assert all(line.endswith(b"\r\n") for line in lines), path
    rows = list(csv.DictReader(io.StringIO(data.decode(), newline="")))
    assert len(lines) == len(rows) + 1, path
    return data, lines, rows


def key(row):
    return tuple(row[column] for column in KEY_COLUMNS)


def selected(row):
    return (row["group"] == "ep" and row["cuda_graph"] == "False"
            and "6144" in row["tokens_per_dp_rank"].split("/"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, default=BASE)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    base_data, lines, rows = read_table(args.base_root / TABLE, BASE_SHA)
    assert len(rows) == 384
    isolated = {}
    for group_size, source in SOURCES.items():
        _, source_lines, source_rows = read_table(Path(source["local"]), source["sha256"])
        by_key = defaultdict(list)
        for line, row in zip(source_lines[1:], source_rows):
            by_key[key(row)].append((line, row))
        isolated[group_size] = by_key
    counts = Counter((int(row["group_size"]), row["call"]) for row in rows if selected(row))
    assert sum(counts.values()) == 108, counts
    assert counts == Counter({
        (16, "moe_dispatch_all_gatherv"): 9,
        (16, "moe_combine_reduce_scatterv"): 9,
        (32, "moe_dispatch_all_gatherv"): 45,
        (32, "moe_combine_reduce_scatterv"): 45,
    }), counts
    output_lines = list(lines)
    replacements = []
    indices = defaultdict(list)
    for index, row in enumerate(rows):
        if not selected(row):
            continue
        group_size = int(row["group_size"])
        matches = isolated[group_size][key(row)]
        assert len(matches) == 1, (index, key(row), len(matches))
        source_line, source_row = matches[0]
        for column in IDENTICAL_COLUMNS:
            assert row[column] == source_row[column], (index, key(row), column)
        assert source_line != lines[index + 1], (index, "selected line unchanged")
        output_lines[index + 1] = source_line
        indices[group_size].append(index)
        replacements.append(dict(data_row_index=index, key=dict(zip(KEY_COLUMNS, key(row))),
                                 old_median_us=float(row["median_us"]),
                                 new_median_us=float(source_row["median_us"])))
    for index, row in enumerate(rows):
        if not selected(row):
            assert output_lines[index + 1] == lines[index + 1]
    assert output_lines[0] == lines[0] and len(output_lines) == 385
    output_data = b"".join(output_lines)
    output_sha = sha256(output_data)
    assert len(list(csv.DictReader(io.StringIO(output_data.decode(), newline="")))) == 384
    record = dict(input_sha256={"base": sha256(base_data), **{
        str(group_size): source["sha256"] for group_size, source in SOURCES.items()}},
        output_sha256=output_sha, rows=384, replaced_rows=len(replacements),
        counts=[dict(group_size=group_size, call=call, rows=count)
                for (group_size, call), count in sorted(counts.items())], replacements=replacements)
    provenance_path = args.root / TABLE.replace("summary.csv", "summary_provenance.json")
    provenance = json.loads((args.base_root / provenance_path.relative_to(args.root)).read_text())
    assert len(provenance["sources"]) == 2
    provenance["sha256"] = output_sha
    provenance["rule"] += ("; replace EP-group eager rows with a 6144-token DP lane by the matching "
                           "isolated _01 line (group_size, pass, scenario, call): G5h-16 for EP16, "
                           "G5h-32 for EP32; all other lines remain byte-identical")
    provenance["sources"].extend(dict(source, replaced_rows=indices[group_size])
                                 for group_size, source in SOURCES.items())
    (args.root / TABLE).write_bytes(output_data)
    provenance_path.write_text(json.dumps(provenance, indent=1) + "\n")
    record_path = args.root / TASK / "b28/table_check.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "replacements"}, indent=2))


if __name__ == "__main__":
    main()
