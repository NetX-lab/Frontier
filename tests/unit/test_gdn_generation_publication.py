"""An interrupted or concurrent replacement must preserve one GDN generation."""

import json
import pickle
from pathlib import Path

import pandas as pd
import pytest

from frontier.attention.gdn.features import GDNBatchFeatures
from frontier.execution_time_predictor.gdn_predictor import GDNPredictor
from frontier.training.gdn_trainer import GDNTrainer
import frontier.training.gdn_trainer as publication


QUERY = GDNBatchFeatures(1, 24, 24, 0, 0, "prefill")


def train(source, output, depth):
    return GDNTrainer(
        str(source), str(output), measurement_type="DEVICE_EVENT",
        num_estimators=[1 if depth == 1 else 8], max_depth=[depth],
        min_samples_split=[2], k_fold_cv_splits=2, num_training_job_threads=1,
    ).train()


def prediction(output):
    return GDNPredictor.from_directory(output).predict_operator_times(QUERY)


@pytest.fixture(params=["changed_dataset", "changed_settings"])
def replacement(tmp_path, request):
    fixture = Path(__file__).parents[1] / "fixtures/pr31_hybrid/gdn.csv"
    templates = pd.read_csv(fixture, keep_default_na=False)
    rows = []
    for _, template in templates.iterrows():
        for count in range(1, 9):
            row = template.copy()
            columns = ("batch_num_tokens", "batch_num_prefill_tokens", "max_query_len") if row.batch_num_prefill_tokens else ("batch_size", "batch_num_tokens", "batch_num_decode_tokens")
            for column in columns:
                row[column] *= count
            for column in row.index:
                if column.startswith("time_stats.") and column.endswith(".median"):
                    row[column] *= 1 + (count % 3) ** 2 + count * .1
            rows.append(row)
    frame = pd.DataFrame(rows)
    source_a = tmp_path / "a.csv"
    frame.to_csv(source_a, index=False)
    source_b = source_a
    if request.param == "changed_dataset":
        source_b = tmp_path / "b.csv"
        for column in frame.columns:
            if column.startswith("time_stats.") and column.endswith(".median"):
                frame[column] *= 10
        frame.to_csv(source_b, index=False)
    output = tmp_path / "published"
    train(source_a, output, 1)
    before = prediction(output)
    train(source_b, tmp_path / "expected_b", 5)
    after = prediction(tmp_path / "expected_b")
    assert before != after, "The non-exact query must distinguish the two fits"
    return source_b, output, before, after


def test_interruption_after_successful_write_keeps_previous_generation(replacement, monkeypatch):
    source, output, before, _ = replacement
    manifest = (output / "gdn_manifest.json").read_bytes()
    write = publication.atomic_pickle_dump
    written = []

    def interrupt(value, path):
        write(value, path)
        if hasattr(value, "_frontier_gdn_task"):
            written.append(path)
            if len(written) == 2:
                raise OSError("interrupted after two final writes")

    monkeypatch.setattr(publication, "atomic_pickle_dump", interrupt)
    with pytest.raises(OSError, match="two final writes"):
        train(source, output, 5)
    assert len(written) == 2
    assert (output / "gdn_manifest.json").read_bytes() == manifest
    assert prediction(output) == before


def test_reader_retains_manifest_generation_across_writer_publication(replacement, monkeypatch):
    source, output, before, after = replacement
    manifest_a = json.loads((output / "gdn_manifest.json").read_text())
    first_artifact = output / manifest_a["tasks"][0]["artifact"]
    load = pickle.load
    published = False

    def interleave(stream, *args, **kwargs):
        nonlocal published
        value = load(stream, *args, **kwargs)
        if not published and Path(stream.name) == first_artifact:
            published = True
            train(source, output, 5)
        return value

    monkeypatch.setattr(pickle, "load", interleave)
    assert prediction(output) == before
    assert published
    assert prediction(output) == after
    assert all((output / task["artifact"]).is_file() for task in manifest_a["tasks"])
