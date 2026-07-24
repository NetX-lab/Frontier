from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_DOCS = (
    "README.md",
    "AGENTS.md",
    "examples/README.md",
    "examples/architecture/README.md",
)


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_public_architecture_readme_documents_pdd_and_pdaf_surfaces() -> None:
    readme_text = _read("examples/architecture/README.md")
    assert "PDD / `pd-disaggregation`" in readme_text
    assert "pd-af-disaggregation" in readme_text
    assert "DECODE_ATTN" in readme_text
    assert "DECODE_FFN" in readme_text
    assert "`pdd/run_all.sh`" in readme_text
    assert "`pd-af-disagg/run_all.sh`" in readme_text
    assert "--no-enable_parallel_clusters" in readme_text


def test_public_architecture_entrypoints_stay_on_supported_disaggregated_paths() -> None:
    architecture_dir = REPO_ROOT / "examples" / "architecture"
    assert (architecture_dir / "pdd").is_dir()
    assert (architecture_dir / "pdd" / "run_all.sh").is_file()

    public_paths = [
        path.relative_to(architecture_dir).as_posix()
        for path in architecture_dir.rglob("*")
    ]
    assert "pd-af-disagg/run_all.sh" in public_paths
    assert "pd-af-disagg/offline/moe_model_basic.sh" in public_paths
    assert "pd-af-disagg/online/moe_model_basic_online.sh" in public_paths


def test_top_level_docs_advertise_supported_pdd_without_upcoming_claims() -> None:
    for relative_path in PUBLIC_DOCS[1:]:
        text = _read(relative_path)
        assert "pd-disaggregation" in text, relative_path
        assert "pdd/run_all.sh" in text, relative_path

    top_level_readme = _read("README.md")
    assert "Prefill-Decode Disaggregation (PDD) and sequential Attention-FFN Disaggregation" in top_level_readme
    assert "PDD serving" in top_level_readme
    assert "examples/architecture/pdd/offline/dense_model_basic.sh" in top_level_readme
    assert "examples/architecture/pdd/online/dense_model_basic_online.sh" in top_level_readme

    stale_claims = (
        "PDD and AFD support is planned",
        "PDD and AFD are upcoming roadmap items",
        "Current public branch supports co-location only",
        "Disaggregated architectures are intentionally not included",
        "The disaggregated version will be available soon",
        "not enabled in this branch yet",
    )
    combined_docs = "\n".join(_read(path) for path in PUBLIC_DOCS)
    for claim in stale_claims:
        assert claim not in combined_docs, f"Stale PDD release claim leaked: {claim}"

    assert "pd-af-disaggregation" in _read("AGENTS.md")
    assert "pd-af-disaggregation" in _read("README.md")
    assert "pd-af-disaggregation" in _read("examples/README.md")
    assert "pd-af-disaggregation" in _read("examples/architecture/README.md")
