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


def test_public_architecture_readme_uses_clean_pdd_surface_terms() -> None:
    forbidden_terms = (
        "DECODE_ATTN",
        "DECODE_FFN",
    )
    for relative_path in ("examples/README.md", "examples/architecture/README.md"):
        readme_text = _read(relative_path)
        for term in forbidden_terms:
            assert term not in readme_text, (
                f"Legacy public PDD term leaked from {relative_path}: {term}"
            )

    readme_text = _read("examples/architecture/README.md")
    assert "pd-af-disaggregation" not in readme_text

    assert "PDD / `pd-disaggregation`" in readme_text
    assert "`pdd/run_all.sh`" in readme_text
    assert "--no-enable_parallel_clusters" in readme_text


def test_public_architecture_entrypoints_stay_on_supported_pdd_path() -> None:
    architecture_dir = REPO_ROOT / "examples" / "architecture"
    assert (architecture_dir / "pdd").is_dir()
    assert (architecture_dir / "pdd" / "run_all.sh").is_file()

    forbidden_path_fragments = (
        "pd-af",
        "pd_disaggregation",
        "pd-disaggregation",
        "decode_attn",
        "decode-ffn",
        "decode_ffn",
        "decode-attn",
    )
    public_paths = [
        path.relative_to(architecture_dir).as_posix()
        for path in architecture_dir.rglob("*")
    ]
    for public_path in public_paths:
        normalized = public_path.lower()
        for fragment in forbidden_path_fragments:
            assert fragment not in normalized, (
                f"Unsupported public PDD entrypoint leaked: {public_path}"
            )


def test_top_level_docs_advertise_supported_pdd_without_upcoming_claims() -> None:
    for relative_path in PUBLIC_DOCS:
        text = _read(relative_path)
        assert "pd-disaggregation" in text, relative_path
        assert "pdd/run_all.sh" in text, relative_path

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
    assert "pd-af-disaggregation" not in _read("README.md")
    assert "pd-af-disaggregation" not in _read("examples/README.md")
    assert "pd-af-disaggregation" not in _read("examples/architecture/README.md")
