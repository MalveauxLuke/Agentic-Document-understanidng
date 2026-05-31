import json
from pathlib import Path

from sleuth.evaluation.categories import normalize_categories, normalize_evidence_source
from sleuth.evaluation.dataset import load_mmlongbench_examples


def test_load_official_style_samples_and_zero_indexed_pages(tmp_path):
    data_dir = tmp_path / "mmlongbench"
    documents_dir = data_dir / "data" / "documents"
    documents_dir.mkdir(parents=True)
    (documents_dir / "doc-a.pdf").write_bytes(b"%PDF-1.4\n")
    samples = [
        {
            "doc_id": "doc-a.pdf",
            "doc_type": "Report",
            "question": "Which page has the answer?",
            "answer": "page zero",
            "evidence_pages": "[0, 2]",
            "evidence_sources": "['Pure-text (Plain-text)', 'Table']",
            "answer_format": "Str",
        }
    ]
    (data_dir / "data" / "samples.json").write_text(json.dumps(samples), encoding="utf-8")

    examples = load_mmlongbench_examples(data_dir, evidence_page_base="zero")

    assert len(examples) == 1
    assert examples[0].question_id == "0"
    assert examples[0].evidence_pages == [0, 2]
    assert examples[0].source_evidence_pages == [0, 2]
    assert examples[0].categories == ["Pure-text", "Table"]
    assert Path(examples[0].pdf_path).name == "doc-a.pdf"


def test_load_bundled_sample_from_repo_root():
    root = Path(__file__).resolve().parents[1]
    examples = load_mmlongbench_examples(root, limit=1)

    assert len(examples) == 1
    assert examples[0].doc_id.endswith(".pdf")
    assert examples[0].question == "WHAT IS USCA CASE NUMBER?"
    assert examples[0].evidence_pages == [0]
    assert examples[0].source_evidence_pages == [1]
    assert examples[0].categories == ["Pure-text"]
    assert Path(examples[0].pdf_path).exists()


def test_official_style_auto_normalizes_one_based_pages(tmp_path):
    data_dir = tmp_path / "mmlongbench"
    documents_dir = data_dir / "data" / "documents"
    documents_dir.mkdir(parents=True)
    (documents_dir / "doc-a.pdf").write_bytes(b"%PDF-1.4\n")
    samples = [
        {
            "doc_id": "doc-a.pdf",
            "doc_type": "Report",
            "question": "Which page has the answer?",
            "answer": "page one",
            "evidence_pages": "[1, 3]",
            "evidence_sources": "['Pure-text (Plain-text)']",
            "answer_format": "Str",
        }
    ]
    (data_dir / "data" / "samples.json").write_text(json.dumps(samples), encoding="utf-8")

    examples = load_mmlongbench_examples(data_dir)

    assert examples[0].source_evidence_pages == [1, 3]
    assert examples[0].evidence_pages == [0, 2]


def test_category_normalization_and_filter(tmp_path):
    assert normalize_evidence_source("Generalized-text (Layout)") == "Layout"
    assert normalize_evidence_source("Pure-text (Plain-text)") == "Pure-text"
    assert normalize_categories([]) == ["None"]

    data_dir = tmp_path / "mmlongbench"
    documents_dir = data_dir / "data" / "documents"
    documents_dir.mkdir(parents=True)
    (documents_dir / "doc.pdf").write_bytes(b"%PDF-1.4\n")
    rows = [
        {
            "doc_id": "doc.pdf",
            "question": "q1",
            "answer": "a1",
            "evidence_pages": "[]",
            "evidence_sources": "[]",
            "answer_format": "None",
        },
        {
            "doc_id": "doc.pdf",
            "question": "q2",
            "answer": "a2",
            "evidence_pages": "[0]",
            "evidence_sources": "['Figure']",
            "answer_format": "Str",
        },
    ]
    (data_dir / "data" / "samples.json").write_text(json.dumps(rows), encoding="utf-8")

    none_examples = load_mmlongbench_examples(data_dir, category="None")
    figure_examples = load_mmlongbench_examples(data_dir, category="Figure")

    assert [example.question for example in none_examples] == ["q1"]
    assert [example.question for example in figure_examples] == ["q2"]


def test_qid_filter_runs_before_limit(tmp_path):
    data_dir = tmp_path / "mmlongbench"
    documents_dir = data_dir / "data" / "documents"
    documents_dir.mkdir(parents=True)
    (documents_dir / "doc.pdf").write_bytes(b"%PDF-1.4\n")
    rows = [
        {
            "question_id": "a",
            "doc_id": "doc.pdf",
            "question": "q-a",
            "answer": "a",
            "evidence_pages": "[]",
            "evidence_sources": "[]",
            "answer_format": "None",
        },
        {
            "question_id": "b",
            "doc_id": "doc.pdf",
            "question": "q-b",
            "answer": "b",
            "evidence_pages": "[]",
            "evidence_sources": "[]",
            "answer_format": "None",
        },
    ]
    (data_dir / "data" / "samples.json").write_text(json.dumps(rows), encoding="utf-8")

    examples = load_mmlongbench_examples(data_dir, qids={"b"}, limit=1)

    assert [example.question_id for example in examples] == ["b"]
