from __future__ import annotations

import json

from sleuth.evaluation.qid_filter import resolve_qids


def test_resolve_qids_from_repeated_comma_values_and_file(tmp_path):
    qid_file = tmp_path / "qids.json"
    qid_file.write_text(json.dumps(["18", "19"]), encoding="utf-8")

    qids = resolve_qids(["13,15", "17"], qid_file)

    assert qids == {"13", "15", "17", "18", "19"}


def test_resolve_qids_from_text_file(tmp_path):
    qid_file = tmp_path / "qids.txt"
    qid_file.write_text("2,3\n# comment\n7\n", encoding="utf-8")

    assert resolve_qids(None, qid_file) == {"2", "3", "7"}
