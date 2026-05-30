from __future__ import annotations

from sleuth.evaluation import harness


def test_agent_source_fingerprint_is_short_hex():
    fingerprint = harness.compute_agent_source_fingerprint()

    assert len(fingerprint) == 16
    int(fingerprint, 16)


def test_agent_source_fingerprint_changes_when_source_changes(monkeypatch):
    payload = {"content": b"first version"}

    def fake_read_bytes(self):
        return payload["content"]

    monkeypatch.setattr(harness, "AGENT_SOURCE_FILES", ["sleuth/agents/core_decision.py"])
    monkeypatch.setattr(harness.Path, "read_bytes", fake_read_bytes)

    first = harness.compute_agent_source_fingerprint()
    payload["content"] = b"second version"
    second = harness.compute_agent_source_fingerprint()

    assert first != second
