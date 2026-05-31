#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sleuth.config import get_nested, load_config
from sleuth.evaluation.answer_extraction import build_answer_extractor
from sleuth.evaluation.dataset import load_mmlongbench_examples
from sleuth.evaluation.harness import (
    PIPELINE_CACHE_VERSION,
    MMLongBenchEvaluator,
    compute_agent_source_fingerprint,
    load_sol_instructions_if_needed,
)
from sleuth.evaluation.qid_filter import resolve_qids
from sleuth.evaluation.reporting import print_results_summary
from sleuth.llm.mock_client import MockClient
from sleuth.llm.qwen_vl_client import QwenVLClient
from sleuth.retrieval.colpali_retriever import ColPaliRetriever
from sleuth.retrieval.dummy_retriever import DummyRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MMLongBench-Doc evaluation for base or SLEUTH.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--method", choices=["base", "sleuth"], required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--thinking-model", default=None)
    parser.add_argument("--retriever", default="vidore/colpali-v1.3-hf")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--qid", action="append", default=None, help="Only run one question id; repeatable and comma-compatible")
    parser.add_argument("--qid-file", default=None, help="JSON/list/text file of question ids to run")
    parser.add_argument("--category", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None, help="Shared cache directory for rendered pages, ColPali, retrieval, and agents")
    parser.add_argument("--mode", choices=["mock", "local", "sol"], default="sol")
    parser.add_argument("--render_dpi", type=int, default=144)
    parser.add_argument("--evidence-page-base", choices=["auto", "zero", "one", "0-based", "1-based"], default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--agent-prompts-md", default="agent_prompts.md")
    parser.add_argument("--sol-instructions-md", default="sol_instructions.md")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--answer-extractor", choices=["auto", "none", "heuristic", "openai_compatible"], default="auto")
    parser.add_argument("--summary-examples", type=int, default=5)
    return parser.parse_args()


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def build_llm_client(args: argparse.Namespace, config: dict, model_name: str, max_new_tokens: int):
    if args.mode == "mock":
        return MockClient()
    return QwenVLClient(
        model_name_or_path=model_name,
        device=args.device or get_nested(config, ["model", "device"], "cuda"),
        dtype=args.dtype or get_nested(config, ["model", "dtype"], "bfloat16"),
        max_new_tokens=max_new_tokens,
    )


def build_retriever(args: argparse.Namespace, config: dict):
    if args.mode == "mock":
        return DummyRetriever(), "dummy"
    return (
        ColPaliRetriever(
            model_name_or_path=args.retriever,
            device=args.device or get_nested(config, ["model", "device"], "cuda"),
            top_k_default=args.top_k,
        ),
        args.retriever,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    max_tokens = get_nested(config, ["model", "max_new_tokens"], {})
    region_refinement = str(get_nested(config, ["pipeline", "region_refinement"], "fallback"))
    evidence_page_base = args.evidence_page_base or str(get_nested(config, ["evaluation", "evidence_page_base"], "auto"))
    difficulty_model_switching_enabled = (
        args.mode != "mock"
        and args.method == "sleuth"
        and _as_bool(get_nested(config, ["pipeline", "difficulty_model_switching_enabled"], True))
    )
    thinking_model = (
        args.thinking_model
        or get_nested(config, ["model", "thinking_name_or_path"], "Qwen/Qwen3-VL-8B-Thinking")
    )
    agent_source_fingerprint = compute_agent_source_fingerprint()
    qids = resolve_qids(args.qid, args.qid_file)
    examples = load_mmlongbench_examples(
        args.data_dir,
        limit=args.limit,
        category=args.category,
        evidence_page_base=evidence_page_base,
        qids=qids,
    )
    if not examples:
        raise ValueError("No MMLongBench-Doc examples matched the requested filters.")

    sol_instruction_text = load_sol_instructions_if_needed(args.mode, args.sol_instructions_md)
    answer_extractor = build_answer_extractor(args.answer_extractor, mode=args.mode)
    llm_client = build_llm_client(
        args,
        config,
        model_name=args.model,
        max_new_tokens=int(max_tokens.get("core_decision", 512)),
    )
    core_decision_thinking_client = None
    if difficulty_model_switching_enabled:
        core_decision_thinking_client = build_llm_client(
            args,
            config,
            model_name=str(thinking_model),
            max_new_tokens=int(max_tokens.get("core_decision_thinking", 4096)),
        )
    retriever, actual_retriever_name = build_retriever(args, config)

    run_config = {
        "data_dir": args.data_dir,
        "method": args.method,
        "mode": args.mode,
        "model": args.model,
        "retriever": args.retriever,
        "actual_retriever": actual_retriever_name,
        "thinking_model": str(thinking_model) if difficulty_model_switching_enabled else None,
        "top_k": args.top_k,
        "temperature": args.temperature,
        "limit": args.limit,
        "qids": sorted(qids) if qids else None,
        "qid_file": args.qid_file,
        "category": args.category,
        "output_dir": args.output_dir,
        "cache_dir": args.cache_dir or str(Path(args.output_dir) / "cache"),
        "render_dpi": args.render_dpi,
        "evidence_page_base": evidence_page_base,
        "num_examples": len(examples),
        "answer_extractor_requested": args.answer_extractor,
        "answer_extractor": answer_extractor.name,
        "answer_extractor_model": getattr(answer_extractor, "model", None),
        "answer_extractor_base_url": getattr(answer_extractor, "base_url", None),
        "pipeline_cache_version": PIPELINE_CACHE_VERSION,
        "region_refinement": region_refinement,
        "agent_source_fingerprint": agent_source_fingerprint,
        "paper_comparable_scoring": answer_extractor.paper_comparable,
        "difficulty_model_switching_enabled": difficulty_model_switching_enabled,
        "core_decision_thinking_max_tokens": int(max_tokens.get("core_decision_thinking", 4096)),
    }

    evaluator = MMLongBenchEvaluator(
        output_dir=args.output_dir,
        method=args.method,
        retriever=retriever,
        retriever_name=actual_retriever_name,
        llm_client=llm_client,
        core_decision_thinking_client=core_decision_thinking_client,
        top_k=args.top_k,
        temperature=args.temperature,
        render_dpi=args.render_dpi,
        agent_prompts_md=args.agent_prompts_md,
        sol_instruction_text=sol_instruction_text,
        max_tokens={
            "clue_discovery": int(max_tokens.get("clue_discovery", 3072)),
            "page_screening": int(max_tokens.get("page_screening", 512)),
            "difficulty_assessment": int(max_tokens.get("difficulty_assessment", 512)),
            "core_decision": int(max_tokens.get("core_decision", 512)),
            "core_decision_thinking": int(max_tokens.get("core_decision_thinking", 4096)),
        },
        answer_extractor=answer_extractor,
        region_refinement=region_refinement,
        difficulty_model_switching_enabled=difficulty_model_switching_enabled,
        thinking_model=str(thinking_model) if difficulty_model_switching_enabled else None,
        agent_source_fingerprint=agent_source_fingerprint,
        cache_dir=args.cache_dir,
    )
    metrics = evaluator.run(examples, run_config)
    print(json.dumps(metrics, indent=2))
    print()
    print_results_summary(args.output_dir, max_examples=args.summary_examples)


if __name__ == "__main__":
    main()
