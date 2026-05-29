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
from sleuth.evaluation.harness import PIPELINE_CACHE_VERSION, MMLongBenchEvaluator, load_sol_instructions_if_needed
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
    parser.add_argument("--retriever", default="vidore/colpali-v1.3-hf")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=["mock", "local", "sol"], default="sol")
    parser.add_argument("--render_dpi", type=int, default=144)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--agent-prompts-md", default="agent_prompts.md")
    parser.add_argument("--sol-instructions-md", default="sol_instructions.md")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--answer-extractor", choices=["auto", "none", "heuristic", "openai_compatible"], default="auto")
    parser.add_argument("--summary-examples", type=int, default=5)
    return parser.parse_args()


def build_llm_client(args: argparse.Namespace, config: dict, max_new_tokens: int):
    if args.mode == "mock":
        return MockClient()
    return QwenVLClient(
        model_name_or_path=args.model,
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
    examples = load_mmlongbench_examples(args.data_dir, limit=args.limit, category=args.category)
    if not examples:
        raise ValueError("No MMLongBench-Doc examples matched the requested filters.")

    llm_client = build_llm_client(args, config, max_new_tokens=int(max_tokens.get("core_decision", 512)))
    retriever, actual_retriever_name = build_retriever(args, config)
    sol_instruction_text = load_sol_instructions_if_needed(args.mode, args.sol_instructions_md)
    answer_extractor = build_answer_extractor(args.answer_extractor, mode=args.mode)

    run_config = {
        "data_dir": args.data_dir,
        "method": args.method,
        "mode": args.mode,
        "model": args.model,
        "retriever": args.retriever,
        "actual_retriever": actual_retriever_name,
        "top_k": args.top_k,
        "temperature": args.temperature,
        "limit": args.limit,
        "category": args.category,
        "output_dir": args.output_dir,
        "render_dpi": args.render_dpi,
        "num_examples": len(examples),
        "answer_extractor_requested": args.answer_extractor,
        "answer_extractor": answer_extractor.name,
        "answer_extractor_model": getattr(answer_extractor, "model", None),
        "answer_extractor_base_url": getattr(answer_extractor, "base_url", None),
        "pipeline_cache_version": PIPELINE_CACHE_VERSION,
        "paper_comparable_scoring": answer_extractor.paper_comparable,
        "difficulty_model_switching_enabled": False,
        "difficulty_model_switching_note": (
            "Difficulty Assessment generates an instruction set, but this prototype uses one Qwen-VL "
            "client for both ordinary and reasoning modes."
        ),
    }

    evaluator = MMLongBenchEvaluator(
        output_dir=args.output_dir,
        method=args.method,
        retriever=retriever,
        retriever_name=actual_retriever_name,
        llm_client=llm_client,
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
        },
        answer_extractor=answer_extractor,
    )
    metrics = evaluator.run(examples, run_config)
    print(json.dumps(metrics, indent=2))
    print()
    print_results_summary(args.output_dir, max_examples=args.summary_examples)


if __name__ == "__main__":
    main()
