#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sleuth.agents.clue_discovery import ClueDiscoveryAgent
from sleuth.agents.core_decision import CoreDecisionAgent
from sleuth.agents.difficulty_assessment import DifficultyAssessmentAgent
from sleuth.agents.page_screening import PageScreeningAgent
from sleuth.config import get_nested, load_config
from sleuth.instructions.instruction_loader import read_markdown_file, save_instruction_copy
from sleuth.instructions.prompt_loader import get_prompt_section, load_agent_prompt_markdown
from sleuth.llm.mock_client import MockClient
from sleuth.llm.openai_compatible import OpenAICompatibleClient
from sleuth.llm.qwen_vl_client import QwenVLClient
from sleuth.pipeline.sleuth_pipeline import SleuthPipeline
from sleuth.retrieval.colpali_retriever import ColPaliRetriever
from sleuth.retrieval.dummy_retriever import DummyRetriever
from sleuth.retrieval.text_bm25_retriever import BM25TextRetriever
from sleuth.utils.file_utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the static SLEUTH-style baseline.")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=["mock", "local", "sol"], default="mock")
    parser.add_argument("--retriever", choices=["dummy", "bm25", "colpali"], default=None)
    parser.add_argument("--llm", choices=["mock", "openai", "qwen-vl"], default=None)
    parser.add_argument("--agent-prompts-md", default="agent_prompts.md")
    parser.add_argument("--sol-instructions-md", default="sol_instructions.md")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--render-dpi", type=int, default=144)
    return parser.parse_args()


def resolve_mode(args: argparse.Namespace) -> tuple[str, str]:
    if args.mode == "sol":
        if args.llm is not None and args.llm != "qwen-vl":
            raise ValueError("SOL mode requires --llm qwen-vl.")
        if args.retriever is not None and args.retriever != "colpali":
            raise ValueError("SOL mode requires --retriever colpali.")
        if args.top_k != 5:
            raise ValueError("SOL mode uses top-5 retrieval; set --top-k 5.")
        if abs(args.temperature - 0.1) > 1e-9:
            raise ValueError("SOL mode uses temperature 0.1.")
        return "qwen-vl", "colpali"

    if args.mode == "mock":
        return args.llm or "mock", args.retriever or "dummy"

    return args.llm or "mock", args.retriever or "bm25"


def build_llm(llm_name: str, config: dict, max_new_tokens: int):
    if llm_name == "mock":
        return MockClient()
    if llm_name == "openai":
        return OpenAICompatibleClient()
    if llm_name == "qwen-vl":
        return QwenVLClient(
            model_name_or_path=get_nested(config, ["model", "name_or_path"], "Qwen/Qwen3-VL-8B-Instruct"),
            device=get_nested(config, ["model", "device"], "cuda"),
            dtype=get_nested(config, ["model", "dtype"], "bfloat16"),
            max_new_tokens=max_new_tokens,
        )
    raise ValueError(f"Unsupported LLM: {llm_name}")


def build_retriever(retriever_name: str, config: dict):
    if retriever_name == "dummy":
        return DummyRetriever()
    if retriever_name == "bm25":
        return BM25TextRetriever()
    if retriever_name == "colpali":
        return ColPaliRetriever(
            model_name_or_path=get_nested(config, ["retriever", "name_or_path"], "vidore/colpali-v1.3"),
            device=get_nested(config, ["model", "device"], "cuda"),
            top_k_default=get_nested(config, ["retriever", "top_k"], 5),
        )
    raise ValueError(f"Unsupported retriever: {retriever_name}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    llm_name, retriever_name = resolve_mode(args)

    agent_prompt_markdown = load_agent_prompt_markdown(args.agent_prompts_md)
    sol_instruction_text = None
    if args.mode == "sol":
        sol_instruction_text = read_markdown_file(args.sol_instructions_md)

    final_dir = ensure_dir(Path(args.out_dir) / "final")
    save_instruction_copy(agent_prompt_markdown, final_dir / "agent_prompts_used.md")
    if sol_instruction_text is not None:
        save_instruction_copy(sol_instruction_text, final_dir / "sol_instructions_used.md")

    max_tokens = get_nested(config, ["model", "max_new_tokens"], {})
    region_refinement = str(get_nested(config, ["pipeline", "region_refinement"], "fallback"))
    llm_client = build_llm(
        llm_name,
        config,
        max_new_tokens=int(max_tokens.get("clue_discovery", 3072)),
    )
    retriever = build_retriever(retriever_name, config)

    clue_agent = ClueDiscoveryAgent(
        llm_client=llm_client,
        agent_prompt_text=get_prompt_section(agent_prompt_markdown, "clue_discovery"),
        sol_instruction_text=sol_instruction_text,
        temperature=args.temperature,
        max_new_tokens=int(max_tokens.get("clue_discovery", 3072)),
        region_refinement=region_refinement,
    )
    page_screening_agent = PageScreeningAgent(
        llm_client=llm_client,
        agent_prompt_text=get_prompt_section(agent_prompt_markdown, "page_screening"),
        sol_instruction_text=sol_instruction_text,
        temperature=args.temperature,
        max_new_tokens=int(max_tokens.get("page_screening", 512)),
    )
    difficulty_agent = DifficultyAssessmentAgent(
        llm_client=llm_client,
        agent_prompt_text=get_prompt_section(agent_prompt_markdown, "difficulty_assessment"),
        sol_instruction_text=sol_instruction_text,
        temperature=args.temperature,
        max_new_tokens=int(max_tokens.get("difficulty_assessment", 512)),
    )
    core_decision_agent = CoreDecisionAgent(
        llm_client=llm_client,
        text_agent_prompt_text=get_prompt_section(agent_prompt_markdown, "core_decision_text"),
        visual_agent_prompt_text=get_prompt_section(agent_prompt_markdown, "core_decision_visual"),
        sol_instruction_text=sol_instruction_text,
        temperature=args.temperature,
        max_new_tokens=int(max_tokens.get("core_decision", 512)),
    )

    pipeline = SleuthPipeline(
        retriever=retriever,
        clue_agent=clue_agent,
        page_screening_agent=page_screening_agent,
        difficulty_agent=difficulty_agent,
        core_decision_agent=core_decision_agent,
        render_dpi=args.render_dpi,
    )
    result = pipeline.run(
        pdf_path=args.pdf,
        question=args.question,
        top_k=args.top_k,
        out_dir=args.out_dir,
    )

    print(f"Final answer: {result.final_answer.answer}")
    print("Evidence references:")
    print(json.dumps(result.final_answer.evidence_references, indent=2))
    print(f"Output directory: {result.output_dir}")


if __name__ == "__main__":
    main()
