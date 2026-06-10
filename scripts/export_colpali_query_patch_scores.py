#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sleuth.evaluation.cache_paths import (
    colpali_embedding_cache_path,
    load_cached_document_pages,
)
from sleuth.evaluation.qid_filter import resolve_qids
from sleuth.retrieval.colpali_retriever import ColPaliRetriever


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_id(value: Any) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in str(value))


def _load_cases(corpus_json: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = _read_json(corpus_json)
    if isinstance(payload, dict):
        cases = payload.get("cases") or payload.get("questions") or []
        return list(cases), payload
    if isinstance(payload, list):
        return list(payload), {}
    raise ValueError(f"Unsupported corpus JSON shape in {corpus_json}")


def _case_top_pages(case: dict[str, Any], top_k: int | None = None) -> list[dict[str, Any]]:
    if case.get("colpali_top5_pages"):
        pages = list(case["colpali_top5_pages"])
    elif case.get("retrieved"):
        pages = list(case["retrieved"])
    elif case.get("top5_display_pages"):
        pages = [
            {
                "rank": rank,
                "page_index": int(display_page) - 1,
                "display_page": int(display_page),
                "score": None,
            }
            for rank, display_page in enumerate(case["top5_display_pages"], start=1)
        ]
    else:
        pages = []
    if top_k is not None:
        pages = pages[:top_k]
    normalized = []
    for rank, page in enumerate(pages, start=1):
        page_index = int(page.get("page_index", int(page.get("display_page")) - 1))
        normalized.append(
            {
                **page,
                "rank": int(page.get("rank", rank)),
                "page_index": page_index,
                "display_page": int(page.get("display_page", page_index + 1)),
            }
        )
    return normalized


def _group_cases_by_doc(cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[str(case["doc_id"])].append(case)
    return dict(grouped)


def _torch_load(torch: Any, path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu")
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def _tensor_to_numpy(tensor: Any, dtype: np.dtype = np.float32) -> np.ndarray:
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().float().cpu()
    return np.asarray(tensor, dtype=dtype)


def _move_batch_to_device(batch: Any, device: Any) -> Any:
    return batch.to(device) if hasattr(batch, "to") else batch


def _encode_query(retriever: ColPaliRetriever, question: str) -> dict[str, Any]:
    torch = retriever.torch
    if retriever.backend == "transformers":
        batch = retriever.processor(text=[question], return_tensors="pt")
        batch = _move_batch_to_device(batch, retriever._model_device())
        with torch.no_grad():
            embeddings = retriever.model(**batch).embeddings
    else:
        batch = retriever.processor.process_queries([question])
        batch = _move_batch_to_device(batch, retriever.device)
        with torch.no_grad():
            embeddings = retriever.model(**batch)

    input_ids = batch.get("input_ids") if hasattr(batch, "get") else None
    attention_mask = batch.get("attention_mask") if hasattr(batch, "get") else None
    input_ids_list = input_ids[0].detach().cpu().tolist() if input_ids is not None else list(range(embeddings.shape[1]))
    mask_list = attention_mask[0].detach().cpu().tolist() if attention_mask is not None else [1] * len(input_ids_list)

    tokenizer = getattr(retriever.processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "convert_ids_to_tokens"):
        tokens = tokenizer.convert_ids_to_tokens(input_ids_list)
    else:
        tokens = [str(token_id) for token_id in input_ids_list]

    valid_positions = [index for index, mask in enumerate(mask_list) if int(mask) == 1]
    valid_token_ids = [int(input_ids_list[index]) for index in valid_positions]
    valid_tokens = [tokens[index] for index in valid_positions]
    valid_embeddings = embeddings[0, valid_positions, :].detach().float()
    return {
        "embeddings": valid_embeddings,
        "token_ids": valid_token_ids,
        "tokens": valid_tokens,
        "valid_positions": valid_positions,
        "raw_token_count": len(input_ids_list),
        "valid_token_count": len(valid_positions),
    }


def _infer_patch_grid(patch_count: int, page_width: int | None, page_height: int | None) -> dict[str, Any]:
    sqrt_n = int(math.sqrt(patch_count))
    if sqrt_n * sqrt_n == patch_count:
        return {
            "rows": sqrt_n,
            "cols": sqrt_n,
            "padded_patch_count": patch_count,
            "patch_count": patch_count,
            "method": "perfect_square",
            "confidence": "medium",
        }

    page_aspect = (page_width / page_height) if page_width and page_height else 1.0
    factors: list[tuple[float, int, int]] = []
    for rows in range(1, int(math.sqrt(patch_count)) + 1):
        if patch_count % rows == 0:
            cols = patch_count // rows
            factors.append((abs((cols / rows) - page_aspect), rows, cols))
            factors.append((abs((rows / cols) - page_aspect), cols, rows))
    if factors:
        _, rows, cols = min(factors, key=lambda item: item[0])
        return {
            "rows": rows,
            "cols": cols,
            "padded_patch_count": patch_count,
            "patch_count": patch_count,
            "method": "factor_pair_closest_to_page_aspect",
            "confidence": "low",
        }

    rows = max(1, int(round(math.sqrt(patch_count / max(page_aspect, 0.1)))))
    cols = int(math.ceil(patch_count / rows))
    return {
        "rows": rows,
        "cols": cols,
        "padded_patch_count": rows * cols,
        "patch_count": patch_count,
        "method": "padded_aspect_estimate",
        "confidence": "low",
    }


def _page_dimensions(path: Path | None) -> tuple[int | None, int | None]:
    if path is None or not path.exists():
        return None, None
    with Image.open(path) as image:
        return image.width, image.height


def _copy_page_image(page_image_path: Path | None, target_dir: Path, display_page: int) -> str | None:
    if page_image_path is None or not page_image_path.exists():
        return None
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"page_{display_page:04d}{page_image_path.suffix or '.png'}"
    shutil.copy2(page_image_path, target)
    return str(target)


def _top_indices(values: np.ndarray, k: int) -> list[dict[str, Any]]:
    if values.size == 0:
        return []
    k = min(max(1, k), int(values.size))
    indices = np.argpartition(-values, k - 1)[:k]
    indices = indices[np.argsort(-values[indices])]
    return [{"patch_index": int(index), "score": float(values[index])} for index in indices]


def _score_page(
    *,
    torch: Any,
    query: dict[str, Any],
    page_embedding: Any,
    top_patches_per_token: int,
) -> dict[str, Any]:
    q = query["embeddings"]
    p = page_embedding.detach().float().to(q.device)
    similarity = torch.matmul(q, p.transpose(0, 1)).float()
    token_best_score, token_best_patch = similarity.max(dim=1)
    patch_scores_contribution = torch.zeros(similarity.shape[1], dtype=torch.float32, device=similarity.device)
    patch_scores_contribution.scatter_add_(0, token_best_patch, token_best_score)
    patch_scores_dense_max = similarity.max(dim=0).values
    patch_scores_dense_mean = similarity.mean(dim=0)

    similarity_np = _tensor_to_numpy(similarity)
    token_best_patch_np = _tensor_to_numpy(token_best_patch, dtype=np.int32)
    token_best_score_np = _tensor_to_numpy(token_best_score)
    contribution_np = _tensor_to_numpy(patch_scores_contribution)
    dense_max_np = _tensor_to_numpy(patch_scores_dense_max)
    dense_mean_np = _tensor_to_numpy(patch_scores_dense_mean)

    token_matches = []
    for token_index, token in enumerate(query["tokens"]):
        token_scores = similarity_np[token_index]
        token_matches.append(
            {
                "token_index": token_index,
                "query_position": int(query["valid_positions"][token_index]),
                "token_id": int(query["token_ids"][token_index]),
                "token": str(token),
                "best_patch_index": int(token_best_patch_np[token_index]),
                "best_patch_score": float(token_best_score_np[token_index]),
                "top_patches": _top_indices(token_scores, top_patches_per_token),
            }
        )

    return {
        "similarity": similarity_np,
        "token_best_patch": token_best_patch_np,
        "token_best_score": token_best_score_np,
        "patch_scores_contribution": contribution_np,
        "patch_scores_dense_max": dense_max_np,
        "patch_scores_dense_mean": dense_mean_np,
        "score_from_token_best_sum": float(token_best_score_np.sum()),
        "token_matches": token_matches,
        "top_contribution_patches": _top_indices(contribution_np, 25),
        "top_dense_max_patches": _top_indices(dense_max_np, 25),
    }


def _save_npz(
    path: Path,
    *,
    score_data: dict[str, Any],
    query: dict[str, Any],
    include_similarity_matrix: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "token_best_patch": score_data["token_best_patch"].astype(np.int32),
        "token_best_score": score_data["token_best_score"].astype(np.float32),
        "patch_scores_contribution": score_data["patch_scores_contribution"].astype(np.float32),
        "patch_scores_dense_max": score_data["patch_scores_dense_max"].astype(np.float32),
        "patch_scores_dense_mean": score_data["patch_scores_dense_mean"].astype(np.float32),
        "query_token_ids": np.asarray(query["token_ids"], dtype=np.int64),
        "valid_query_positions": np.asarray(query["valid_positions"], dtype=np.int64),
    }
    if include_similarity_matrix:
        arrays["similarity"] = score_data["similarity"].astype(np.float32)
    np.savez_compressed(path, **arrays)


def export_query_patch_scores(
    *,
    corpus_json: Path,
    cache_dir: Path,
    output_dir: Path,
    retriever_name: str,
    render_dpi: int,
    top_k: int,
    device: str,
    qids: set[str] | None,
    max_cases: int | None,
    copy_page_images: bool,
    include_similarity_matrix: bool,
    top_patches_per_token: int,
) -> dict[str, Any]:
    cases, corpus_meta = _load_cases(corpus_json)
    if qids is not None:
        cases = [case for case in cases if str(case.get("question_id")) in qids]
    if max_cases is not None:
        cases = cases[:max_cases]
    if not cases:
        raise ValueError("No cases matched the requested filters.")

    output_dir.mkdir(parents=True, exist_ok=True)
    retriever = ColPaliRetriever(model_name_or_path=retriever_name, device=device)
    torch = retriever.torch
    grouped = _group_cases_by_doc(cases)

    manifest: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "corpus_json": str(corpus_json),
        "cache_dir": str(cache_dir),
        "output_dir": str(output_dir),
        "retriever": retriever_name,
        "backend": retriever.backend,
        "render_dpi": render_dpi,
        "top_k": top_k,
        "device": device,
        "copy_page_images": copy_page_images,
        "include_similarity_matrix": include_similarity_matrix,
        "top_patches_per_token": top_patches_per_token,
        "corpus_name": corpus_meta.get("name"),
        "case_count": len(cases),
        "document_count": len(grouped),
        "page_score_count": 0,
        "failures": [],
    }
    question_index: list[dict[str, Any]] = []

    for doc_number, (doc_id, doc_cases) in enumerate(grouped.items(), start=1):
        embedding_path = colpali_embedding_cache_path(cache_dir, retriever_name, render_dpi, doc_id)
        print(f"[colpali-heatmap-export] {doc_number}/{len(grouped)} doc={doc_id} cases={len(doc_cases)}", flush=True)
        if not embedding_path.exists():
            failure = {"doc_id": doc_id, "error": f"missing embedding cache: {embedding_path}"}
            manifest["failures"].append(failure)
            print(f"[colpali-heatmap-export] ERROR {failure['error']}", file=sys.stderr, flush=True)
            continue
        try:
            payload = _torch_load(torch, embedding_path)
            doc_embeddings = payload["embeddings"]
            page_indices = [int(page) for page in payload.get("page_indices") or []]
            page_row_by_index = {page_index: row for row, page_index in enumerate(page_indices)}
            pages = load_cached_document_pages(doc_id, cache_dir, render_dpi) or []
            page_image_by_index = {page.page_index: Path(page.image_path) for page in pages}

            for case in doc_cases:
                qid = str(case["question_id"])
                question = str(case["question"])
                case_dir = output_dir / "cases" / _safe_id(qid)
                pages_dir = case_dir / "pages"
                arrays_dir = case_dir / "arrays"
                query = _encode_query(retriever, question)
                top_pages = _case_top_pages(case, top_k=top_k)
                case_records: list[dict[str, Any]] = []
                _write_json(
                    case_dir / "query_tokens.json",
                    {
                        "question_id": qid,
                        "question": question,
                        "raw_token_count": query["raw_token_count"],
                        "valid_token_count": query["valid_token_count"],
                        "tokens": [
                            {
                                "token_index": index,
                                "query_position": int(query["valid_positions"][index]),
                                "token_id": int(query["token_ids"][index]),
                                "token": str(token),
                            }
                            for index, token in enumerate(query["tokens"])
                        ],
                    },
                )

                for page in top_pages:
                    page_index = int(page["page_index"])
                    if page_index not in page_row_by_index:
                        manifest["failures"].append(
                            {"doc_id": doc_id, "question_id": qid, "page_index": page_index, "error": "page missing from embedding payload"}
                        )
                        continue
                    display_page = int(page["display_page"])
                    page_image_path = page_image_by_index.get(page_index)
                    page_width, page_height = _page_dimensions(page_image_path)
                    embedding_row = page_row_by_index[page_index]
                    page_embedding = doc_embeddings[embedding_row]
                    score_data = _score_page(
                        torch=torch,
                        query=query,
                        page_embedding=page_embedding,
                        top_patches_per_token=top_patches_per_token,
                    )
                    patch_count = int(score_data["patch_scores_contribution"].shape[0])
                    patch_grid = _infer_patch_grid(patch_count, page_width, page_height)
                    array_path = arrays_dir / f"page_{display_page:04d}_scores.npz"
                    _save_npz(
                        array_path,
                        score_data=score_data,
                        query=query,
                        include_similarity_matrix=include_similarity_matrix,
                    )
                    copied_image_path = _copy_page_image(page_image_path, pages_dir, display_page) if copy_page_images else None
                    page_record = {
                        "question_id": qid,
                        "doc_id": doc_id,
                        "rank": int(page["rank"]),
                        "page_index": page_index,
                        "display_page": display_page,
                        "retrieval_score": page.get("score"),
                        "is_gold": page.get("is_gold"),
                        "page_image_path": str(page_image_path) if page_image_path else None,
                        "copied_page_image_path": copied_image_path,
                        "page_width": page_width,
                        "page_height": page_height,
                        "embedding_cache_path": str(embedding_path),
                        "embedding_row": embedding_row,
                        "patch_count": patch_count,
                        "patch_grid": patch_grid,
                        "valid_query_token_count": query["valid_token_count"],
                        "score_from_token_best_sum": score_data["score_from_token_best_sum"],
                        "array_path": str(array_path),
                        "token_matches": score_data["token_matches"],
                        "top_contribution_patches": score_data["top_contribution_patches"],
                        "top_dense_max_patches": score_data["top_dense_max_patches"],
                    }
                    metadata_path = case_dir / f"page_{display_page:04d}_metadata.json"
                    _write_json(metadata_path, page_record)
                    case_records.append({**page_record, "metadata_path": str(metadata_path)})
                    manifest["page_score_count"] += 1

                case_summary = {
                    "question_id": qid,
                    "doc_id": doc_id,
                    "modalities": case.get("modalities") or case.get("categories") or [],
                    "selected_for_category": case.get("selected_for_category"),
                    "question": question,
                    "answer": case.get("answer"),
                    "answer_format": case.get("answer_format"),
                    "gold_standard_pages": case.get("gold_standard_pages"),
                    "top_pages": top_pages,
                    "query_tokens_path": str(case_dir / "query_tokens.json"),
                    "pages": case_records,
                }
                _write_json(case_dir / "case_heatmap_export.json", case_summary)
                question_index.append(case_summary)
        except Exception as exc:
            failure = {"doc_id": doc_id, "error": f"{type(exc).__name__}: {exc}"}
            manifest["failures"].append(failure)
            print(f"[colpali-heatmap-export] ERROR doc={doc_id}: {failure['error']}", file=sys.stderr, flush=True)

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(output_dir / "questions.json", question_index)
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ColPali query-token x image-patch scores for heatmap visualization."
    )
    parser.add_argument("--corpus_json", default="colpali_crop_test_corpus/crop_colpali_test_corpus.json")
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/colpali_query_patch_scores")
    parser.add_argument("--retriever", default="vidore/colpali-v1.3-hf")
    parser.add_argument("--render_dpi", type=int, default=144)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--qid", action="append", default=None, help="Only export one question id; repeatable and comma-compatible")
    parser.add_argument("--qid-file", default=None, help="JSON/list/text file of question ids to export")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--no-copy-page-images", action="store_true")
    parser.add_argument("--no-full-similarity-matrix", action="store_true")
    parser.add_argument("--top-patches-per-token", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qids = resolve_qids(args.qid, args.qid_file)
    result = export_query_patch_scores(
        corpus_json=Path(args.corpus_json),
        cache_dir=Path(args.cache_dir),
        output_dir=Path(args.output_dir),
        retriever_name=args.retriever,
        render_dpi=args.render_dpi,
        top_k=args.top_k,
        device=args.device,
        qids=qids,
        max_cases=args.max_cases,
        copy_page_images=not bool(args.no_copy_page_images),
        include_similarity_matrix=not bool(args.no_full_similarity_matrix),
        top_patches_per_token=max(1, int(args.top_patches_per_token)),
    )
    print(json.dumps(result, indent=2), flush=True)
    if result.get("failures"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
