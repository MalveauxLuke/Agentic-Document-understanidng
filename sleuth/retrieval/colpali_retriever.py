from __future__ import annotations

from pathlib import Path

from PIL import Image

from sleuth.retrieval.base import BaseRetriever
from sleuth.schemas import DocumentPage, RetrievedPage


COLPALI_INSTALL_MESSAGE = (
    "ColPali retrieval requires colpali-engine. Try: pip install colpali-engine. "
    "If that fails, install the official ColPali package or repository according "
    "to the current ColPali documentation."
)


class ColPaliRetriever(BaseRetriever):
    def __init__(
        self,
        model_name_or_path: str = "vidore/colpali-v1.3",
        device: str = "cuda",
        top_k_default: int = 5,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.top_k_default = top_k_default
        self.document_pages: list[DocumentPage] = []
        self.image_embeddings = None

        try:
            import torch
            from colpali_engine.models import ColPali, ColPaliProcessor
        except ImportError as exc:
            raise ImportError(COLPALI_INSTALL_MESSAGE) from exc

        self.torch = torch
        self.processor = ColPaliProcessor.from_pretrained(model_name_or_path)
        try:
            self.model = ColPali.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
            ).to(device)
        except TypeError:
            self.model = ColPali.from_pretrained(model_name_or_path).to(device)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load ColPali. Check GPU availability, package versions, "
                f"and model identifier '{model_name_or_path}'."
            ) from exc
        self.model.eval()

    def index(self, document_pages: list[DocumentPage]) -> None:
        self.document_pages = list(document_pages)
        if not self.document_pages:
            self.image_embeddings = None
            return

        images = [Image.open(Path(page.image_path)).convert("RGB") for page in self.document_pages]
        try:
            batch = self.processor.process_images(images).to(self.device)
            with self.torch.no_grad():
                self.image_embeddings = self.model(**batch)
        except AttributeError as exc:
            raise RuntimeError(
                "The installed ColPali API differs from this prototype's expected "
                "ColPaliProcessor.process_images/process_queries interface. Keep "
                "the BaseRetriever contract stable and update only this file."
            ) from exc

    def retrieve(self, question: str, top_k: int = 5) -> list[RetrievedPage]:
        if self.image_embeddings is None:
            return []

        k = top_k or self.top_k_default
        try:
            query_batch = self.processor.process_queries([question]).to(self.device)
            with self.torch.no_grad():
                query_embeddings = self.model(**query_batch)
                scores = self.processor.score_multi_vector(query_embeddings, self.image_embeddings)
        except AttributeError as exc:
            raise RuntimeError(
                "The installed ColPali API differs from this prototype's expected "
                "multi-vector scoring interface. Keep the BaseRetriever contract "
                "stable and update only this file."
            ) from exc

        if hasattr(scores, "detach"):
            score_values = scores[0].detach().float().cpu().tolist()
        else:
            score_values = list(scores[0])
        ranked = sorted(enumerate(score_values), key=lambda item: item[1], reverse=True)[:k]
        return [
            RetrievedPage(
                page_index=self.document_pages[index].page_index,
                score=float(score),
                reason="colpali visual page score",
            )
            for index, score in ranked
        ]
