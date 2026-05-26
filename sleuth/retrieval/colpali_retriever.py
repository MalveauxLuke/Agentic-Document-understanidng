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
        model_name_or_path: str = "vidore/colpali-v1.3-hf",
        device: str = "cuda",
        top_k_default: int = 5,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.top_k_default = top_k_default
        self.document_pages: list[DocumentPage] = []
        self.image_embeddings = None
        self.backend = "transformers"

        try:
            import torch
        except ImportError as exc:
            raise ImportError("ColPali retrieval requires torch.") from exc

        self.torch = torch
        self.processor = None
        self.model = None
        self._load_model()

    def _hf_native_model_name(self) -> str:
        if self.model_name_or_path.endswith("-hf"):
            return self.model_name_or_path
        if self.model_name_or_path == "vidore/colpali-v1.3":
            return "vidore/colpali-v1.3-hf"
        return self.model_name_or_path

    def _load_model(self) -> None:
        try:
            from transformers import ColPaliForRetrieval, ColPaliProcessor
        except ImportError:
            ColPaliForRetrieval = None
            ColPaliProcessor = None

        if ColPaliForRetrieval is not None and ColPaliProcessor is not None:
            model_name = self._hf_native_model_name()
            try:
                self.processor = ColPaliProcessor.from_pretrained(model_name)
                self.model = ColPaliForRetrieval.from_pretrained(
                    model_name,
                    dtype=self.torch.bfloat16,
                    device_map="auto" if self.device == "cuda" else self.device,
                )
                self.model.eval()
                self.backend = "transformers"
                return
            except TypeError:
                self.processor = ColPaliProcessor.from_pretrained(model_name)
                self.model = ColPaliForRetrieval.from_pretrained(
                    model_name,
                    torch_dtype=self.torch.bfloat16,
                    device_map="auto" if self.device == "cuda" else self.device,
                )
                self.model.eval()
                self.backend = "transformers"
                return
            except Exception:
                pass

        try:
            from colpali_engine.models import ColPali, ColPaliProcessor as EngineProcessor
        except ImportError as exc:
            raise ImportError(
                "Could not load HF-native ColPali via transformers and colpali-engine is unavailable. "
                f"{COLPALI_INSTALL_MESSAGE}"
            ) from exc

        self.processor = EngineProcessor.from_pretrained(self.model_name_or_path)
        try:
            self.model = ColPali.from_pretrained(
                self.model_name_or_path,
                torch_dtype=self.torch.bfloat16,
                device_map="cuda:0" if self.device == "cuda" else self.device,
            )
        except TypeError:
            self.model = ColPali.from_pretrained(
                self.model_name_or_path,
                torch_dtype=self.torch.bfloat16,
            ).to(self.device)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load ColPali. Check GPU availability, package versions, "
                f"and model identifier '{self.model_name_or_path}'. Prefer the HF-native "
                "checkpoint 'vidore/colpali-v1.3-hf' with current Transformers."
            ) from exc
        self.model.eval()
        self.backend = "colpali-engine"

    def _model_device(self):
        return getattr(self.model, "device", self.device)

    def index(self, document_pages: list[DocumentPage]) -> None:
        self.document_pages = list(document_pages)
        if not self.document_pages:
            self.image_embeddings = None
            return

        images = [Image.open(Path(page.image_path)).convert("RGB") for page in self.document_pages]
        if self.backend == "transformers":
            batch = self.processor(images=images, return_tensors="pt").to(self._model_device())
            with self.torch.no_grad():
                self.image_embeddings = self.model(**batch).embeddings
            return

        batch = self.processor.process_images(images).to(self.device)
        with self.torch.no_grad():
            self.image_embeddings = self.model(**batch)

    def retrieve(self, question: str, top_k: int = 5) -> list[RetrievedPage]:
        if self.image_embeddings is None:
            return []

        k = top_k or self.top_k_default
        if self.backend == "transformers":
            query_batch = self.processor(text=[question], return_tensors="pt").to(self._model_device())
            with self.torch.no_grad():
                query_embeddings = self.model(**query_batch).embeddings
                scores = self.processor.score_retrieval(query_embeddings, self.image_embeddings)
        else:
            query_batch = self.processor.process_queries([question]).to(self.device)
            with self.torch.no_grad():
                query_embeddings = self.model(**query_batch)
                scores = self.processor.score_multi_vector(query_embeddings, self.image_embeddings)

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
