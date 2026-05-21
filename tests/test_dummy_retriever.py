from sleuth.retrieval.dummy_retriever import DummyRetriever
from sleuth.schemas import DocumentPage


def test_dummy_retriever_returns_first_top_k_in_order():
    pages = [
        DocumentPage(page_index=index, image_path=f"page_{index}.png", text=f"text {index}")
        for index in range(5)
    ]
    retriever = DummyRetriever()
    retriever.index(pages)
    results = retriever.retrieve("question", top_k=3)

    assert [result.page_index for result in results] == [0, 1, 2]
    assert [result.score for result in results] == [3.0, 2.0, 1.0]
    assert all(result.reason == "dummy retriever fallback" for result in results)
