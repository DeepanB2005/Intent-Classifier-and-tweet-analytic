
import faiss
import pandas as pd


class AppleRetriever:

    def __init__(
        self,
        rag_path,
        index_path,
        embedding_model,
        classifier
    ):

        self.rag_df = pd.read_csv(
            rag_path
        )

        self.index = faiss.read_index(
            index_path
        )

        self.embedding_model = embedding_model
        self.classifier = classifier

    def retrieve(
        self,
        query,
        predicted_intent,
        initial_k=20,
        final_k=5,
        intent_bonus=0.10
    ):

        # --------------------------------
        # Query embedding
        # --------------------------------

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        # --------------------------------
        # Semantic retrieval
        # --------------------------------

        scores, indices = self.index.search(
            query_embedding,
            initial_k
        )

        candidates = self.rag_df.iloc[
            indices[0]
        ].copy()

        candidates["semantic_score"] = scores[0]

        # --------------------------------
        # Intent-aware reranking
        # --------------------------------

        candidates["intent_match"] = (
            candidates["intent"] == predicted_intent
        ).astype(int)

        candidates["rerank_score"] = (
            candidates["semantic_score"]
            + intent_bonus * candidates["intent_match"]
        )

        candidates = candidates.sort_values(
            "rerank_score",
            ascending=False
        )

        return {
    "intent": predicted_intent,
    "results": candidates.head(final_k)
}
