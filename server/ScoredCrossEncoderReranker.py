from __future__ import annotations

import operator
from typing import Optional, Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document

from langchain.retrievers.document_compressors.cross_encoder import BaseCrossEncoder

from sentence_transformers import SentenceTransformer, util

class ScoredSBERTReRanker(BaseDocumentCompressor):
    """Document compressor that uses Sentence-BERT for reranking."""

    model_name: str
    """Name of the SBERT model."""
    top_n: int = 3
    """Number of top documents to return."""

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    def _init_(self, model_name: str, top_n: int = 3):
        self.model = SentenceTransformer(model_name)
        self.top_n = top_n

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Rerank documents using SBERT.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of top-N ranked documents.
        """
        # Encode the query and document contents
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        doc_embeddings = self.model.encode([doc.page_content for doc in documents], convert_to_tensor=True)

        # Compute cosine similarity scores
        scores = util.cos_sim(query_embedding, doc_embeddings).squeeze(0).cpu().tolist()

        # Pair documents with scores
        docs_with_scores = list(zip(documents, scores))

        # Sort documents by score
        ranked_docs = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)

        # Return top-N documents with updated metadata
        return [
            doc.copy(update={"metadata": {**doc.metadata, "relevance_score": score}})
            for doc, score in ranked_docs[:self.top_n]]


class ScoredCrossEncoderReranker(BaseDocumentCompressor):
    """Document compressor that uses CrossEncoder for reranking."""

    model: BaseCrossEncoder
    """CrossEncoder model to use for scoring similarity
      between the query and documents."""
    top_n: int = 3
    """Number of documents to return."""

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Rerank documents using CrossEncoder.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        scores = self.model.score([(query, doc.page_content) for doc in documents])
        docs_with_scores = list(zip(documents, scores))
        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
        return [doc.copy(update={"metadata": {**doc.metadata, "relevance_score": score}}) for doc, score in result[:self.top_n]]