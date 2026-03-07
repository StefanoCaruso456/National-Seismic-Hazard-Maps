from __future__ import annotations

from dataclasses import dataclass

from app.config import settings


@dataclass(frozen=True)
class PineconePricing:
    read_unit_per_million_usd: float
    write_unit_per_million_usd: float


@dataclass(frozen=True)
class TokenPricing:
    input_per_million_usd: float
    cached_input_per_million_usd: float = 0.0
    output_per_million_usd: float = 0.0


@dataclass(frozen=True)
class RerankPricing:
    input_per_million_usd: float


@dataclass(frozen=True)
class RagPricing:
    pinecone: PineconePricing
    llm: TokenPricing
    embedding: TokenPricing
    rerank: RerankPricing


RAG_PRICING = RagPricing(
    pinecone=PineconePricing(
        read_unit_per_million_usd=max(float(settings.pricing_pinecone_read_unit_per_million_usd), 0.0),
        write_unit_per_million_usd=max(float(settings.pricing_pinecone_write_unit_per_million_usd), 0.0),
    ),
    llm=TokenPricing(
        input_per_million_usd=max(float(settings.pricing_llm_input_per_million_usd), 0.0),
        cached_input_per_million_usd=max(float(settings.pricing_llm_cached_input_per_million_usd), 0.0),
        output_per_million_usd=max(float(settings.pricing_llm_output_per_million_usd), 0.0),
    ),
    embedding=TokenPricing(
        input_per_million_usd=max(float(settings.pricing_embedding_input_per_million_usd), 0.0),
    ),
    rerank=RerankPricing(
        input_per_million_usd=max(float(settings.pricing_rerank_input_per_million_usd), 0.0),
    ),
)
