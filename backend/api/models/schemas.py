"""
schemas.py — Pydantic models for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=2000, description="Legal question in French or English")
    top_k: int = Field(default=5, ge=1, le=20)
    treaty_filter: Optional[str] = Field(default=None, description="Filter to 'TEU' or 'TFEU'")
    stream: bool = Field(default=False, description="Stream the response token by token")


class Citation(BaseModel):
    article: str              # "Article 6 TEU"
    article_id: str
    treaty: str
    title: Optional[str]
    score: float
    sources: list[str]        # ["vector", "graph"]


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    context_used: int         # Number of context chunks used
    model: str


class ConversationTurn(BaseModel):
    role: str   # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=2000)
    history: list[ConversationTurn] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=20)
    stream: bool = Field(default=True)


class ArticleResponse(BaseModel):
    article_id: str
    number: str
    treaty: str
    text: str
    title_num: Optional[str]
    title_name: Optional[str]
    chapter_num: Optional[str]
    chapter_name: Optional[str]
    ex_article: Optional[str]
    page_start: int
    page_end: int


class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    properties: dict


class GraphEdge(BaseModel):
    source: str
    target: str
    relationship: str


class SubgraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    center_id: str


class HealthResponse(BaseModel):
    status: str
    neo4j: bool
    chromadb: bool
    ollama: bool
    ollama_model: str
    chunks_count: int
    articles_count: int
