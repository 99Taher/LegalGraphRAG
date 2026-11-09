"""
main.py — FastAPI application entry point.

Endpoints:
  GET  /health                     — System health check
  POST /api/query                  — Single legal question
  POST /api/chat                   — Conversational QA
  GET  /api/articles/{id}          — Get full article
  GET  /api/articles/search?q=...  — Search articles
  GET  /api/graph/{article_id}     — Get subgraph for visualization
  WS   /ws/chat                    — WebSocket streaming chat
"""
import sys
import json
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.config import settings
from backend.graph.neo4j_client import Neo4jClient, setup_schema
from backend.graph.graph_retriever import GraphRetriever
from backend.vector.vector_store import VectorStore
from backend.vector.embedder import Embedder
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.llm.llm_client import OllamaClient, PromptBuilder
from backend.api.models.schemas import (
    QueryRequest, QueryResponse, Citation,
    ChatRequest, ArticleResponse,
    SubgraphResponse, GraphNode, GraphEdge,
    HealthResponse,
)

# ─── Global Services (initialized at startup) ─────────────────────────────────

neo4j_client: Neo4jClient | None = None
graph_retriever: GraphRetriever | None = None
vector_store: VectorStore | None = None
embedder: Embedder | None = None
hybrid_retriever: HybridRetriever | None = None
llm: OllamaClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup, cleanup on shutdown."""
    global neo4j_client, graph_retriever, vector_store, embedder, hybrid_retriever, llm

    print("🚀 Starting Legal GraphRAG API...")

    # Neo4j
    neo4j_client = Neo4jClient(
        settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password
    )
    neo4j_ok = neo4j_client.connect()
    if neo4j_ok:
        setup_schema(neo4j_client)
    graph_retriever = GraphRetriever(neo4j_client)

    # Vector store
    vector_store = VectorStore(settings.chroma_persist_dir)

    # Embedder (lazy-loaded on first use)
    embedder = Embedder()

    # Hybrid retriever
    hybrid_retriever = HybridRetriever(
        graph_retriever=graph_retriever,
        vector_store=vector_store,
        embedder=embedder,
    )

    # LLM
    llm = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
    )

    print("✅ Legal GraphRAG API ready!")
    yield

    # Cleanup
    if neo4j_client:
        neo4j_client.close()
    print("👋 API shutdown complete")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Legal GraphRAG API",
    description="Intelligent legal document analysis using Neo4j + Vector Search + Ollama",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check status of all services."""
    neo4j_ok = False
    articles_count = 0
    try:
        result = neo4j_client.run("MATCH (n:Article) RETURN count(n) as c")
        neo4j_ok = True
        articles_count = result[0]["c"] if result else 0
    except Exception:
        pass

    chroma_ok = False
    chunks_count = 0
    try:
        chunks_count = vector_store.count()
        chroma_ok = True
    except Exception:
        pass

    ollama_ok = llm.is_available() if llm else False

    return HealthResponse(
        status="ok" if all([neo4j_ok, chroma_ok]) else "degraded",
        neo4j=neo4j_ok,
        chromadb=chroma_ok,
        ollama=ollama_ok,
        ollama_model=settings.ollama_model,
        chunks_count=chunks_count,
        articles_count=articles_count,
    )


# ─── Query ────────────────────────────────────────────────────────────────────

@app.post("/api/query", response_model=QueryResponse, tags=["RAG"])
async def query_legal(request: QueryRequest):
    """
    Answer a legal question using hybrid retrieval + LLM.
    Returns the answer with source citations.
    """
    # Retrieve
    results = hybrid_retriever.retrieve(
        query=request.question,
        top_k=request.top_k,
    )

    if not results:
        return QueryResponse(
            question=request.question,
            answer="Aucun article pertinent trouvé pour cette question.",
            citations=[],
            context_used=0,
            model=settings.ollama_model,
        )

    # Build context
    context = hybrid_retriever.format_context(results)
    prompt = PromptBuilder.build_qa_prompt(request.question, context)

    # Generate answer
    answer = await llm.agenerate(prompt)

    # Build citations
    citations = [
        Citation(**c) for c in hybrid_retriever.get_citations(results)
    ]

    return QueryResponse(
        question=request.question,
        answer=answer,
        citations=citations,
        context_used=len(results),
        model=settings.ollama_model,
    )


@app.post("/api/query/stream", tags=["RAG"])
async def query_legal_stream(request: QueryRequest):
    """Stream a legal answer token by token (SSE)."""
    results = hybrid_retriever.retrieve(query=request.question, top_k=request.top_k)
    context = hybrid_retriever.format_context(results)
    prompt = PromptBuilder.build_qa_prompt(request.question, context)
    citations = hybrid_retriever.get_citations(results)

    async def event_generator():
        # First emit citations as metadata
        yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"
        # Then stream the answer
        async for token in llm.agenerate_stream(prompt):
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── Chat ─────────────────────────────────────────────────────────────────────

@app.post("/api/chat", tags=["RAG"])
async def chat(request: ChatRequest):
    """Conversational legal QA with history."""
    results = hybrid_retriever.retrieve(query=request.question, top_k=request.top_k)
    context = hybrid_retriever.format_context(results)

    history = [{"role": t.role, "content": t.content} for t in request.history]
    prompt = PromptBuilder.build_conversation_prompt(request.question, context, history)

    answer = await llm.agenerate(prompt)
    citations = hybrid_retriever.get_citations(results)

    return {
        "answer": answer,
        "citations": citations,
        "model": settings.ollama_model,
    }


# ─── Articles ─────────────────────────────────────────────────────────────────

@app.get("/api/articles/{article_id}", response_model=ArticleResponse, tags=["Articles"])
async def get_article(article_id: str):
    """Get a specific article by ID (e.g. TEU-Art-6)."""
    result = neo4j_client.run(
        "MATCH (a:Article {id: $id}) RETURN a",
        {"id": article_id}
    )
    if not result:
        raise HTTPException(status_code=404, detail=f"Article '{article_id}' not found")

    a = result[0]["a"]
    return ArticleResponse(
        article_id=a.get("id", ""),
        number=a.get("number", ""),
        treaty=a.get("treaty", ""),
        text=a.get("text", ""),
        title_num=a.get("title_num"),
        title_name=a.get("title_name"),
        chapter_num=a.get("chapter_num"),
        chapter_name=a.get("chapter_name"),
        ex_article=a.get("ex_article"),
        page_start=a.get("page_start", 0),
        page_end=a.get("page_end", 0),
    )


@app.get("/api/articles", tags=["Articles"])
async def search_articles(
    q: str = Query(..., min_length=3),
    treaty: str | None = None,
    limit: int = Query(default=10, le=50),
):
    """Full-text search for articles."""
    results = graph_retriever.fulltext_search(q, limit=limit)
    if treaty:
        results = [r for r in results if r.treaty == treaty]
    return [
        {
            "article_id": r.article_id,
            "number": r.article_num,
            "treaty": r.treaty,
            "title": r.title_name,
            "preview": r.text[:300] + "..." if len(r.text) > 300 else r.text,
            "score": r.score,
        }
        for r in results
    ]


# ─── Graph Visualization ──────────────────────────────────────────────────────

@app.get("/api/graph/{article_id}", response_model=SubgraphResponse, tags=["Graph"])
async def get_subgraph(article_id: str):
    """
    Get the local subgraph around an article for visualization.
    Returns nodes and edges for D3/Cytoscape.
    """
    query = """
    MATCH (center:Article {id: $id})
    OPTIONAL MATCH (center)<-[:CONTAINS]-(parent)
    OPTIONAL MATCH (center)-[r1:CITES]->(cited:Article)
    OPTIONAL MATCH (citing:Article)-[r2:CITES]->(center)
    OPTIONAL MATCH (center)-[r3:REFERENCES]->(inst:Institution)
    RETURN center, parent,
           collect(DISTINCT {node: cited, rel: 'CITES', dir: 'out'}) as cited_nodes,
           collect(DISTINCT {node: citing, rel: 'CITES', dir: 'in'}) as citing_nodes,
           collect(DISTINCT {node: inst, rel: 'REFERENCES', dir: 'out'}) as institutions
    """
    results = neo4j_client.run(query, {"id": article_id})
    if not results:
        raise HTTPException(status_code=404, detail=f"Article '{article_id}' not found")

    r = results[0]
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    seen_ids = set()

    def add_node(node_data, node_type: str):
        if node_data is None:
            return
        nid = node_data.get("id", "")
        if nid and nid not in seen_ids:
            seen_ids.add(nid)
            label = (
                f"Art. {node_data['number']}" if node_type == "Article"
                else node_data.get("name", nid)
            )
            nodes.append(GraphNode(
                id=nid,
                label=label,
                type=node_type,
                properties={k: str(v) for k, v in node_data.items() if k != "text"},
            ))

    # Center article
    center = r["center"]
    add_node(center, "Article")

    # Parent (Title or Chapter)
    if r.get("parent"):
        add_node(r["parent"], "Title")
        edges.append(GraphEdge(
            source=r["parent"].get("id", ""),
            target=center.get("id", ""),
            relationship="CONTAINS",
        ))

    # Cited articles
    for item in r.get("cited_nodes", []):
        if item.get("node"):
            add_node(item["node"], "Article")
            edges.append(GraphEdge(
                source=center.get("id", ""),
                target=item["node"].get("id", ""),
                relationship="CITES",
            ))

    # Citing articles
    for item in r.get("citing_nodes", []):
        if item.get("node"):
            add_node(item["node"], "Article")
            edges.append(GraphEdge(
                source=item["node"].get("id", ""),
                target=center.get("id", ""),
                relationship="CITES",
            ))

    # Institutions
    for item in r.get("institutions", []):
        if item.get("node"):
            add_node(item["node"], "Institution")
            edges.append(GraphEdge(
                source=center.get("id", ""),
                target=item["node"].get("id", ""),
                relationship="REFERENCES",
            ))

    return SubgraphResponse(
        nodes=nodes,
        edges=edges,
        center_id=article_id,
    )


# ─── WebSocket Chat ───────────────────────────────────────────────────────────

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """Real-time streaming chat over WebSocket."""
    await websocket.accept()
    history = []

    try:
        while True:
            data = await websocket.receive_json()
            question = data.get("question", "")
            if not question:
                continue

            # Retrieve context
            results = hybrid_retriever.retrieve(query=question, top_k=5)
            context = hybrid_retriever.format_context(results)
            citations = hybrid_retriever.get_citations(results)
            prompt = PromptBuilder.build_conversation_prompt(question, context, history)

            # Send citations first
            await websocket.send_json({
                "type": "citations",
                "citations": citations,
            })

            # Stream answer
            full_answer = ""
            async for token in llm.agenerate_stream(prompt):
                full_answer += token
                await websocket.send_json({
                    "type": "token",
                    "content": token,
                })

            await websocket.send_json({"type": "done"})

            # Update history
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": full_answer})
            if len(history) > 10:
                history = history[-10:]

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
