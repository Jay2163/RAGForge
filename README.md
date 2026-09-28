# RAGForge

### Engineering Knowledge Intelligence Platform

RAGForge is a production-oriented **Retrieval-Augmented Generation (RAG)** platform for retrieving and answering questions from engineering and technical documentation.

The system is designed around a retrieval-first architecture where documents are parsed, chunked, enriched with metadata, and eventually made searchable through semantic and lexical retrieval before being provided to an LLM for grounded response generation.

The project focuses on understanding and implementing the core components of a RAG system rather than treating RAG as a black-box abstraction.

---

## Problem

Engineering knowledge is often distributed across:

* API documentation
* Database documentation
* Troubleshooting guides
* Deployment runbooks
* Incident reports
* Internal FAQs
* Developer documentation
* Architecture documents

Traditional keyword search can miss semantically related information, while providing entire documents to an LLM can increase context size and introduce irrelevant information.

RAGForge addresses this by retrieving relevant evidence from the knowledge base before generating an answer.

---

## Architecture

```text
                           Client
                             │
                             ▼
                        ┌─────────┐
                        │ FastAPI │
                        └────┬────┘
                             │
                       Document API
                             │
                             ▼
                       ┌───────────┐
                       │ Ingestion │
                       └─────┬─────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
                 Parsing           Chunking
                    │                 │
                    └────────┬────────┘
                             ▼
                         Metadata
                             │
                             ▼
                        PostgreSQL
                             │
                      ┌──────┴──────┐
                      ▼             ▼
                 Embeddings    Lexical Index
                      │             │
                      ▼             ▼
                 Vector Search   Text Search
                      │             │
                      └──────┬──────┘
                             ▼
                     Hybrid Retrieval
                             │
                             ▼
                            RRF
                             │
                             ▼
                         Reranking
                             │
                             ▼
                       Relevant Context
                             │
                             ▼
                            LLM
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
                 Answer           Citations
```

---

## Document Processing

The document-processing pipeline is designed as separate stages:

```text
Document
   │
   ▼
Parser
   │
   ▼
Structured Document
   │
   ▼
Chunker
   │
   ▼
Document Chunks
   │
   ▼
Embeddings
   │
   ▼
Vector Storage
```

### Parser

The parser extracts usable text from supported document formats while preserving useful source information such as page boundaries.

Currently supported:

* TXT
* Markdown
* PDF

PDF extraction is implemented using `pypdf`.

### Chunker

The chunker converts extracted document content into smaller retrieval units.

The current chunking strategy includes:

* Paragraph-aware splitting
* Sentence-aware fallback splitting
* Word-based fallback splitting
* Word-boundary overlap
* Global chunk indexing
* Page metadata

Chunking is treated as a retrieval problem because the chunk becomes the fundamental unit that the retrieval system can discover.

---

## Design Decisions

### Parser and chunker separation

The parser answers:

> What information exists in the source document?

The chunker answers:

> How should that information be divided into retrieval units?

Keeping these responsibilities separate allows document extraction and retrieval optimization to evolve independently.

### Metadata preservation

Chunks retain their relationship to the source document.

Current metadata includes:

```text
document
chunk_index
page_number
content
```

This provides the foundation for source attribution and citation generation.

### Retrieval-first architecture

The LLM is not treated as the source of truth.

The intended pipeline retrieves relevant evidence first and provides that evidence to the generation layer.

### Hybrid retrieval

Different retrieval methods solve different problems.

Semantic retrieval is useful for conceptual and paraphrased questions.

Lexical retrieval is useful for exact technical information such as:

```text
error codes
API endpoints
function names
configuration keys
version numbers
SQL keywords
```

The system is designed to combine both approaches.

---

## Technology Stack

| Area             | Technology                  |
| ---------------- | --------------------------- |
| Language         | Python                      |
| API              | FastAPI                     |
| ORM              | SQLAlchemy                  |
| Database         | PostgreSQL                  |
| Migrations       | Alembic                     |
| Document parsing | pypdf                       |
| Configuration    | Pydantic Settings           |
| Testing          | pytest                      |
| Vector search    | pgvector, ChromaDB          |
| Lexical search   | PostgreSQL tsvector, BM25   |
| Embedding Model  | all-MiniLM-L6-v2 (384-dim)  |
| Retrieval        | Dense, Lexical, Hybrid (RRF)|
| Generation       | Google Gemini (LLM)         |
| Evaluation       | Hit@k, MRR, NDCG, RAG Triad |

---

## RAG Evaluation & Benchmarking

Evaluating a RAG pipeline is split into two distinct stages: **Retrieval Evaluation** (evaluating evidence discovery) and **Generation Evaluation** (evaluating response fidelity and groundedness).

### 1. Evaluation Metrics Defined

#### A. Retrieval Quality Metrics
* **Hit@K**: Measures if *at least one* relevant document chunk appears in the top $K$ retrieved results.
  $$\text{Hit@K} = \begin{cases} 1 & \text{if } \exists d \in \text{Top-}K \text{ where } d \in \text{Relevant} \\ 0 & \text{otherwise} \end{cases}$$
* **Mean Reciprocal Rank (MRR)**: Measures how high the *first relevant chunk* is placed. A score of $1.0$ means the top result is always relevant.
  $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
* **NDCG@K (Normalized Discounted Cumulative Gain)**: Measures ranking quality by penalizing relevant documents that appear lower in the result list (using logarithmic discounting).
  $$\text{DCG@K} = \sum_{i=1}^K \frac{\text{rel}_i}{\log_2(i + 1)}, \quad \text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$
* **Precision@K & Recall@K**:
  * **Precision@K**: What proportion of retrieved chunks are actually relevant? $\frac{|\text{Retrieved} \cap \text{Relevant}|}{K}$
  * **Recall@K**: What proportion of all ground-truth relevant chunks were found? $\frac{|\text{Retrieved} \cap \text{Relevant}|}{|\text{Total Relevant}|}$

#### B. End-to-End Generation Metrics (RAG Triad)
* **Faithfulness / Groundedness**: Evaluates whether every claim in the LLM's answer is strictly supported by the retrieved context chunks (zero hallucinations).
* **Answer Relevance**: Measures whether the generated answer directly addresses the user's question without extraneous drift.
* **Context Precision & Recall**: Verifies if the retrieved context contains sufficient signal with minimal noise.

---

### 2. Baseline Benchmark Results (PostgreSQL pgvector vs Lexical vs Hybrid)

Evaluated on *Building Python Web APIs with FastAPI* (322 chunks indexed, 20 comprehensive ground-truth test queries covering routing, validation, security, database integration, CORS, and testing):

| Retrieval Strategy | Hit@1 | Hit@3 | Hit@5 | Precision@5 | Recall@5 | MRR | NDCG@5 | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PGVector (Dense Semantic)** | **0.600** | **0.850** | **0.850** | **0.360** | **0.688** | **0.725** | **0.617** | ~199 ms |
| **Postgres tsvector (Lexical)** | 0.250 | 0.250 | 0.350 | 0.135 | 0.183 | 0.270 | 0.183 | **~55 ms** |
| **Postgres Hybrid (RRF $k=60$)** | 0.500 | 0.800 | 0.850 | 0.290 | 0.550 | 0.662 | 0.517 | ~248 ms |

#### Key Insights from 20-Question Baseline Analysis:
1. **PGVector (Dense Semantic)** achieves **0.725 MRR** and **85.0% Hit@5**, performing solidly on high-level conceptual questions and semantic phrasing.
2. **Postgres tsvector (Lexical)** achieves only **0.270 MRR** and **35.0% Hit@5** because standard SQL full-text search lacks true BM25 term saturation, term frequency damping, and struggles with vocabulary mismatch.
3. **Postgres Hybrid RRF** gets pulled down by `tsvector`'s low accuracy (MRR drops from $0.725 \rightarrow 0.662$), demonstrating that naive fusion with a weak lexical retriever hurts overall rank.
4. **Next Upgrade Path**: Introducing **Rank-BM25** (for true probabilistic keyword saturation) + **ChromaDB** + **Cross-Encoder Re-ranking** to elevate Hit@5, Precision@5, and MRR.

---

## Project Structure

```text
ragforge/
│
├── app/
│   ├── api/
│   │   └── health.py
│   │
│   ├── core/
│   │   └── config.py
│   │
│   ├── db/
│   │   └── database.py
│   │
│   ├── models/
│   │   ├── document.py
│   │   └── document_chunk.py
│   │
│   ├── services/
│   │   ├── chunking.py
│   │   └── document_parser.py
│   │
│   └── main.py
│
├── alembic/
│
├── tests/
│   ├── test_chunking.py
│   └── test_document_parser.py
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Local Development

### Requirements

* Python 3.12+
* PostgreSQL
* Git

### Setup

```bash
git clone <repository-url>
cd ragforge

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Create the environment configuration:

```bash
cp .env.example .env
```

Configure the PostgreSQL connection in `.env`.

Run database migrations:

```bash
alembic upgrade head
```

Start the API:

```bash
uvicorn app.main:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Testing

Run the complete test suite:

```bash
pytest -v
```

Run individual test modules:

```bash
pytest tests/test_document_parser.py -v
pytest tests/test_chunking.py -v
```

The test suite covers document parsing, supported file types, PDF extraction, chunk creation, overlap behavior, and chunk metadata.

---

## Engineering Approach

RAGForge is implemented incrementally, with each component designed to be independently understandable and testable.

The project starts with the underlying RAG primitives before introducing higher-level abstractions.

The core concepts being implemented include:

* Document parsing
* Chunking strategies
* Metadata and provenance
* Embedding generation
* Vector similarity
* Lexical retrieval
* Hybrid retrieval
* Ranking and reranking
* Citation grounding
* Retrieval evaluation

This approach keeps the retrieval pipeline explicit and makes the behavior of each stage observable and testable.
