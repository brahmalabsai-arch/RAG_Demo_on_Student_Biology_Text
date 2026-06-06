# RAG Demo on Student Biology Text

A small, self-contained Retrieval-Augmented Generation (RAG) pipeline that ingests school-level biology PDFs (class 7–8), stores them in a local vector database, and answers questions in a **grounded** way — meaning answers are constrained to the retrieved textbook content rather than the model's open-ended world knowledge.

The project has two stages:

1. **Ingestion** (`pdf_ingestion.py`) — parses PDFs into structured chunks (narrative text, tables, figures), embeds them, and stores them in a persistent ChromaDB collection.
2. **Answering** (`answering_rag.py`) — retrieves the most relevant chunks for a user question and uses a Groq-hosted LLM to generate an answer grounded in those chunks.

---

## Architecture at a glance

```
PDFs (./BIOLOGY)
      │
      ▼
 pdf_ingestion.py ──► structured elements ──► chunks ──► ChromaDB (./chroma_biology)
   • narrative text        (table / figure /        (atomic + split)
   • tables → Markdown       narrative)
   • figures → vision caption + description
      │
      ▼
 answering_rag.py ──► retrieve top-k chunks ──► Groq LLM ──► grounded answer
```

---

## Setup

```bash
# 1. Clone
git clone https://github.com/brahmalabsai-arch/RAG_Demo_on_Student_Biology_Text.git
cd RAG_Demo_on_Student_Biology_Text

# 2. Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install chromadb pdfplumber pymupdf anthropic groq langchain-text-splitters python-dotenv
```

Create a `.env` file in the project root for your API keys:

```
GROQ_API_KEY=your_groq_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here   # only needed if ingesting with vision enabled
```

> The Anthropic key is used during ingestion to describe figures with a vision model. If you ingest with `use_vision=False`, it isn't required.

## Usage

```bash
# Stage 1 — ingest PDFs into the vector store (run once, or when documents change)
python pdf_ingestion.py

# Stage 2 — ask questions
python answering_rag.py
```

Put your biology PDFs in the `./BIOLOGY` folder before ingesting.

---

## Design choices, benefits, and trade-offs

This section documents *why* the pipeline is built the way it is, since the interesting part of a RAG system is the engineering decisions, not the glue code.

### 1. Element-aware parsing (tables, figures, and narrative handled separately)

Instead of dumping each page into one flat text blob, ingestion classifies content into three element types and treats each on its own terms: tables are converted to Markdown, figures are turned into a caption-plus-description, and the remaining body text becomes narrative.

**Benefit:** Tables keep their row/column structure (a flattened table is nearly meaningless to a retriever), and diagrams — which carry a lot of meaning in a biology textbook — become searchable text instead of being lost.

**Trade-off:** It's more complex and slower than naive text extraction, and it depends on `pdfplumber`'s table/image detection, which can miss borderless tables or misjudge image regions on messy layouts.

### 2. Atomic chunks for tables and figures, recursive splitting for narrative

Tables and figures are stored as single, indivisible chunks. Narrative text is split with `RecursiveCharacterTextSplitter` (≈800 chars, 150 overlap).

**Benefit:** A table or figure description is never cut in half, so retrieval returns it whole and intelligible. The overlap on narrative chunks reduces the chance that an answer-bearing sentence is severed at a chunk boundary.

**Trade-off:** A very large table becomes one big chunk that can crowd the context window, and fixed-size narrative splitting can still cut mid-paragraph since it isn't semantically aware. The overlap also slightly inflates storage and the number of near-duplicate retrievals.

### 3. Vision-generated descriptions for figures

Each figure region is rendered to an image and passed to a Claude Haiku vision model, which produces a 3–5 sentence description anchored by the figure's caption.

**Benefit:** Diagrams (cell structure, photosynthesis flows, labeled organs) become retrievable and answerable — content that pure text extraction would discard entirely.

**Trade-off:** It adds an external API call per figure, which means cost, latency, and a dependency on network access during ingestion. Descriptions are also model-generated, so they can occasionally introduce small inaccuracies. Failures are handled gracefully by falling back to the caption alone.

### 4. Section headings prepended to narrative chunks

Each narrative sub-chunk is prefixed with its detected section heading (e.g. `[Section: Photosynthesis]`). Headings are detected heuristically by font size (words larger than ~1.2× the page's median font size).

**Benefit:** A chunk retains its topical context after retrieval, which improves both retrieval relevance and the LLM's ability to interpret an isolated fragment.

**Trade-off:** The font-size heuristic is fragile — it can misfire on PDFs with unusual typography, decorative fonts, or inconsistent heading sizes, occasionally tagging the wrong heading or none at all.

### 5. ChromaDB with cosine similarity, persisted locally

The vector store is a local `PersistentClient` using an HNSW index with cosine distance.

**Benefit:** Zero infrastructure — no separate database server to run. It persists to disk (`./chroma_biology`), so ingestion is a one-time cost, and HNSW gives fast approximate nearest-neighbour search. Cosine similarity is a sensible default for text embeddings.

**Trade-off:** A local, single-node store doesn't scale to very large corpora or concurrent multi-user serving, and HNSW is approximate, so recall isn't guaranteed to be perfect at the margins.

### 6. Groq for the answering LLM

Generation is served by a Groq-hosted model.

**Benefit:** Groq's inference is very fast and low-latency, which keeps the question-answer loop responsive — a noticeable quality-of-life improvement for an interactive demo.

**Trade-off:** It's another external dependency and API key to manage, the available model selection is whatever Groq hosts, and you're subject to its rate limits and availability.

### 7. Grounded answering

The answering stage feeds retrieved chunks to the LLM and constrains it to answer from that context rather than from open-ended prior knowledge.

**Benefit:** Answers stay faithful to the source textbook, which sharply reduces hallucination and keeps responses age-appropriate and curriculum-aligned. It also makes answers traceable back to specific pages/chunks.

**Trade-off:** If retrieval misses the relevant chunk, the model has nothing to work with and should decline rather than guess — so overall quality is capped by retrieval quality. Grounding is also only as strong as the prompt and the retrieved context; it constrains the model but doesn't make fabrication impossible.

---

## Repository structure

```
.
├── BIOLOGY/              # Source biology PDFs (input)
├── pdf_ingestion.py      # Stage 1: parse → chunk → embed → store in ChromaDB
├── answering_rag.py      # Stage 2: retrieve → ground → answer via Groq
├── chroma_biology/       # Persistent vector store (generated; gitignored)
└── README.md
```

> Note: `venv/`, `.env`, `__pycache__/`, and `chroma_biology/` should be listed in `.gitignore` so your environment, API keys, and generated database are not committed.

---

## Possible improvements

- Add citations to the answer output (page number + element type are already in chunk metadata).
- Use semantic / heading-based chunking instead of fixed character windows.
- Add a small evaluation set to measure retrieval recall and answer faithfulness.
- Expose the pipeline through a simple CLI or web UI.

---

## License

Add a license of your choice (e.g. MIT) if you intend others to reuse this.
