import re
import statistics
from pathlib import Path

import chromadb
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MIN_IMG_AREA = 4000      # pt² — skip tiny decorative icons/bullets
CAPTION_BELOW_PTS = 60  # how far below an image to search for its caption


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text):
    text = text.replace('ﬁ','fi').replace('ﬂ','fl').replace('ﬀ','ff').replace('ﬃ','ffi').replace('ﬄ','ffl')
    text = re.sub(r'-\n([a-z])', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _table_to_markdown(rows):
    """Convert a pdfplumber table (list-of-lists) into a Markdown table string."""
    if not rows or not rows[0]:
        return ""
    header = [str(c or "").strip() for c in rows[0]]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows[1:]:
        lines.append("| " + " | ".join(str(c or "").strip() for c in row) + " |")
    return "\n".join(lines)


def _in_any_bbox(x, y, bboxes, margin=2.0):
    """True if point (x, y) falls inside any (x0, top, x1, bottom) bbox."""
    for x0, top, x1, bottom in bboxes:
        if x0 - margin <= x <= x1 + margin and top - margin <= y <= bottom + margin:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Per-page element extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_page_elements(pdf_path, page_num):
    """
    Extract all logical elements from one PDF page as a list of dicts:
      {"type": "table"|"figure"|"narrative", "content": str, ...metadata}

    Tables and figures are kept atomic (never split later).
    Narrative text has table/image/caption regions excluded before being returned.
    """
    elements = []

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num - 1]
        words = page.extract_words(extra_attrs=["size"])

        # ── 1. Tables ─────────────────────────────────────────────────────────
        table_bboxes = []
        for t in page.find_tables():
            bbox = t.bbox  # (x0, top, x1, bottom) in pdfplumber coords
            table_bboxes.append(bbox)
            md = _table_to_markdown(t.extract())
            if md:
                elements.append({"type": "table", "content": md, "bbox": bbox})

        # ── 2. Figures ────────────────────────────────────────────────────────
        image_bboxes = []
        for img in page.images:
            # Skip tiny decorative elements (bullets, icons, borders)
            if img["width"] * img["height"] < MIN_IMG_AREA:
                continue
            bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
            image_bboxes.append(bbox)
            ix0, _, ix1, ibottom = bbox

            # Caption = words directly below the image, outside any table region
            cap_words = [
                w for w in words
                if not _in_any_bbox(w["x0"], w["top"], table_bboxes)
                and ix0 - 20 <= w["x0"] <= ix1 + 20
                and ibottom <= w["top"] <= ibottom + CAPTION_BELOW_PTS
            ]
            caption = " ".join(w["text"] for w in cap_words).strip()
            content = caption or f"[Figure on page {page_num} — no caption detected]"

            elements.append({
                "type": "figure",
                "content": content,
                "bbox": bbox,
                "caption": caption,
            })

        # ── 3. Narrative text ─────────────────────────────────────────────────
        # Exclude: table regions + image regions + caption strips below images
        caption_strips = [
            (ix0 - 20, ibottom, ix1 + 20, ibottom + CAPTION_BELOW_PTS)
            for ix0, _, ix1, ibottom in image_bboxes
        ]
        excluded = table_bboxes + image_bboxes + caption_strips

        # Section headings: words with font size >= 1.2× the page median
        sizes = [w["size"] for w in words if w.get("size")]
        size_cutoff = (statistics.median(sizes) * 1.2) if len(sizes) >= 2 else 14.0

        heading_words, body_words = [], []
        for w in words:
            if _in_any_bbox(w["x0"], w["top"], excluded):
                continue
            if w.get("size", 0) >= size_cutoff:
                heading_words.append(w)
            else:
                body_words.append(w)

        section_heading = " ".join(w["text"] for w in heading_words).strip()
        narrative = clean_text(" ".join(w["text"] for w in body_words))

        if narrative:
            elements.append({
                "type": "narrative",
                "content": narrative,
                "section_heading": section_heading,
            })

    return elements


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_elements(elements, source, page_num, chunk_size=800, chunk_overlap=150):
    """
    Convert page elements into (documents, ids, metadatas) ready for ChromaDB upsert.

    Chunking rules:
    - table     → single atomic chunk (splitting a table breaks column relationships)
    - figure    → single atomic chunk (caption is the only text we have for it)
    - narrative → RecursiveCharacterTextSplitter; each sub-chunk is prefixed with
                  its section heading so retrieval context is never lost
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents, ids, metadatas = [], [], []
    counters = {"table": 0, "figure": 0, "narrative": 0}

    for el in elements:
        etype = el["type"]
        idx = counters[etype]
        base_meta = {"source": source, "page": page_num, "element_type": etype}

        if etype == "table":
            documents.append(el["content"])
            ids.append(f"{source}_p{page_num}_table{idx}")
            metadatas.append({**base_meta, "chunk_index": idx})
            counters["table"] += 1

        elif etype == "figure":
            documents.append(el["content"])
            ids.append(f"{source}_p{page_num}_figure{idx}")
            metadatas.append({**base_meta, "caption": el.get("caption", ""), "chunk_index": idx})
            counters["figure"] += 1

        elif etype == "narrative":
            section = el.get("section_heading", "")
            for ci, chunk in enumerate(splitter.split_text(el["content"])):
                # Prefix keeps section context intact after retrieval
                full_chunk = f"[Section: {section}]\n{chunk}" if section else chunk
                documents.append(full_chunk)
                ids.append(f"{source}_p{page_num}_text{idx}_c{ci}")
                metadatas.append({**base_meta, "section_heading": section, "chunk_index": idx * 1000 + ci})
            counters["narrative"] += 1

    return documents, ids, metadatas


# ─────────────────────────────────────────────────────────────────────────────
# Folder ingestion
# ─────────────────────────────────────────────────────────────────────────────

def ingest_folder(folder_path, collection):
    stats = {"files": 0, "pages": 0, "chunks": 0, "tables": 0, "figures": 0}

    for pdf_path in Path(folder_path).glob("*.pdf"):
        source = pdf_path.name
        print(f"\nProcessing: {source}")

        with pdfplumber.open(str(pdf_path)) as pdf:
            num_pages = len(pdf.pages)

        all_docs, all_ids, all_metas = [], [], []

        for page_num in range(1, num_pages + 1):
            print(f"  Page {page_num}/{num_pages} ...", end="\r", flush=True)
            elements = extract_page_elements(str(pdf_path), page_num)
            docs, ids, metas = chunk_elements(elements, source, page_num)
            all_docs.extend(docs)
            all_ids.extend(ids)
            all_metas.extend(metas)

            stats["tables"] += sum(1 for e in elements if e["type"] == "table")
            stats["figures"] += sum(1 for e in elements if e["type"] == "figure")

        if all_docs:
            collection.upsert(documents=all_docs, ids=all_ids, metadatas=all_metas)

        t = sum(1 for m in all_metas if m["element_type"] == "table")
        f = sum(1 for m in all_metas if m["element_type"] == "figure")
        n = sum(1 for m in all_metas if m["element_type"] == "narrative")
        print(f"\n  {source}: {len(all_docs)} chunks  (tables={t}, figures={f}, narrative={n})")

        stats["files"] += 1
        stats["pages"] += num_pages
        stats["chunks"] += len(all_docs)

    print(
        f"\nDone. {stats['files']} files · {stats['pages']} pages · "
        f"{stats['chunks']} chunks  (tables={stats['tables']}, figures={stats['figures']})"
    )
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    client = chromadb.PersistentClient(path="./chroma_biology")
    collection = client.get_or_create_collection(
        name="biology_knowledge_base",
        metadata={"hnsw:space": "cosine"},
    )

    ingest_folder("./BIOLOGY", collection)
    print(f"\nCollection size: {collection.count()} chunks")

    results = collection.query(
        query_texts=["What are chlorophiles?"],
        n_results=3,
    )
    for doc_id, doc, dist in zip(results["ids"][0], results["documents"][0], results["distances"][0]):
        print(f"\n--- {doc_id} (dist={dist:.3f}) ---")
        print(doc[:300])
