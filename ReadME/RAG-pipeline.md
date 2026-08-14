# Veloitt RAG Assistant

A lightweight Retrieval-Augmented Generation (RAG) assistant for answering grounded questions about Veloitt and its eTimeTracker SaaS platform.

The system turns structured website knowledge into a conversational assistant that retrieves relevant product information before generating an answer. Its design prioritizes factual responses, low infrastructure cost, maintainability, and deployment on a CPU-only Ubuntu server with a strict memory budget.

---

## What This Project Does

The assistant helps users ask natural-language questions such as:

- How does attendance processing work?
- Does the platform support field or distributed teams?
- Which product capabilities are relevant for a growing business?
- What information is available about plans, pricing, or commercial terms?
- How do biometric devices, attendance corrections, and field workflows work?

Instead of relying only on an LLM's general knowledge, the assistant first searches a curated Veloitt knowledge base. It then gives the language model only the most relevant sections as context.

This makes responses more accurate, more relevant to the product, and less likely to invent unsupported claims.

---

## High-Level Architecture

```text
Structured Markdown documents
        |
        v
Structure-aware chunking + metadata extraction
        |
        v
Embedding generation
        |
        v
Qdrant vector database
        |
        v
User question
        |
        v
Query embedding + semantic retrieval
        |
        v
Relevant chunks selected as context
        |
        v
Qwen 0.6B generates a grounded response
        |
        v
Frontend displays answer and supporting sources
```

The system has two major paths:

1. **Offline ingestion path**  
   Documents are prepared, chunked, embedded, and indexed once.

2. **Online query path**  
   A user asks a question, the system retrieves relevant knowledge, and the LLM generates a grounded response.

---

## Why RAG Instead of a Standalone LLM?

A standalone language model may produce fluent answers, but it does not inherently know the latest Veloitt product content, feature details, policies, or commercial information.

RAG solves this by supplying relevant company knowledge at answer time.

| Need | How the RAG system solves it |
|---|---|
| Product-specific answers | Searches the Veloitt knowledge base before generating |
| Reduced hallucinations | Instructs the model to answer only from retrieved context |
| Easier content updates | Update Markdown documents and re-index affected chunks |
| Lower infrastructure cost | Uses a compact LLM and lightweight retrieval stack |
| Better traceability | Returns supporting retrieved chunks with the final answer |
| Safer handling of unknowns | Can state that the knowledge base does not contain enough evidence |

---

## Knowledge Base Design

The source knowledge is stored as structured Markdown documents rather than unstructured scraped text.

The current corpus is organized into intent-based content spheres:

| Document | Purpose |
|---|---|
| `product.md` | Product capabilities, workflows, modules, and features |
| `buyer-fit.md` | Suitable business types, workforce scenarios, and use cases |
| `commercial.md` | Plans, pricing-related content, commercial terms, and buying information |

This organization helps the system distinguish between:

- “What does the product do?”
- “Is this suitable for my organization?”
- “What commercial or plan information is available?”

It also keeps the corpus easier for humans to review and maintain.

---

## Markdown Structure

Each source document uses a predictable hierarchy:

```md
***
intent_sphere: product
title: Product Capabilities
***

# Product

## Attendance Management

### Attendance Processing

Content explaining the feature.

#### Supported Workflows

Additional details, examples, or conditions.
```

The Markdown hierarchy is meaningful. It is not just formatting.

- YAML front matter stores document-level context.
- `#` and `##` headings provide high-level topic structure.
- `###` sections become the primary retrieval units.
- `####` subsections remain attached to their parent section where possible.

This keeps chunks coherent and prevents unrelated topics from being mixed together.

---

## Chunking Strategy

The project uses **structure-aware chunking** rather than blindly splitting text every fixed number of tokens.

### Default Chunking Rule

One primary chunk is created for each `###` section, while its `####` subsections remain part of the same chunk.

For example:

```text
Product
  └── Attendance Management
        └── Attendance Processing
              └── Biometric Synchronization
              └── Attendance Corrections
              └── Device Health Monitoring
```

The system can create one chunk for `Attendance Processing` containing the closely related child details.

### Why This Works

This approach is useful because SaaS documentation is naturally organized by features and workflows.

It helps retrieval return complete concepts instead of fragments such as:

```text
"supports biometric devices"
```

without the related explanation about synchronization, corrections, or monitoring.

### Chunk Size Target

The preferred target is approximately:

```text
250–700 tokens per chunk
```

If a section is too large, it can be split at natural paragraph or subsection boundaries. Small overlap is used only when needed to avoid losing meaning across a split.

---

## Dynamic Metadata

Metadata is generated automatically during ingestion instead of being manually repeated throughout every document.

Each chunk can include information such as:

```json
{
  "intent_sphere": "product",
  "chunk_type": "section",
  "section_path": "Product > Attendance Management > Attendance Processing",
  "category": "attendance",
  "topic": "attendance_processing",
  "source_document": "product.md",
  "content_hash": "unique-content-version-hash"
}
```

### Why Metadata Matters

Metadata improves retrieval quality and makes debugging easier.

For example, a question about pricing should prioritize commercial content, while a question about geofencing should prioritize feature and buyer-fit content.

Metadata also enables future improvements such as:

- Filtering results by content sphere
- Re-indexing only changed chunks
- Showing meaningful source labels in the frontend
- Measuring which document areas are most frequently retrieved
- Detecting gaps in the knowledge base

---

## Embedding and Indexing

Each chunk is converted into an embedding: a numerical representation of its meaning.

Similar meanings are placed near each other in vector space, which allows the system to retrieve relevant content even when the user's wording differs from the documentation.

For example:

| User wording | Relevant source wording |
|---|---|
| “Can my field team check in from customer locations?” | “Field visit workflow and geofenced check-ins” |
| “Can HR fix missed punches?” | “Attendance override and correction workflows” |
| “How does biometric attendance sync?” | “Biometric synchronization and device monitoring” |

The generated embeddings and metadata are stored in **Qdrant**, a vector database designed for similarity search and metadata filtering. The current design uses Qdrant because it supports persistent storage, vector search, payload metadata, and memory-saving options for constrained deployments. [page:1]

---

## Retrieval Flow

When a user submits a question, the application follows this flow:

```text
1. Receive user question
2. Convert question into an embedding
3. Search Qdrant for the most similar chunks
4. Retrieve the top relevant chunks and metadata
5. Build a concise grounded context
6. Send context plus question to the LLM
7. Return answer and supporting sources
```

The retrieval layer is responsible for finding evidence. The LLM is responsible for turning that evidence into a readable answer.

This separation is important: the model should not be expected to remember company facts that are already available in the knowledge base.

---

## Generation Model

The project uses **Qwen 0.6B** as the generation model.

This choice supports the project constraints:

- CPU-only deployment
- No CUDA requirement
- Approximately 2 GB total memory budget
- Faster startup and inference than larger models
- Enough capability for short, grounded SaaS question-answering

The model is intentionally not asked to perform unrestricted, long-form reasoning. It receives a small number of highly relevant retrieved chunks and is instructed to answer only from that context. [page:1]

---

## Grounding Policy

The assistant follows a context-first answer policy.

### Expected Behavior

- Answer using the retrieved Veloitt knowledge only
- Keep answers direct and concise
- Avoid claiming unsupported features or policies
- State uncertainty when the evidence is insufficient
- Return the supporting retrieved sections with the response

### Example

**Question**

```text
Does Veloitt support field sales attendance tracking?
```

**Good response style**

```text
The available knowledge supports field and distributed workforce workflows,
including field visits and geofenced check-ins. However, field-sales-specific
attendance tracking is not explicitly stated in the current knowledge base.
```

This is better than confidently claiming a feature that is only indirectly suggested.

---

## API Layer

The backend exposes a FastAPI-based query service.

Initial retrieval endpoint:

```text
POST /query
```

Example request:

```json
{
  "question": "How does attendance processing work?",
  "limit": 4
}
```

Example response shape:

```json
{
  "question": "How does attendance processing work?",
  "answer": "Veloitt processes attendance using biometric synchronization...",
  "hits": [
    {
      "score": 0.84,
      "section_path": "Product > Attendance Management > Attendance Processing",
      "content": "..."
    }
  ]
}
```

The `hits` field is retained so the frontend can show supporting source cards and users can inspect where the answer came from.

---

## Streaming Responses

The planned conversational endpoint is:

```text
POST /query/stream
```

The response will stream generated tokens in real time.

```text
Browser
   |
   v
Next.js frontend
   |
   v
FastAPI streaming endpoint
   |
   v
Qdrant retrieval
   |
   v
Qwen 0.6B via Ollama
   |
   v
Stream answer back to the browser
```

Streaming is useful because users can begin reading immediately rather than waiting for the complete response.

The frontend will use HTTP streaming through the browser `ReadableStream` API. WebSockets are unnecessary for this stage because the communication is primarily one-way: the server streams the answer to the user.

---

## Frontend Plan

The frontend will be built as a clean single-page assistant interface.

Recommended stack:

```text
Next.js App Router
TypeScript
React
Tailwind CSS
FastAPI backend
Ollama for local Qwen inference
Qdrant for retrieval
```

### Initial UI Components

- Product header with assistant status
- Chat conversation area
- User input box with submit and Enter-to-send support
- Live streaming assistant response
- Loading state while retrieval begins
- Stop-generation button
- Error and empty states
- Source panel showing retrieved sections
- Small metadata badges such as section path or similarity score

### UI Layout

```text
---------------------------------------------------------
| Veloitt Knowledge Assistant          Model: Qwen 0.6B |
---------------------------------------------------------
|                                                       |
|                    Chat Area                          |
|                                                       |
| User: How does attendance processing work?            |
|                                                       |
| Assistant: Veloitt processes attendance through ...   |
|                                                       |
---------------------------------------------------------
| Ask about Veloitt features, workflows, or plans  Send |
---------------------------------------------------------
| Supporting Sources                                    |
| Product > Attendance > Attendance Processing           |
| Product > Biometric Devices > Device Monitoring        |
---------------------------------------------------------
```

The UI should remain focused on answer quality and traceability before adding authentication, saved conversations, analytics, or advanced personalization.

---

## Resource-Constrained 