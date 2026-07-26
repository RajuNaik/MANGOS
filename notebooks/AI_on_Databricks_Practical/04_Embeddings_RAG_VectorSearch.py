# Databricks notebook source

# MAGIC %md
# MAGIC # 🔍 Notebook 04 — Embeddings, RAG & Vector Search
# MAGIC
# MAGIC **Handbook Sections Covered**: A6 (Foundation Models/LLMs), A7 (RAG), A8 (Embeddings & Vector Similarity), B1 (Unstructured Data), B2 (Delta as Source of Truth), B3 (Chunking), B5 (Vector Search), I5 (Hybrid Search)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Learning Objectives
# MAGIC 1. **Embeddings** — convert text to vectors, understand semantic similarity
# MAGIC 2. **Chunking** — split documents into retrievable pieces, compare strategies
# MAGIC 3. **Vector Index** — build a FAISS index (open-source equivalent of Databricks Vector Search)
# MAGIC 4. **RAG Pipeline** — full Retrieval-Augmented Generation from scratch
# MAGIC 5. **Hybrid Search** — combine semantic + keyword search for better retrieval
# MAGIC 6. **Delta Sync** — understand how Vector Search auto-syncs from Delta tables

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Setup

# COMMAND ----------

%pip install -q sentence-transformers faiss-cpu rank-bm25

# COMMAND ----------

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
import mlflow
import json
import warnings
warnings.filterwarnings('ignore')

mlflow.set_experiment("/Users/{}/AI_Handbook_04_RAG_VectorSearch".format(
    spark.sql("SELECT current_user()").collect()[0][0]
))

print("✅ Setup complete! Importing models...")

from sentence_transformers import SentenceTransformer
import faiss

# Load a small, free embedding model that runs on CPU
# This is the open-source equivalent of a Databricks Foundation Model embedding endpoint
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
EMBEDDING_DIM = 384  # This model produces 384-dimensional vectors

print(f"✅ Embedding model loaded: all-MiniLM-L6-v2")
print(f"   Embedding dimension: {EMBEDDING_DIM}")
print(f"   This runs locally on CPU — no API key needed!")
print(f"\n📌 On full Databricks, you'd use a Foundation Model API endpoint:")
print(f"   endpoint_name = 'databricks-bge-large-en'  (managed, pay-per-token)")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 1: Understanding Embeddings (Handbook A8)
# MAGIC
# MAGIC An **embedding** is a numeric vector (list of ~300-1000 numbers) representing the **meaning**
# MAGIC of a piece of text. Texts with **similar meaning** end up with vectors that are
# MAGIC mathematically **close** together, even if they share no words.

# COMMAND ----------

# ============================================================================
# EMBEDDINGS: See semantic similarity in action
# ============================================================================

# These sentences have similar MEANING but very different WORDS
demo_texts = [
    "How do I cancel my subscription?",       # Intent: cancel
    "I want to end my membership",             # Intent: cancel (different words!)
    "Please terminate my account",             # Intent: cancel (yet another way)
    "What's the weather like today?",          # Completely different topic
    "How do I upgrade my plan?",               # Related to account, but opposite intent
    "I need to reset my password",             # Account-related but different action
]

# Generate embeddings
demo_embeddings = embedding_model.encode(demo_texts)

print("=" * 70)
print("  EMBEDDINGS — Semantic similarity demo")
print("=" * 70)
print(f"\n  Each text → a vector of {demo_embeddings.shape[1]} numbers")
print(f"  Example (first 10 values of first text's embedding):")
print(f"  {demo_embeddings[0][:10].round(4)}")

# Compute pairwise cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(demo_embeddings)

print(f"\n  SIMILARITY MATRIX (cosine similarity, 0-1):")
print(f"  {'':50s}", end="")
for i in range(len(demo_texts)):
    print(f"  [{i}]", end="")
print()

for i, text in enumerate(demo_texts):
    print(f"  [{i}] {text[:48]:<50s}", end="")
    for j in range(len(demo_texts)):
        sim = similarity_matrix[i][j]
        marker = "█" if sim > 0.5 and i != j else " "
        print(f" {sim:.2f}", end="")
    print()

print(f"\n  💡 Key insight:")
print(f"     'cancel subscription' ↔ 'end membership' similarity: {similarity_matrix[0][1]:.3f} (HIGH — same meaning)")
print(f"     'cancel subscription' ↔ 'weather today' similarity:  {similarity_matrix[0][3]:.3f} (LOW — unrelated)")
print(f"     This is what makes SEMANTIC search work — searching by meaning, not keywords!")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 2: Building a Knowledge Base (Handbook B1, B2)
# MAGIC
# MAGIC A RAG system needs a **knowledge base** — the documents your AI will retrieve from.
# MAGIC On Databricks, this lives as governed Delta tables and/or Volumes (for raw files like PDFs).

# COMMAND ----------

# ============================================================================
# SYNTHETIC KNOWLEDGE BASE: Company documentation for a fictional SaaS company
# ============================================================================

knowledge_base = [
    {
        "doc_id": "DOC_001", "source": "help_center",
        "title": "How to Cancel Your Subscription",
        "content": """To cancel your subscription, follow these steps:
1. Log into your account at app.acmecorp.com
2. Navigate to Settings > Billing > Subscription
3. Click 'Cancel Subscription' at the bottom of the page
4. Select a cancellation reason from the dropdown
5. Confirm by clicking 'Yes, Cancel My Subscription'

Your subscription will remain active until the end of your current billing period. You will not be charged again after cancellation. Your data will be retained for 90 days after cancellation, during which time you can reactivate your account. After 90 days, your data will be permanently deleted.

If you're on an Enterprise plan, please contact your account manager or email enterprise-support@acmecorp.com to process the cancellation, as Enterprise plans require manual processing.

Refund Policy: If you cancel within the first 14 days of a new billing cycle, you are eligible for a prorated refund. Contact billing@acmecorp.com with your account ID to request a refund."""
    },
    {
        "doc_id": "DOC_002", "source": "help_center",
        "title": "Upgrading or Downgrading Your Plan",
        "content": """AcmeCorp offers four plans: Free, Basic ($9.99/month), Premium ($24.99/month), and Enterprise (custom pricing).

To change your plan:
1. Go to Settings > Billing > Change Plan
2. Select your desired plan
3. Review the price difference
4. Confirm the change

When upgrading: The price difference is prorated for the remainder of your current billing cycle. New features are available immediately.

When downgrading: The change takes effect at the start of your next billing cycle. You'll retain access to premium features until then. Note that downgrading may result in data limitations — Premium allows 100GB storage while Basic allows 10GB. If you exceed the new plan's limits, you'll need to reduce your usage before the downgrade takes effect.

Enterprise customers: Contact sales@acmecorp.com for custom pricing and feature negotiations."""
    },
    {
        "doc_id": "DOC_003", "source": "help_center",
        "title": "Resetting Your Password",
        "content": """If you've forgotten your password or need to reset it:

Method 1 — Email Reset:
1. Go to app.acmecorp.com/login
2. Click 'Forgot Password?'
3. Enter your registered email address
4. Check your inbox for a reset link (valid for 1 hour)
5. Click the link and set a new password

Password Requirements: Minimum 12 characters, at least one uppercase letter, one lowercase letter, one number, and one special character.

Method 2 — SMS Reset (if enabled):
1. Click 'Forgot Password?' > 'Use SMS Instead'
2. Enter your registered phone number
3. Enter the 6-digit verification code sent to your phone
4. Set a new password

If you don't receive the reset email within 5 minutes, check your spam/junk folder. If it's not there, contact support@acmecorp.com. For security reasons, we cannot reset passwords over the phone.

Two-Factor Authentication: If you have 2FA enabled, you'll need your authentication app or backup codes after resetting your password."""
    },
    {
        "doc_id": "DOC_004", "source": "api_docs",
        "title": "REST API Authentication",
        "content": """AcmeCorp API uses Bearer token authentication. All API requests must include an Authorization header.

Getting your API key:
1. Navigate to Settings > API > Generate Key
2. Choose key permissions (read-only, read-write, admin)
3. Set an expiration date (maximum 365 days)
4. Copy the key immediately — it won't be shown again

Usage:
  curl -H 'Authorization: Bearer YOUR_API_KEY' https://api.acmecorp.com/v2/data

Rate Limits:
- Free plan: 100 requests/hour
- Basic plan: 1,000 requests/hour
- Premium plan: 10,000 requests/hour
- Enterprise plan: Custom limits

If you exceed your rate limit, the API returns HTTP 429 (Too Many Requests) with a Retry-After header indicating when you can resume requests. Implement exponential backoff in your client code for production reliability.

API versioning: The current version is v2. The v1 API is deprecated and will be removed on December 31, 2024. Please migrate to v2 before that date."""
    },
    {
        "doc_id": "DOC_005", "source": "help_center",
        "title": "Data Export and Backup",
        "content": """You can export your data from AcmeCorp at any time.

Full Export:
1. Go to Settings > Data > Export
2. Select the data types to export (projects, files, settings, history)
3. Choose export format (JSON, CSV, or both)
4. Click 'Start Export' — you'll receive an email when the export is ready
5. Download the ZIP file from the provided link (available for 7 days)

Automated Backups:
- Premium and Enterprise plans include automatic daily backups
- Backups are retained for 30 days (Premium) or 90 days (Enterprise)
- To restore from a backup, go to Settings > Data > Backups > Restore

API-based Export:
Use GET /v2/export/full to initiate a programmatic export. The response includes a job_id to poll for completion status.

Data Portability: AcmeCorp supports the Data Transfer Project standard. You can transfer your data directly to supported services without downloading and re-uploading."""
    },
    {
        "doc_id": "DOC_006", "source": "help_center",
        "title": "Team and User Management",
        "content": """Manage team members and permissions from Settings > Team.

Adding Users:
1. Click 'Invite Member'
2. Enter their email address
3. Assign a role: Viewer (read-only), Editor (read-write), Admin (full control)
4. They'll receive an invitation email to join

Roles and Permissions:
- Viewer: Can view projects and data, cannot edit or delete
- Editor: Can create, edit, and delete their own content. Can edit shared projects.
- Admin: Full control including billing, user management, and settings
- Owner: Cannot be removed, can transfer ownership to another Admin

Groups: Organize users into groups for easier permission management. Assign permissions to a group rather than individual users. Users inherit the highest permission level from their groups.

SSO Integration: Enterprise plans support SAML-based Single Sign-On with Okta, Azure AD, OneLogin, and Google Workspace. Contact enterprise-support@acmecorp.com to configure SSO."""
    },
    {
        "doc_id": "DOC_007", "source": "security",
        "title": "Security and Compliance",
        "content": """AcmeCorp takes security seriously. Here's an overview of our security posture:

Data Encryption:
- At rest: AES-256 encryption for all stored data
- In transit: TLS 1.3 for all API and web traffic
- Customer-managed encryption keys available on Enterprise plan (BYOK)

Compliance Certifications:
- SOC 2 Type II (annually audited)
- GDPR compliant (EU data processing)
- HIPAA compliant (with signed BAA on Enterprise plan)
- ISO 27001 certified

Infrastructure:
- Hosted on AWS with multi-region redundancy
- 99.99% uptime SLA on Enterprise plan
- Daily automated backups with point-in-time recovery
- DDoS protection via AWS Shield Advanced

Incident Response: Security incidents are classified by severity (P1-P4). P1 incidents trigger immediate notification to affected customers within 24 hours. Our security team maintains a 24/7 on-call rotation.

Vulnerability Reporting: Report security vulnerabilities to security@acmecorp.com. We operate a bug bounty program — see https://acmecorp.com/security/bounty for details."""
    },
    {
        "doc_id": "DOC_008", "source": "help_center",
        "title": "Billing and Invoices",
        "content": """Billing FAQ:

Payment Methods: We accept Visa, Mastercard, American Express, and PayPal. Enterprise customers can pay via invoice (NET 30 terms).

Billing Cycle: Subscriptions are billed monthly or annually. Annual plans receive a 20% discount. Your billing date is the date you first subscribed.

Viewing Invoices:
1. Go to Settings > Billing > Invoice History
2. Each invoice is available as a downloadable PDF
3. Invoices include a detailed breakdown of charges

Failed Payments: If a payment fails, we retry 3 times over 7 days. Your account remains active during this period. After 3 failed attempts, your account is downgraded to the Free plan. Update your payment method to restore your subscription.

Taxes: Prices shown are exclusive of tax. Sales tax or VAT is calculated based on your billing address and added to the invoice. Tax-exempt organizations can upload their exemption certificate in Settings > Billing > Tax Exemption.

Currency: All prices are in USD. Charges on your statement may vary slightly due to currency conversion fees applied by your bank."""
    },
    {
        "doc_id": "DOC_009", "source": "help_center",
        "title": "Troubleshooting Common Issues",
        "content": """Common problems and their solutions:

Issue: App not loading / blank screen
Solution: Clear your browser cache and cookies, then try again. If the issue persists, try a different browser. Check our status page at status.acmecorp.com for any ongoing incidents.

Issue: File upload failing
Solution: Check that your file is under the size limit (50MB for Free/Basic, 500MB for Premium/Enterprise). Supported formats: PDF, DOCX, XLSX, CSV, JSON, PNG, JPG. If the file is valid, try disabling your browser extensions — ad blockers sometimes interfere with uploads.

Issue: Email notifications not arriving
Solution: Check your spam/junk folder. Add notifications@acmecorp.com to your contacts/whitelist. If using a corporate email, ask your IT team to whitelist our sending domain: mail.acmecorp.com.

Issue: Slow performance
Solution: If the app feels slow, check your internet connection speed. AcmeCorp requires at least 5 Mbps for optimal performance. If your connection is fine, the issue may be related to the size of your workspace — workspaces with over 10,000 items may experience some lag. Contact support for optimization recommendations.

Issue: Two-factor authentication locked out
Solution: Use one of your backup codes to log in. If you've lost your backup codes, contact support@acmecorp.com with your account email and a government-issued photo ID for identity verification."""
    },
    {
        "doc_id": "DOC_010", "source": "release_notes",
        "title": "Release Notes - Version 4.2 (July 2024)",
        "content": """What's New in AcmeCorp 4.2:

New Features:
- Real-time collaboration: Multiple users can now edit the same project simultaneously with live cursor tracking and instant sync
- AI-powered search: Natural language search across all your projects and files (powered by our new semantic search engine)
- Custom dashboards: Build personalized dashboards with drag-and-drop widgets, charts, and KPI cards
- Webhook integrations: Trigger external workflows when events occur in AcmeCorp (e.g., new file uploaded, project completed)

Improvements:
- 40% faster page load times across the application
- Redesigned mobile experience with offline support
- Enhanced CSV import with automatic column type detection
- Improved API response times (average 50ms reduction)

Bug Fixes:
- Fixed an issue where file previews didn't render for PDFs over 50 pages
- Resolved intermittent login failures when using SSO with Azure AD
- Fixed data export occasionally missing attachments
- Corrected timezone display issues in activity logs

Breaking Changes:
- API v1 endpoints are now deprecated. Please migrate to v2 before December 31, 2024
- The legacy 'Projects' view has been replaced with 'Workspaces'. Existing projects are automatically migrated."""
    },
]

# Save knowledge base as a Delta table (the governed source of truth)
kb_df = pd.DataFrame(knowledge_base)
spark_kb = spark.createDataFrame(kb_df)
spark_kb.write.format("delta").mode("overwrite").saveAsTable("default.knowledge_base")

print(f"✅ Knowledge base created: default.knowledge_base")
print(f"   {len(knowledge_base)} documents")
for doc in knowledge_base:
    print(f"   [{doc['doc_id']}] {doc['title']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 3: Document Chunking (Handbook B3)
# MAGIC
# MAGIC Documents are usually too long to embed as a single vector. **Chunking** splits them
# MAGIC into smaller pieces, each getting its own embedding.
# MAGIC
# MAGIC > **Chunk too small** → loses context around the answer
# MAGIC > **Chunk too large** → dilutes the embedding's specificity, wastes prompt space

# COMMAND ----------

# ============================================================================
# CHUNKING: Split documents into retrievable pieces
# ============================================================================

def chunk_text_fixed(text: str, chunk_size: int = 300, overlap: int = 50) -> list:
    """
    Fixed-size chunking with overlap.
    chunk_size: approximate number of characters per chunk
    overlap: number of characters that overlap between consecutive chunks
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Try to break at a sentence boundary
        if end < len(text):
            # Look for the last period, newline, or sentence end within a window
            for boundary_char in ['. ', '.\n', '\n\n', '\n']:
                boundary = text.rfind(boundary_char, start + chunk_size // 2, end + 50)
                if boundary != -1:
                    end = boundary + len(boundary_char)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks

def chunk_text_paragraph(text: str) -> list:
    """
    Paragraph-based chunking: split on double newlines.
    More natural boundaries but variable sizes.
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    # Merge very short paragraphs with the next one
    merged = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) < 400:
            current = (current + "\n\n" + p).strip()
        else:
            if current:
                merged.append(current)
            current = p
    if current:
        merged.append(current)
    return merged

# Demonstrate chunking on a long document
demo_doc = knowledge_base[0]
print(f"Document: '{demo_doc['title']}'")
print(f"Full length: {len(demo_doc['content'])} characters")
print()

# Strategy 1: Fixed-size chunks
fixed_chunks = chunk_text_fixed(demo_doc['content'], chunk_size=300, overlap=50)
print(f"Strategy 1 — Fixed-size (300 chars, 50 overlap): {len(fixed_chunks)} chunks")
for i, chunk in enumerate(fixed_chunks):
    print(f"  Chunk {i+1} ({len(chunk)} chars): {chunk[:80]}...")
print()

# Strategy 2: Paragraph-based chunks
para_chunks = chunk_text_paragraph(demo_doc['content'])
print(f"Strategy 2 — Paragraph-based: {len(para_chunks)} chunks")
for i, chunk in enumerate(para_chunks):
    print(f"  Chunk {i+1} ({len(chunk)} chars): {chunk[:80]}...")

# COMMAND ----------

# ============================================================================
# CHUNK ALL DOCUMENTS and store with metadata
# ============================================================================

all_chunks = []
for doc in knowledge_base:
    chunks = chunk_text_fixed(doc['content'], chunk_size=400, overlap=75)
    for i, chunk_text in enumerate(chunks):
        all_chunks.append({
            'chunk_id': f"{doc['doc_id']}_chunk_{i}",
            'doc_id': doc['doc_id'],
            'source': doc['source'],
            'title': doc['title'],
            'chunk_index': i,
            'chunk_text': chunk_text,
            'char_count': len(chunk_text),
        })

chunks_df = pd.DataFrame(all_chunks)

# Save chunks as Delta table
spark_chunks = spark.createDataFrame(chunks_df)
spark_chunks.write.format("delta").mode("overwrite").saveAsTable("default.knowledge_chunks")

print(f"✅ Chunked {len(knowledge_base)} documents into {len(all_chunks)} chunks")
print(f"   Avg chunk size: {chunks_df['char_count'].mean():.0f} chars")
print(f"   Min: {chunks_df['char_count'].min()} | Max: {chunks_df['char_count'].max()}")
print(f"\n   Saved as: default.knowledge_chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 4: Building a Vector Index (Handbook B5)
# MAGIC
# MAGIC A **vector index** stores embeddings and enables fast nearest-neighbor search.
# MAGIC
# MAGIC | CE (this notebook) | Full Databricks |
# MAGIC |---|---|
# MAGIC | FAISS (open-source, in-process) | Mosaic AI Vector Search (managed service) |
# MAGIC | Manual embed + insert | Auto-embeds from source Delta table |
# MAGIC | Manual rebuild on data change | **Auto-syncs** when source Delta table changes |

# COMMAND ----------

# ============================================================================
# VECTOR INDEX: Embed all chunks and build a FAISS index
# ============================================================================

# Step 1: Generate embeddings for all chunks
print("Generating embeddings for all chunks...")
chunk_texts = chunks_df['chunk_text'].tolist()
chunk_embeddings = embedding_model.encode(chunk_texts, show_progress_bar=True)

print(f"✅ Generated {len(chunk_embeddings)} embeddings, shape: {chunk_embeddings.shape}")

# Step 2: Build FAISS index
# FAISS = Facebook AI Similarity Search — the standard open-source vector index
index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner Product (cosine similarity for normalized vectors)

# Normalize embeddings for cosine similarity
faiss.normalize_L2(chunk_embeddings)
index.add(chunk_embeddings.astype(np.float32))

print(f"✅ FAISS index built: {index.ntotal} vectors indexed")
print(f"\n📌 On full Databricks, this is done with Vector Search:")
print(f"   vsc = VectorSearchClient()")
print(f"   vsc.create_delta_sync_index(")
print(f"       endpoint_name='my_vs_endpoint',")
print(f"       index_name='catalog.schema.knowledge_index',")
print(f"       source_table_name='catalog.schema.knowledge_chunks',")
print(f"       embedding_source_column='chunk_text',  # auto-embeds this column!")
print(f"       pipeline_type='TRIGGERED'  # auto-syncs when source table changes")
print(f"   )")

# COMMAND ----------

# ============================================================================
# SEMANTIC SEARCH: Query the vector index
# ============================================================================

def semantic_search(query: str, top_k: int = 5) -> list:
    """
    Search the FAISS index for the most semantically similar chunks.
    This is the open-source equivalent of Databricks Vector Search's similarity_search().
    """
    # Embed the query
    query_embedding = embedding_model.encode([query])
    faiss.normalize_L2(query_embedding)
    
    # Search the index
    scores, indices = index.search(query_embedding.astype(np.float32), top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        chunk = all_chunks[idx]
        results.append({
            'chunk_id': chunk['chunk_id'],
            'title': chunk['title'],
            'score': float(score),
            'chunk_text': chunk['chunk_text'],
        })
    return results

# Test searches
test_queries = [
    "How do I cancel my account?",
    "What encryption does the platform use?",
    "How to get an API key?",
    "Can I export my data?",
]

for query in test_queries:
    print(f"\n{'=' * 70}")
    print(f"  QUERY: \"{query}\"")
    print(f"{'=' * 70}")
    results = semantic_search(query, top_k=3)
    for i, r in enumerate(results):
        print(f"  #{i+1} [score: {r['score']:.4f}] {r['title']}")
        print(f"      {r['chunk_text'][:100]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 5: Hybrid Search (Handbook I5)
# MAGIC
# MAGIC Pure semantic search can miss **exact matches** (a specific error code, product SKU,
# MAGIC or acronym). **Hybrid search** combines:
# MAGIC - **Semantic** (embedding similarity) — finds conceptually related content
# MAGIC - **Keyword** (BM25 lexical) — finds exact term matches
# MAGIC
# MAGIC > Databricks Vector Search supports hybrid search natively.

# COMMAND ----------

# ============================================================================
# HYBRID SEARCH: Semantic + Keyword (BM25)
# ============================================================================
from rank_bm25 import BM25Okapi
import re

# Build BM25 index for keyword search
tokenized_chunks = [re.findall(r'\w+', chunk.lower()) for chunk in chunk_texts]
bm25 = BM25Okapi(tokenized_chunks)

def hybrid_search(query: str, top_k: int = 5, semantic_weight: float = 0.7) -> list:
    """
    Hybrid search: combine semantic (FAISS) and keyword (BM25) results.
    semantic_weight: 0.0 = pure keyword, 1.0 = pure semantic
    """
    # Semantic search
    query_emb = embedding_model.encode([query])
    faiss.normalize_L2(query_emb)
    sem_scores, sem_indices = index.search(query_emb.astype(np.float32), top_k * 2)
    
    # Keyword search
    tokenized_query = re.findall(r'\w+', query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
    
    # Normalize scores to [0, 1]
    sem_max = max(sem_scores[0]) if max(sem_scores[0]) > 0 else 1
    bm25_max = max(bm25_scores) if max(bm25_scores) > 0 else 1
    
    # Combine scores
    combined = {}
    for score, idx in zip(sem_scores[0], sem_indices[0]):
        combined[idx] = combined.get(idx, 0) + semantic_weight * (score / sem_max)
    
    for idx in bm25_top_indices:
        norm_score = bm25_scores[idx] / bm25_max
        combined[idx] = combined.get(idx, 0) + (1 - semantic_weight) * norm_score
    
    # Sort by combined score
    sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    results = []
    for idx, score in sorted_results:
        chunk = all_chunks[idx]
        results.append({
            'chunk_id': chunk['chunk_id'],
            'title': chunk['title'],
            'combined_score': round(score, 4),
            'chunk_text': chunk['chunk_text'],
        })
    return results

# Compare: Semantic-only vs Hybrid for an exact-match query
exact_query = "API v1 deprecated December 2024"
print("=" * 70)
print(f"  QUERY: \"{exact_query}\"")
print(f"  (This query has SPECIFIC terms a keyword search should nail)")
print("=" * 70)

print(f"\n  🔵 Semantic-only search:")
for i, r in enumerate(semantic_search(exact_query, top_k=3)):
    print(f"    #{i+1} [{r['score']:.3f}] {r['title']}: {r['chunk_text'][:70]}...")

print(f"\n  🟢 Hybrid search (70% semantic + 30% keyword):")
for i, r in enumerate(hybrid_search(exact_query, top_k=3)):
    print(f"    #{i+1} [{r['combined_score']:.3f}] {r['title']}: {r['chunk_text'][:70]}...")

print(f"\n💡 Hybrid search boosts results with exact keyword matches")
print(f"   (like 'v1', 'deprecated', 'December 2024') alongside semantic similarity")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 6: The Full RAG Pipeline (Handbook A7)
# MAGIC
# MAGIC Now we assemble the complete **Retrieval-Augmented Generation** pipeline:
# MAGIC 1. Take the user's question
# MAGIC 2. **SEARCH** the knowledge base (hybrid search) for relevant chunks
# MAGIC 3. **STUFF** those chunks into a prompt alongside the question
# MAGIC 4. Ask the LLM to answer **USING** the retrieved context
# MAGIC
# MAGIC > Since we don't have an LLM API on CE, we'll build the full pipeline and simulate
# MAGIC > the generation step. The retrieval logic is 100% real and functional.

# COMMAND ----------

# ============================================================================
# FULL RAG PIPELINE
# ============================================================================

def rag_pipeline(question: str, top_k: int = 3) -> dict:
    """
    Complete RAG pipeline:
    1. Retrieve relevant context via hybrid search
    2. Build a grounded prompt
    3. (On full Databricks: send to Model Serving endpoint)
    """
    # Step 1: RETRIEVE relevant chunks
    retrieved = hybrid_search(question, top_k=top_k)
    
    # Step 2: BUILD the prompt with retrieved context
    context_text = "\n\n---\n\n".join([
        f"[Source: {r['title']}]\n{r['chunk_text']}" for r in retrieved
    ])
    
    prompt = f"""You are a helpful customer support assistant for AcmeCorp. Answer the user's question using ONLY the context provided below. If the context doesn't contain enough information to fully answer the question, say so honestly — do not make up information.

CONTEXT:
{context_text}

USER QUESTION: {question}

ANSWER:"""
    
    # Step 3: On full Databricks, this prompt would be sent to a Model Serving endpoint:
    #   response = client.chat.completions.create(
    #       model="databricks-meta-llama-3-1-70b-instruct",
    #       messages=[{"role": "user", "content": prompt}]
    #   )
    
    return {
        'question': question,
        'retrieved_chunks': retrieved,
        'prompt': prompt,
        'prompt_length_chars': len(prompt),
        'num_chunks_used': len(retrieved),
    }

# Run the RAG pipeline
test_questions = [
    "How do I cancel my subscription and get a refund?",
    "What security certifications does AcmeCorp have?",
    "My file upload keeps failing, what should I do?",
    "How do I set up SSO for my team?",
]

for question in test_questions:
    result = rag_pipeline(question)
    print(f"\n{'=' * 70}")
    print(f"  Q: {result['question']}")
    print(f"{'=' * 70}")
    print(f"  Retrieved {result['num_chunks_used']} chunks | Prompt: {result['prompt_length_chars']} chars")
    print(f"  Sources:")
    for r in result['retrieved_chunks']:
        print(f"    📄 {r['title']} (score: {r['combined_score']:.3f})")
    print(f"\n  --- GENERATED PROMPT (first 300 chars) ---")
    print(f"  {result['prompt'][:300]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 7: Delta Sync Concept (Handbook B5)
# MAGIC
# MAGIC On full Databricks, Vector Search indexes **auto-sync** from a source Delta table:
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────────┐     auto-sync      ┌──────────────────────┐
# MAGIC │ Source Delta Table│ ──────────────────→ │ Vector Search Index  │
# MAGIC │ (knowledge_chunks)│  (TRIGGERED or     │ (auto-embedded,      │
# MAGIC │                   │   CONTINUOUS)       │  searchable)         │
# MAGIC └──────────────────┘                     └──────────────────────┘
# MAGIC         │                                          │
# MAGIC    INSERT/UPDATE                              auto re-embeds
# MAGIC    new documents                              changed rows
# MAGIC ```
# MAGIC
# MAGIC This means: when the source Delta table changes (new document added, old one updated),
# MAGIC the vector index updates automatically — **no separate ETL job** to keep embeddings fresh.

# COMMAND ----------

# ============================================================================
# SIMULATE DELTA SYNC: Add new documents and re-index
# ============================================================================

# Add a new document to the knowledge base
new_doc = {
    'doc_id': 'DOC_011',
    'source': 'help_center',
    'title': 'Mobile App Installation Guide',
    'content': """AcmeCorp is available on iOS and Android devices.

iOS Installation:
1. Open the App Store on your iPhone or iPad
2. Search for "AcmeCorp"
3. Tap 'Get' to download (requires iOS 15.0 or later)
4. Open the app and sign in with your existing account

Android Installation:
1. Open Google Play Store
2. Search for "AcmeCorp"
3. Tap 'Install' (requires Android 10 or later)
4. Open the app and sign in

Offline Mode: The mobile app supports offline access to your most recent projects. Changes made offline sync automatically when you reconnect. Note: offline mode is available on Premium and Enterprise plans only."""
}

# Chunk and embed the new document
new_chunks = chunk_text_fixed(new_doc['content'], chunk_size=400, overlap=75)
for i, chunk_text_str in enumerate(new_chunks):
    new_chunk = {
        'chunk_id': f"{new_doc['doc_id']}_chunk_{i}",
        'doc_id': new_doc['doc_id'],
        'source': new_doc['source'],
        'title': new_doc['title'],
        'chunk_index': i,
        'chunk_text': chunk_text_str,
        'char_count': len(chunk_text_str),
    }
    all_chunks.append(new_chunk)
    chunk_texts.append(chunk_text_str)

# Re-embed and rebuild index (on full Databricks, this happens automatically!)
print("Simulating Delta Sync: re-embedding after new document added...")
chunk_embeddings_updated = embedding_model.encode(chunk_texts)
faiss.normalize_L2(chunk_embeddings_updated)

index_updated = faiss.IndexFlatIP(EMBEDDING_DIM)
index_updated.add(chunk_embeddings_updated.astype(np.float32))

# Update the global index
index = index_updated

# Update BM25
tokenized_chunks = [re.findall(r'\w+', chunk.lower()) for chunk in chunk_texts]
bm25 = BM25Okapi(tokenized_chunks)

print(f"✅ Index updated: {index.ntotal} vectors (was {index.ntotal - len(new_chunks)})")

# Test: search for the new content
results = hybrid_search("How do I install the mobile app?", top_k=3)
print(f"\nSearch for 'How do I install the mobile app?':")
for r in results:
    print(f"  [{r['combined_score']:.3f}] {r['title']}: {r['chunk_text'][:80]}...")

print(f"\n📌 On full Databricks, this re-indexing happens AUTOMATICALLY via Delta Sync!")
print(f"   No manual rebuild needed — insert new rows into the source table,")
print(f"   and the Vector Search index updates itself.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # ✅ Key Takeaways
# MAGIC
# MAGIC | Concept | What We Did | Handbook Section |
# MAGIC |---------|------------|------------------|
# MAGIC | Foundation Models / LLMs | Used `all-MiniLM-L6-v2` for embeddings | A6 |
# MAGIC | Embeddings | Converted text to vectors, showed semantic similarity | A8 |
# MAGIC | RAG | Built full retrieve-stuff-generate pipeline | A7 |
# MAGIC | Unstructured data governance | Knowledge base stored as governed Delta table | B1, B2 |
# MAGIC | Chunking | Implemented fixed-size and paragraph-based strategies | B3 |
# MAGIC | Vector Search | Built FAISS index, demonstrated similarity search | B5 |
# MAGIC | Hybrid Search | Combined semantic (FAISS) + keyword (BM25) | I5 |
# MAGIC | Delta Sync | Simulated auto-sync after adding new documents | B5 |
# MAGIC
# MAGIC ### 🔗 Full Platform Mapping
# MAGIC | CE Approach | Full Databricks |
# MAGIC |---|---|
# MAGIC | `SentenceTransformer('all-MiniLM-L6-v2')` | Foundation Model API endpoint (`databricks-bge-large-en`) |
# MAGIC | FAISS `IndexFlatIP` | Mosaic AI Vector Search (managed, serverless) |
# MAGIC | Manual embed + insert | Delta Sync Index (auto-embeds from source table) |
# MAGIC | Manual rebuild | Automatic sync on source table changes |
# MAGIC | `rank_bm25` | Vector Search native hybrid search support |
# MAGIC | In-process search | REST API: `index.similarity_search(query_text=...)` |
# MAGIC
# MAGIC **Next**: Notebook 05 — AI Functions & Structured Outputs — calling LLMs from SQL.
