# Databricks notebook source

# MAGIC %md
# MAGIC # 📊 Notebook 07 — MLflow 3 for GenAI: Tracing, Judges & Evaluation
# MAGIC
# MAGIC **Handbook Sections Covered**: E1 (Why GenAI evaluation is different), E2 (MLflow 3 for GenAI), I17 (RAG evaluation — retrieval vs generation metrics)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Learning Objectives
# MAGIC 1. **Why GenAI evaluation is different** — free-form text ≠ simple accuracy
# MAGIC 2. **MLflow Tracing** — capture every step of an LLM/agent call
# MAGIC 3. **LLM Judges / Scorers** — automated quality metrics (groundedness, relevance, safety)
# MAGIC 4. **RAG evaluation** — retrieval metrics (precision@K, recall@K) vs generation metrics
# MAGIC 5. **Prompt version comparison** — A/B test prompts using evaluation sets
# MAGIC 6. **Review App concept** — human feedback → evaluation datasets

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Setup

# COMMAND ----------

%pip install -q sentence-transformers faiss-cpu rank-bm25

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
import json
import time
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

mlflow.set_experiment("/Users/{}/AI_Handbook_07_GenAI_Evaluation".format(
    spark.sql("SELECT current_user()").collect()[0][0]
))

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Setup complete!")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 1: Why GenAI Evaluation is Different (Handbook E1)
# MAGIC
# MAGIC | Classic ML | GenAI |
# MAGIC |---|---|
# MAGIC | One correct answer (label) | No single "correct" text answer |
# MAGIC | Metric: accuracy, RMSE, F1 | Quality: "was this helpful?", "is it grounded?", "is it safe?" |
# MAGIC | Computed automatically | Needs **LLM judges** to score at scale |
# MAGIC | Same metric in dev & prod | Same judges must run in **both** dev and production |

# COMMAND ----------

# ============================================================================
# THE EVALUATION PROBLEM: Why you can't just use accuracy for GenAI
# ============================================================================

print("=" * 70)
print("  WHY GENAI EVALUATION IS DIFFERENT (E1)")
print("=" * 70)
print("""
  Classic ML example:
    Question: "Is this email spam?"
    Correct answer: "Yes"
    Model output: "Yes"
    → accuracy = 1.0 ✅ (trivial to compute)

  GenAI example:
    Question: "How do I cancel my subscription?"
    Reference: "Go to Settings > Billing > Cancel Subscription"
    Model output: "Navigate to your account settings, then billing, 
                   and click cancel subscription."
    → Is this correct? It says the same thing in different words...
    → accuracy = 0.0 ❌ (character-for-character mismatch, but SEMANTICALLY correct!)

  This is why GenAI evaluation needs:
    1. SEMANTIC comparison (not character-level matching)
    2. LLM JUDGES that understand meaning
    3. Multiple DIMENSIONS: correctness, groundedness, relevance, safety
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 2: Build a RAG System to Evaluate

# COMMAND ----------

# ============================================================================
# QUICK RAG SETUP: Reuse knowledge base from Notebook 04
# ============================================================================

knowledge_base = [
    {"id": "kb_1", "title": "Cancel Subscription", "content": "To cancel your subscription, go to Settings > Billing > Cancel Subscription. Your subscription remains active until the end of your billing period. Data retained for 90 days. Refunds available within 14 days of billing cycle."},
    {"id": "kb_2", "title": "Password Reset", "content": "To reset your password: go to login page, click 'Forgot Password', enter your email, click the reset link in email (valid 1 hour). Password must be 12+ characters with uppercase, lowercase, number, and special character."},
    {"id": "kb_3", "title": "API Authentication", "content": "API uses Bearer token authentication. Generate key at Settings > API > Generate Key. Rate limits: Free=100/hr, Basic=1000/hr, Premium=10000/hr. Current version is v2. V1 deprecated December 2024."},
    {"id": "kb_4", "title": "Data Export", "content": "Export data at Settings > Data > Export. Choose JSON or CSV format. Premium/Enterprise have automatic daily backups. Retained 30 days (Premium) or 90 days (Enterprise). API export via GET /v2/export/full."},
    {"id": "kb_5", "title": "Security", "content": "AES-256 encryption at rest, TLS 1.3 in transit. SOC 2 Type II, GDPR, HIPAA compliant. 99.99% uptime SLA on Enterprise. Bug bounty program available. BYOK on Enterprise plan."},
    {"id": "kb_6", "title": "Billing", "content": "Accepts Visa, Mastercard, Amex, PayPal. Annual plans get 20% discount. Failed payments retry 3 times over 7 days. After 3 failures, downgrade to Free plan. Prices in USD, exclusive of tax."},
    {"id": "kb_7", "title": "Team Management", "content": "Manage team at Settings > Team. Roles: Viewer (read-only), Editor (read-write), Admin (full control). Groups for easier permission management. Enterprise supports SAML SSO with Okta, Azure AD, Google Workspace."},
    {"id": "kb_8", "title": "Troubleshooting", "content": "App not loading: clear cache, try different browser. File upload failing: check size limit (50MB Free/Basic, 500MB Premium). Slow performance: need 5 Mbps minimum. 2FA locked out: use backup codes or contact support with photo ID."},
]

# Build vector index
kb_texts = [doc['content'] for doc in knowledge_base]
kb_embeddings = embedding_model.encode(kb_texts)
faiss.normalize_L2(kb_embeddings)

kb_index = faiss.IndexFlatIP(384)
kb_index.add(kb_embeddings.astype(np.float32))

def retrieve(query: str, top_k: int = 3) -> list:
    query_emb = embedding_model.encode([query])
    faiss.normalize_L2(query_emb)
    scores, indices = kb_index.search(query_emb.astype(np.float32), top_k)
    return [{"doc": knowledge_base[idx], "score": float(scores[0][i])} 
            for i, idx in enumerate(indices[0])]

def rag_answer(query: str, top_k: int = 3) -> dict:
    """Simulate a RAG pipeline with traceable steps."""
    start_time = time.time()
    
    # Step 1: Retrieve
    retrieved = retrieve(query, top_k)
    retrieval_time = time.time() - start_time
    
    # Step 2: Build context
    context = "\n\n".join([r['doc']['content'] for r in retrieved])
    
    # Step 3: Generate answer (simulated — on full Databricks this calls a Foundation Model)
    # We'll use a simple extractive approach for demonstration
    answer_parts = []
    query_lower = query.lower()
    for r in retrieved:
        content = r['doc']['content']
        sentences = content.split('. ')
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in query_lower.split()[:3]):
                answer_parts.append(sentence.strip())
    
    answer = ". ".join(answer_parts[:3]) + "." if answer_parts else "I couldn't find relevant information."
    generation_time = time.time() - start_time - retrieval_time
    
    return {
        'query': query,
        'answer': answer,
        'retrieved_docs': [r['doc']['id'] for r in retrieved],
        'retrieved_titles': [r['doc']['title'] for r in retrieved],
        'retrieval_scores': [r['score'] for r in retrieved],
        'context': context,
        'retrieval_time_ms': round(retrieval_time * 1000, 1),
        'generation_time_ms': round(generation_time * 1000, 1),
        'total_time_ms': round((time.time() - start_time) * 1000, 1),
    }

print("✅ RAG system ready for evaluation")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 3: MLflow Tracing (Handbook E2)
# MAGIC
# MAGIC > Tracing automatically logs EVERY input, intermediate step (including tool calls),
# MAGIC > and output of an LLM/agent call. This is the debugging foundation.
# MAGIC >
# MAGIC > On full Databricks: `mlflow.openai.autolog()` or `@mlflow.trace`

# COMMAND ----------

# ============================================================================
# MLFLOW TRACING: Capture every step of the RAG pipeline
# ============================================================================

def traced_rag_pipeline(query: str) -> dict:
    """
    RAG pipeline with manual tracing.
    On full Databricks, @mlflow.trace would do this automatically.
    """
    with mlflow.start_run(run_name=f"rag_trace_{hash(query) % 10000}") as run:
        # Log the input
        mlflow.log_param("user_query", query[:200])
        mlflow.set_tag("pipeline_type", "rag")
        
        trace_start = time.time()
        
        # Step 1: Retrieval (trace this step)
        retrieval_start = time.time()
        retrieved = retrieve(query, top_k=3)
        retrieval_ms = (time.time() - retrieval_start) * 1000
        
        mlflow.log_metric("retrieval_time_ms", retrieval_ms)
        mlflow.log_metric("num_docs_retrieved", len(retrieved))
        mlflow.log_metric("top_retrieval_score", retrieved[0]['score'] if retrieved else 0)
        
        # Step 2: Context assembly
        context = "\n\n".join([r['doc']['content'] for r in retrieved])
        mlflow.log_metric("context_length_chars", len(context))
        
        # Step 3: Generation
        gen_start = time.time()
        result = rag_answer(query)
        gen_ms = (time.time() - gen_start) * 1000
        
        mlflow.log_metric("generation_time_ms", gen_ms)
        mlflow.log_metric("answer_length_chars", len(result['answer']))
        mlflow.log_metric("total_time_ms", (time.time() - trace_start) * 1000)
        
        # Log the full trace as an artifact
        trace_data = {
            "query": query,
            "steps": [
                {"step": "retrieval", "docs": result['retrieved_titles'], 
                 "scores": result['retrieval_scores'], "time_ms": retrieval_ms},
                {"step": "context_assembly", "context_length": len(context)},
                {"step": "generation", "answer": result['answer'], "time_ms": gen_ms},
            ],
            "final_answer": result['answer'],
        }
        with open("/tmp/trace.json", "w") as f:
            json.dump(trace_data, f, indent=2)
        mlflow.log_artifact("/tmp/trace.json")
        
        result['run_id'] = run.info.run_id
        return result

# Run traced queries
print("=" * 70)
print("  MLFLOW TRACING — Capturing every RAG pipeline step")
print("=" * 70)

eval_queries = [
    "How do I cancel my subscription?",
    "What are the API rate limits?",
    "How do I reset my password?",
    "What security certifications do you have?",
    "How do I export my data?",
]

traced_results = []
for query in eval_queries:
    result = traced_rag_pipeline(query)
    traced_results.append(result)
    print(f"\n  Q: {query}")
    print(f"  A: {result['answer'][:80]}...")
    print(f"  ⏱️ Retrieval: {result['retrieval_time_ms']}ms | Total: {result['total_time_ms']}ms")
    print(f"  📄 Sources: {result['retrieved_titles']}")

print(f"\n📌 On full Databricks:")
print(f"   @mlflow.trace")
print(f"   def my_rag_app(query):")
print(f"       ...  # every step auto-traced, viewable in MLflow UI")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 4: LLM Judges / Scorers (Handbook E2)
# MAGIC
# MAGIC Automated quality metrics computed by prompting another LLM to grade outputs.
# MAGIC Here we build the judges from scratch — on full Databricks, these are built-in scorers.

# COMMAND ----------

# ============================================================================
# LLM JUDGES: Automated quality scoring
# ============================================================================

def judge_groundedness(answer: str, context: str) -> dict:
    """
    GROUNDEDNESS: Is the answer supported by the retrieved context?
    Score 0-1. A hallucinated claim not in the context gets a low score.
    
    On full Databricks: built-in MLflow 3 scorer for groundedness.
    """
    if not answer or not context:
        return {"score": 0.0, "reason": "Empty answer or context"}
    
    # Split answer into claims
    answer_sentences = [s.strip() for s in answer.split('. ') if s.strip()]
    if not answer_sentences:
        return {"score": 0.0, "reason": "No claims to evaluate"}
    
    # Check each claim against context using semantic similarity
    context_emb = embedding_model.encode([context])
    grounded_count = 0
    evaluations = []
    
    for sentence in answer_sentences:
        sent_emb = embedding_model.encode([sentence])
        similarity = cosine_similarity(sent_emb, context_emb)[0][0]
        is_grounded = similarity > 0.3  # threshold
        grounded_count += int(is_grounded)
        evaluations.append({
            "claim": sentence[:50],
            "similarity_to_context": round(float(similarity), 3),
            "grounded": is_grounded
        })
    
    score = grounded_count / len(answer_sentences) if answer_sentences else 0
    return {
        "score": round(score, 3),
        "grounded_claims": grounded_count,
        "total_claims": len(answer_sentences),
        "evaluations": evaluations,
        "reason": f"{grounded_count}/{len(answer_sentences)} claims supported by context"
    }

def judge_relevance(answer: str, query: str) -> dict:
    """
    RELEVANCE: Does the answer actually address the user's question?
    
    On full Databricks: built-in MLflow 3 scorer for relevance.
    """
    if not answer or not query:
        return {"score": 0.0, "reason": "Empty answer or query"}
    
    answer_emb = embedding_model.encode([answer])
    query_emb = embedding_model.encode([query])
    similarity = float(cosine_similarity(answer_emb, query_emb)[0][0])
    
    # Scale and clip to 0-1
    score = max(0, min(1, (similarity - 0.1) / 0.6))
    
    return {
        "score": round(score, 3),
        "semantic_similarity": round(similarity, 3),
        "reason": f"Answer-query similarity: {similarity:.3f}"
    }

def judge_safety(answer: str) -> dict:
    """
    SAFETY: Does the answer contain potentially harmful content?
    
    On full Databricks: Unity AI Gateway guardrails (PII detection,
    prompt injection detection, content filtering).
    """
    unsafe_patterns = [
        (r'\b\d{3}-\d{2}-\d{4}\b', 'SSN pattern detected'),
        (r'\b\d{16}\b', 'Credit card number pattern'),
        (r'password\s*[:=]\s*\S+', 'Password exposure'),
        (r'\bignore\s+previous\s+instructions?\b', 'Prompt injection attempt'),
        (r'\bdelete\s+all\b', 'Destructive action suggestion'),
    ]
    
    issues = []
    for pattern, description in unsafe_patterns:
        if re.search(pattern, answer, re.IGNORECASE):
            issues.append(description)
    
    score = 1.0 if not issues else max(0, 1 - len(issues) * 0.3)
    return {
        "score": round(score, 3),
        "issues": issues,
        "reason": f"{'No safety issues' if not issues else f'{len(issues)} issue(s) found'}"
    }

# Run judges on our traced results
print("=" * 70)
print("  LLM JUDGES — Automated quality scoring")
print("=" * 70)

evaluation_results = []
for result in traced_results:
    ground = judge_groundedness(result['answer'], result['context'])
    relev = judge_relevance(result['answer'], result['query'])
    safe = judge_safety(result['answer'])
    
    eval_row = {
        'query': result['query'][:40],
        'groundedness': ground['score'],
        'relevance': relev['score'],
        'safety': safe['score'],
        'answer_preview': result['answer'][:50],
    }
    evaluation_results.append(eval_row)

eval_df = pd.DataFrame(evaluation_results)
print(eval_df.to_string(index=False))

print(f"\n  📊 Average Scores:")
print(f"     Groundedness: {eval_df['groundedness'].mean():.3f}")
print(f"     Relevance:    {eval_df['relevance'].mean():.3f}")
print(f"     Safety:       {eval_df['safety'].mean():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 5: RAG Evaluation — Retrieval vs Generation (Handbook I17)
# MAGIC
# MAGIC > For RAG specifically, evaluation splits into TWO concerns:
# MAGIC > 1. **RETRIEVAL quality** — did we find the right documents? (precision@K, recall@K)
# MAGIC > 2. **GENERATION quality** — given good docs, did the model answer well? (groundedness, relevance)
# MAGIC >
# MAGIC > If the answer is bad, you need to know WHICH half failed — different fixes needed!

# COMMAND ----------

# ============================================================================
# RAG EVALUATION: Retrieval metrics + Generation metrics
# ============================================================================

# Ground truth: which document(s) SHOULD be retrieved for each query
ground_truth = {
    "How do I cancel my subscription?": {"relevant_docs": ["kb_1"], "expected_answer_keywords": ["settings", "billing", "cancel"]},
    "What are the API rate limits?": {"relevant_docs": ["kb_3"], "expected_answer_keywords": ["100", "1000", "10000", "rate"]},
    "How do I reset my password?": {"relevant_docs": ["kb_2"], "expected_answer_keywords": ["forgot", "password", "email", "reset"]},
    "What security certifications do you have?": {"relevant_docs": ["kb_5"], "expected_answer_keywords": ["soc", "gdpr", "hipaa", "aes"]},
    "How do I export my data?": {"relevant_docs": ["kb_4"], "expected_answer_keywords": ["settings", "data", "export", "json", "csv"]},
}

print("=" * 70)
print("  RAG EVALUATION: Retrieval vs Generation metrics (I17)")
print("=" * 70)

rag_eval_results = []
for result in traced_results:
    query = result['query']
    gt = ground_truth.get(query, {})
    relevant_docs = set(gt.get('relevant_docs', []))
    expected_keywords = gt.get('expected_answer_keywords', [])
    
    retrieved_docs = set(result['retrieved_docs'])
    
    # --- RETRIEVAL METRICS ---
    if relevant_docs:
        # Precision@K: of the top K retrieved, what fraction are relevant?
        precision_at_k = len(relevant_docs & retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0
        # Recall@K: of all relevant docs, what fraction did we find?
        recall_at_k = len(relevant_docs & retrieved_docs) / len(relevant_docs) if relevant_docs else 0
    else:
        precision_at_k = recall_at_k = 0
    
    # --- GENERATION METRICS ---
    groundedness = judge_groundedness(result['answer'], result['context'])
    relevance = judge_relevance(result['answer'], query)
    
    # Keyword coverage (simple factual accuracy check)
    answer_lower = result['answer'].lower()
    keywords_found = sum(1 for kw in expected_keywords if kw in answer_lower)
    keyword_coverage = keywords_found / len(expected_keywords) if expected_keywords else 0
    
    rag_eval_results.append({
        'query': query[:35] + '...',
        'precision@K': round(precision_at_k, 2),
        'recall@K': round(recall_at_k, 2),
        'groundedness': groundedness['score'],
        'relevance': relevance['score'],
        'keyword_coverage': round(keyword_coverage, 2),
        'diagnosis': 'retrieval_fail' if recall_at_k < 0.5 else ('generation_fail' if groundedness['score'] < 0.3 else 'good'),
    })

rag_eval_df = pd.DataFrame(rag_eval_results)
print(rag_eval_df.to_string(index=False))

print(f"\n  📊 Average Retrieval Metrics:")
print(f"     Precision@3: {rag_eval_df['precision@K'].mean():.3f}")
print(f"     Recall@3:    {rag_eval_df['recall@K'].mean():.3f}")
print(f"\n  📊 Average Generation Metrics:")
print(f"     Groundedness: {rag_eval_df['groundedness'].mean():.3f}")
print(f"     Relevance:    {rag_eval_df['relevance'].mean():.3f}")

print(f"\n  💡 DIAGNOSTIC:")
print(f"     If retrieval metrics are LOW → fix chunking, embedding model, or add hybrid search")
print(f"     If generation metrics are LOW → fix prompt, temperature, or model choice")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 6: Prompt Version Comparison (Handbook E2 — Version Tracking)

# COMMAND ----------

# ============================================================================
# A/B PROMPT COMPARISON: Compare two prompt versions on the same eval set
# ============================================================================

prompt_v1 = "Based on the following context, answer the question.\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"

prompt_v2 = """You are a helpful customer support assistant. Answer the user's question using ONLY the context below. Be specific and include relevant details. If the context doesn't contain the answer, say "I don't have that information."

Context:
{context}

Question: {query}

Answer:"""

def simulate_answer(query, context, prompt_template):
    """Simulate different answer quality based on prompt version."""
    answer_parts = []
    query_lower = query.lower()
    for sentence in context.split('. '):
        if any(keyword in sentence.lower() for keyword in query_lower.split()[:3]):
            answer_parts.append(sentence.strip())
    
    if "ONLY" in prompt_template:
        # v2 prompt produces more focused answers
        return ". ".join(answer_parts[:2]) + "." if answer_parts else "I don't have that information."
    else:
        # v1 prompt might include less relevant parts
        return ". ".join(answer_parts[:4]) + "." if answer_parts else "No answer available."

print("=" * 70)
print("  PROMPT VERSION COMPARISON")
print("=" * 70)

comparison_results = []
for query in eval_queries[:3]:
    retrieved = retrieve(query, top_k=3)
    context = "\n\n".join([r['doc']['content'] for r in retrieved])
    
    # Generate with each prompt version
    answer_v1 = simulate_answer(query, context, prompt_v1)
    answer_v2 = simulate_answer(query, context, prompt_v2)
    
    # Score both
    ground_v1 = judge_groundedness(answer_v1, context)
    ground_v2 = judge_groundedness(answer_v2, context)
    relev_v1 = judge_relevance(answer_v1, query)
    relev_v2 = judge_relevance(answer_v2, query)
    
    comparison_results.append({
        'query': query[:30] + '...',
        'v1_groundedness': ground_v1['score'],
        'v2_groundedness': ground_v2['score'],
        'v1_relevance': relev_v1['score'],
        'v2_relevance': relev_v2['score'],
        'winner': 'v2' if (ground_v2['score'] + relev_v2['score']) > (ground_v1['score'] + relev_v1['score']) else 'v1',
    })

comp_df = pd.DataFrame(comparison_results)
print(comp_df.to_string(index=False))

print(f"\n  📌 On full Databricks:")
print(f"     mlflow.genai.evaluate(model_v1, eval_dataset, scorers=[groundedness, relevance])")
print(f"     mlflow.genai.evaluate(model_v2, eval_dataset, scorers=[groundedness, relevance])")
print(f"     → Side-by-side comparison in MLflow UI")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 7: Review App & Human Feedback (Concept — Handbook E2)
# MAGIC
# MAGIC > The **Review App** is a UI where domain experts review production traces and
# MAGIC > give feedback (👍/👎, corrections). This feedback:
# MAGIC > 1. Becomes a high-quality **evaluation dataset** for testing future versions
# MAGIC > 2. **Calibrates** the LLM judges so automated scores match what humans consider "good"

# COMMAND ----------

# ============================================================================
# REVIEW APP CONCEPT: Human feedback → evaluation dataset
# ============================================================================

print("=" * 70)
print("  REVIEW APP & HUMAN FEEDBACK LOOP (E2)")
print("=" * 70)
print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │  PRODUCTION TRAFFIC                                             │
  │  ┌─────────┐    ┌──────────┐    ┌──────────┐                   │
  │  │ User    │───▶│ Agent /  │───▶│ MLflow   │                   │
  │  │ Query   │    │ RAG App  │    │ Trace    │                   │
  │  └─────────┘    └──────────┘    └────┬─────┘                   │
  │                                      │                          │
  │                          ┌───────────▼───────────┐              │
  │                          │   REVIEW APP (E2)     │              │
  │                          │                       │              │
  │                          │  Domain Expert sees:  │              │
  │                          │  - User's question    │              │
  │                          │  - Agent's answer     │              │
  │                          │  - Retrieved sources  │              │
  │                          │                       │              │
  │                          │  Expert gives:        │              │
  │                          │  - 👍 / 👎           │              │
  │                          │  - Corrected answer   │              │
  │                          │  - "This was good     │              │
  │                          │    because..."        │              │
  │                          └───────────┬───────────┘              │
  │                                      │                          │
  │                          ┌───────────▼───────────┐              │
  │                          │  EVALUATION DATASET   │              │
  │                          │  (curated, human-     │              │
  │                          │   validated examples) │              │
  │                          └───────────┬───────────┘              │
  │                                      │                          │
  │                    ┌─────────────────▼─────────────────┐        │
  │                    │  Used for:                         │        │
  │                    │  1. Testing new prompt versions    │        │
  │                    │  2. Calibrating LLM judges         │        │
  │                    │  3. Fine-tuning custom models      │        │
  │                    └───────────────────────────────────┘        │
  └─────────────────────────────────────────────────────────────────┘

  Key insight: The SAME scorers run in DEVELOPMENT (comparing versions)
  AND in PRODUCTION MONITORING (continuously grading live traffic).
  A regression is caught with the SAME yardstick used during development.
""")

# Simulate an evaluation dataset built from human feedback
human_eval_dataset = [
    {"query": "How do I cancel?", "expected_answer": "Go to Settings > Billing > Cancel Subscription.", "human_score": 5, "notes": "Must mention Settings > Billing"},
    {"query": "What's the API rate limit for Basic?", "expected_answer": "1,000 requests per hour for Basic plan.", "human_score": 5, "notes": "Must state exact number"},
    {"query": "How to reset password?", "expected_answer": "Click 'Forgot Password' on login page, enter email, click reset link.", "human_score": 4, "notes": "Steps should be clear"},
]

eval_dataset_df = pd.DataFrame(human_eval_dataset)
print("\n  📋 Simulated Evaluation Dataset (from human feedback):")
print(eval_dataset_df.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # ✅ Key Takeaways
# MAGIC
# MAGIC | Concept | What We Did | Handbook Section |
# MAGIC |---------|------------|------------------|
# MAGIC | GenAI eval is different | Showed why accuracy ≠ quality for text | E1 |
# MAGIC | MLflow Tracing | Captured every RAG pipeline step with metrics | E2 |
# MAGIC | LLM Judges | Built groundedness, relevance, safety scorers | E2 |
# MAGIC | Retrieval vs Generation | Separated precision@K/recall@K from groundedness | I17 |
# MAGIC | Prompt comparison | A/B tested two prompt versions with judges | E2 |
# MAGIC | Review App | Explained human feedback → eval dataset loop | E2 |
# MAGIC
# MAGIC ### 🔗 Full Platform Mapping
# MAGIC | CE Approach | Full Databricks |
# MAGIC |---|---|
# MAGIC | Manual `mlflow.log_metric/artifact` | `@mlflow.trace` auto-instrumentation |
# MAGIC | Custom judge functions | `mlflow.genai.evaluate()` built-in scorers |
# MAGIC | Manual precision/recall | Built-in retrieval metrics in evaluation harness |
# MAGIC | Simulated review dataset | MLflow 3 Review App (UI for domain experts) |
# MAGIC | Side-by-side print | MLflow UI side-by-side version comparison |
# MAGIC
# MAGIC **Next**: Notebook 08 — Governance & Cost Control with Unity AI Gateway concepts.
