# Databricks notebook source

# MAGIC %md
# MAGIC # ⚡ Notebook 09 — Batch & Streaming AI Pipelines
# MAGIC
# MAGIC **Handbook Sections Covered**: C1-C5 (Serving & Foundation Models), I7 (Batch Inference at Scale), I8 (AI in Streaming Pipelines)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Learning Objectives
# MAGIC 1. **Model Serving concepts** — Foundation Model APIs, pay-per-token vs provisioned throughput
# MAGIC 2. **Batch inference** — run AI functions across millions of rows efficiently
# MAGIC 3. **Operational patterns** — partitioning, retry, idempotency, cost control
# MAGIC 4. **Streaming AI** — AI enrichment inside a streaming pipeline
# MAGIC 5. **External model proxy** — why wrap external APIs through Databricks

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Setup

# COMMAND ----------

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import datetime, timedelta
import time
import json
import mlflow
import warnings
warnings.filterwarnings('ignore')

mlflow.set_experiment("/Users/{}/AI_Handbook_09_Batch_Streaming".format(
    spark.sql("SELECT current_user()").collect()[0][0]
))

print("✅ Setup complete!")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 1: Model Serving Concepts (Handbook C1-C4)
# MAGIC
# MAGIC Model Serving is the single abstraction for deploying ANY model:
# MAGIC - Databricks-hosted Foundation Models (C1)
# MAGIC - External models (OpenAI, Anthropic, Google) — proxied through Databricks (C3)
# MAGIC - Your own fine-tuned models (C4)

# COMMAND ----------

# ============================================================================
# MODEL SERVING CONCEPTS: The unified endpoint abstraction
# ============================================================================

print("=" * 70)
print("  MODEL SERVING — One endpoint for everything (C2)")
print("=" * 70)
print("""
  ┌──────────────────────────────────────────────────────────────┐
  │  YOUR APPLICATION CODE                                       │
  │  (always calls the SAME Databricks endpoint URL)             │
  │                                                              │
  │  POST /serving-endpoints/my-endpoint/invocations             │
  │                      │                                       │
  │  ┌───────────────────▼───────────────────────────┐          │
  │  │  MODEL SERVING ENDPOINT                       │          │
  │  │  • Auto-scaling (including scale-to-zero)     │          │
  │  │  • Request/response logging (→ MLflow traces) │          │
  │  │  • Unity Catalog governance                   │          │
  │  │  • Unity AI Gateway policies                  │          │
  │  └────────────┬──────────┬──────────┬────────────┘          │
  │               │          │          │                        │
  │  ┌────────────▼──┐ ┌─────▼─────┐ ┌─▼──────────────┐       │
  │  │ DATABRICKS-   │ │ EXTERNAL  │ │ YOUR CUSTOM     │       │
  │  │ HOSTED (C1)   │ │ MODEL (C3)│ │ MODEL (C4)      │       │
  │  │               │ │           │ │                  │       │
  │  │ Llama, DBRX   │ │ OpenAI    │ │ Fine-tuned      │       │
  │  │ Pay-per-token  │ │ Anthropic │ │ MLflow-logged   │       │
  │  │ or Provisioned │ │ Gemini    │ │                 │       │
  │  │ Throughput     │ │ (proxied) │ │                 │       │
  │  └───────────────┘ └───────────┘ └─────────────────┘       │
  └──────────────────────────────────────────────────────────────┘

  KEY BENEFIT of external model proxy (C3):
  • Your code talks to ONE Databricks URL — swapping providers
    is a CONFIG change, not a code rewrite
  • Unity Catalog governance on EVERY call (even to OpenAI/Anthropic)
  • MLflow tracing captures external calls too
  • Unity AI Gateway spend caps apply uniformly

  PRICING MODES (C1):
  • PAY-PER-TOKEN: Simplest, billed per token, good for variable traffic
  • PROVISIONED THROUGHPUT: Reserved capacity (tokens/sec), good for
    predictable high-volume batch workloads (this notebook's focus)
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 2: Synthetic Data for Batch Processing

# COMMAND ----------

# ============================================================================
# SYNTHETIC DATA: 10,000 support tickets for batch processing
# ============================================================================
np.random.seed(2026)

categories = ['billing', 'technical', 'shipping', 'general']
templates = {
    'billing': ["Overcharged ${amt}", "Need refund for ${amt}", "Billing error on {date}", "Wrong invoice amount ${amt}"],
    'technical': ["App crashes on {action}", "Error {code} on dashboard", "Login issues since {date}", "Feature {feat} broken"],
    'shipping': ["Order #{oid} not delivered", "Wrong item in order #{oid}", "Damaged package #{oid}", "Return request #{oid}"],
    'general': ["Upgrade plan question", "Feature request for {feat}", "Account settings help", "Integration question for {tool}"],
}

tickets = []
for i in range(10000):
    cat = np.random.choice(categories, p=[0.3, 0.35, 0.2, 0.15])
    template = np.random.choice(templates[cat])
    text = template.format(
        amt=np.random.choice([9.99, 24.99, 49.99, 99.99]),
        date=f"2024-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}",
        action=np.random.choice(["upload", "login", "export", "search"]),
        code=np.random.randint(400, 504),
        feat=np.random.choice(["search", "export", "dashboards", "reports"]),
        oid=np.random.randint(100000, 999999),
        tool=np.random.choice(["Slack", "Jira", "Salesforce"]),
    )
    tickets.append({
        'ticket_id': f'BATCH_{i:06d}',
        'ticket_text': text,
        'true_category': cat,
        'created_at': (datetime(2024, 6, 1) + timedelta(minutes=np.random.randint(0, 43200))).isoformat(),
        'processed': False,
    })

batch_df = spark.createDataFrame(pd.DataFrame(tickets))
batch_df.write.format("delta").mode("overwrite").saveAsTable("default.batch_tickets_bronze")

print(f"✅ Batch dataset: default.batch_tickets_bronze (10,000 tickets)")
batch_df.groupBy("true_category").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 3: Batch Inference Patterns (Handbook I7)
# MAGIC
# MAGIC Running `ai_query()` across MILLIONS of rows is fundamentally different from
# MAGIC a single chatbot request. Key considerations:

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Simulated AI Function for Batch Processing

# COMMAND ----------

# ============================================================================
# SIMULATED AI FUNCTION: Mimics ai_classify behavior for batch processing
# ============================================================================

call_count = {"total": 0, "retries": 0}
call_latencies = []

def simulate_ai_classify(text: str, labels: list, 
                         fail_rate: float = 0.05, 
                         avg_latency_ms: float = 50) -> dict:
    """
    Simulates a Model Serving endpoint call with realistic behavior:
    - Network latency
    - Occasional failures (rate limiting, timeouts)
    - Token counting for cost tracking
    """
    call_count["total"] += 1
    
    # Simulate latency
    latency = max(10, np.random.normal(avg_latency_ms, avg_latency_ms * 0.3))
    time.sleep(latency / 10000)  # Scaled down for demo
    
    # Simulate occasional failures
    if np.random.random() < fail_rate:
        raise Exception("HTTP 429: Rate Limit Exceeded")
    
    call_latencies.append(latency)
    
    # Simple classification logic
    text_lower = text.lower()
    if any(w in text_lower for w in ['charge', 'refund', 'bill', 'invoice', 'overcharge']):
        category = 'billing'
    elif any(w in text_lower for w in ['crash', 'error', 'broken', 'login', 'issue']):
        category = 'technical'
    elif any(w in text_lower for w in ['deliver', 'order', 'package', 'return', 'damage']):
        category = 'shipping'
    else:
        category = 'general'
    
    tokens_used = len(text.split()) * 2
    return {"category": category, "tokens": tokens_used}

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Pattern: Retry with Exponential Backoff (I7)

# COMMAND ----------

# ============================================================================
# RETRY PATTERN: Exponential backoff for transient failures
# ============================================================================

def call_with_retry(text: str, labels: list, max_retries: int = 3) -> dict:
    """
    Robust AI function call with retry and exponential backoff.
    This is what production batch inference needs — not naked API calls.
    """
    for attempt in range(max_retries + 1):
        try:
            result = simulate_ai_classify(text, labels)
            return result
        except Exception as e:
            if attempt < max_retries:
                backoff = 2 ** attempt * 0.01  # Exponential backoff (scaled down)
                call_count["retries"] += 1
                time.sleep(backoff)
            else:
                return {"category": "error", "tokens": 0, "error": str(e)}

# Demo retry
print("=" * 70)
print("  RETRY WITH EXPONENTIAL BACKOFF (I7)")
print("=" * 70)
print(f"\n  Testing 100 calls with 5% failure rate...")

call_count = {"total": 0, "retries": 0}
successes = 0
for i in range(100):
    result = call_with_retry(f"Test ticket {i}", ['billing', 'technical', 'shipping', 'general'])
    if result.get('category') != 'error':
        successes += 1

print(f"  Results: {successes}/100 successful")
print(f"  Total API calls: {call_count['total']} (including retries: {call_count['retries']})")
print(f"  Retry rate: {call_count['retries'] / call_count['total'] * 100:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Pattern: Idempotent Processing (I7)

# COMMAND ----------

# ============================================================================
# IDEMPOTENT PROCESSING: Skip already-processed rows
# ============================================================================

# Process first batch of 200 tickets
print("=" * 70)
print("  IDEMPOTENT PROCESSING (I7)")
print("=" * 70)

# Step 1: Process first batch
batch_1 = spark.table("default.batch_tickets_bronze").limit(200).toPandas()
results_1 = []
for _, row in batch_1.iterrows():
    result = call_with_retry(row['ticket_text'], ['billing', 'technical', 'shipping', 'general'])
    results_1.append({
        'ticket_id': row['ticket_id'],
        'ai_category': result['category'],
        'tokens_used': result.get('tokens', 0),
        'processed_at': datetime.now().isoformat(),
    })

results_df_1 = spark.createDataFrame(pd.DataFrame(results_1))
results_df_1.write.format("delta").mode("overwrite").saveAsTable("default.batch_tickets_silver")
print(f"\n  Batch 1: Processed {len(results_1)} tickets")

# Step 2: Simulate a SECOND run — only process NEW tickets (idempotency)
already_processed = set(spark.table("default.batch_tickets_silver")
                       .select("ticket_id").toPandas()['ticket_id'])

batch_2 = spark.table("default.batch_tickets_bronze").limit(400).toPandas()
new_tickets = batch_2[~batch_2['ticket_id'].isin(already_processed)]

results_2 = []
for _, row in new_tickets.iterrows():
    result = call_with_retry(row['ticket_text'], ['billing', 'technical', 'shipping', 'general'])
    results_2.append({
        'ticket_id': row['ticket_id'],
        'ai_category': result['category'],
        'tokens_used': result.get('tokens', 0),
        'processed_at': datetime.now().isoformat(),
    })

if results_2:
    results_df_2 = spark.createDataFrame(pd.DataFrame(results_2))
    # MERGE: insert new rows, don't re-process existing ones
    results_df_2.createOrReplaceTempView("new_results")
    spark.sql("""
        MERGE INTO default.batch_tickets_silver AS target
        USING new_results AS source
        ON target.ticket_id = source.ticket_id
        WHEN NOT MATCHED THEN INSERT *
    """)

total = spark.table("default.batch_tickets_silver").count()
print(f"  Batch 2: Processed {len(results_2)} NEW tickets (skipped {len(batch_2) - len(new_tickets)} already done)")
print(f"  Total in silver table: {total}")
print(f"\n  💡 Key: Used MERGE to avoid re-processing (and re-paying for) rows")
print(f"     that were already successfully processed in Batch 1")
print(f"\n  📌 On full Databricks, also use Change Data Feed (CDF) to identify")
print(f"     ONLY rows that changed since last run, instead of scanning the full table")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 Pattern: Partitioning & Concurrency Control (I7)

# COMMAND ----------

# ============================================================================
# PARTITIONING & CONCURRENCY: Control how many parallel calls hit the endpoint
# ============================================================================

print("=" * 70)
print("  PARTITIONING & CONCURRENCY CONTROL (I7)")
print("=" * 70)
print("""
  The number of Spark PARTITIONS directly controls concurrent API calls:

  ┌───────────────────────────────────────────────────────────┐
  │  Spark DataFrame (10,000 rows)                            │
  │                                                           │
  │  repartition(4) → 4 concurrent endpoint calls             │
  │    ├── Partition 1: 2,500 rows → serial calls ─────────┐ │
  │    ├── Partition 2: 2,500 rows → serial calls ────────┐│ │
  │    ├── Partition 3: 2,500 rows → serial calls ───────┐││ │
  │    └── Partition 4: 2,500 rows → serial calls ──────┐│││ │
  │                                                      ││││ │
  │                                      4 concurrent ◄──┘│││ │
  │                                      calls at once ◄───┘││ │
  │                                                    ◄────┘│ │
  │                                                    ◄─────┘ │
  │                                                           │
  │  TOO FEW partitions → under-utilizes available throughput │
  │  TOO MANY partitions → overwhelms the endpoint (429s)    │
  │                                                           │
  │  RULE OF THUMB: match partitions to the endpoint's        │
  │  concurrency limit (e.g., Provisioned Throughput capacity) │
  └───────────────────────────────────────────────────────────┘
""")

# Demo: Process with controlled concurrency via repartitioning
tickets_for_partition_demo = spark.table("default.batch_tickets_bronze").limit(100)

@F.udf(StringType())
def classify_udf(text):
    if text is None:
        return None
    result = call_with_retry(text, ['billing', 'technical', 'shipping', 'general'])
    return result.get('category', 'error')

for num_partitions in [2, 4, 8]:
    call_count = {"total": 0, "retries": 0}
    start = time.time()
    
    result = (
        tickets_for_partition_demo
        .repartition(num_partitions)
        .withColumn("ai_category", classify_udf(F.col("ticket_text")))
    )
    count = result.count()  # trigger execution
    elapsed = time.time() - start
    
    print(f"  Partitions={num_partitions}: {count} rows in {elapsed:.1f}s "
          f"({call_count['total']} API calls, {call_count['retries']} retries)")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 4: Streaming AI Pipelines (Handbook I8)
# MAGIC
# MAGIC AI Functions work inside streaming DataFrames — the same `ai_classify()` UDF
# MAGIC in a `readStream` → `writeStream` pipeline.

# COMMAND ----------

# ============================================================================
# STREAMING AI PIPELINE: Real-time enrichment
# ============================================================================

# Generate streaming source data
streaming_tickets = []
for i in range(500):
    cat = np.random.choice(['billing', 'technical', 'shipping', 'general'], p=[0.3, 0.35, 0.2, 0.15])
    template = np.random.choice(templates[cat])
    text = template.format(
        amt=np.random.choice([9.99, 24.99, 49.99]),
        date="2024-07-15", action="upload", code=500,
        feat="search", oid=np.random.randint(100000, 999999), tool="Slack",
    )
    streaming_tickets.append({
        'ticket_id': f'STREAM_{i:06d}',
        'ticket_text': text,
        'event_time': (datetime(2024, 7, 1) + timedelta(seconds=i * 10)).isoformat(),
    })

streaming_source = spark.createDataFrame(pd.DataFrame(streaming_tickets))
streaming_source.write.format("delta").mode("overwrite").saveAsTable("default.streaming_tickets_source")

print("✅ Streaming source: default.streaming_tickets_source (500 rows)")

# COMMAND ----------

# ============================================================================
# STREAMING PIPELINE: readStream → AI enrichment → writeStream
# ============================================================================

print("=" * 70)
print("  STREAMING AI PIPELINE (I8)")
print("=" * 70)
print("""
  On full Databricks, this would be:
  
    stream_df = spark.readStream.table("bronze_support_tickets")
    enriched = stream_df.selectExpr(
        "*",
        "ai_classify(ticket_text, array('billing','technical','shipping')) as category"
    )
    enriched.writeStream.option("checkpointLocation", "/path").table("silver_tickets")

  The AI Function call is just another expression in a SELECT —
  same checkpointing/exactly-once guarantees as any streaming transformation.
  
  CAVEAT: LLM calls are MUCH higher latency than normal transforms.
  This can become the bottleneck stage of the pipeline.
""")

# Simulate streaming with micro-batches
@F.udf(StringType())
def streaming_classify_udf(text):
    """Simulates an AI Function in a streaming pipeline."""
    if text is None:
        return None
    text_lower = text.lower()
    if any(w in text_lower for w in ['charge', 'refund', 'bill', 'invoice']):
        return 'billing'
    elif any(w in text_lower for w in ['crash', 'error', 'broken', 'login']):
        return 'technical'
    elif any(w in text_lower for w in ['deliver', 'order', 'package', 'return']):
        return 'shipping'
    return 'general'

# Read as stream, enrich, write as stream
checkpoint_path = "/tmp/ai_handbook_streaming_checkpoint"

# Clean up any previous checkpoint
import shutil
try:
    shutil.rmtree(checkpoint_path)
except:
    pass

try:
    dbutils.fs.rm(checkpoint_path, recurse=True)
except:
    pass

streaming_query = (
    spark.readStream
    .format("delta")
    .table("default.streaming_tickets_source")
    .withColumn("ai_category", streaming_classify_udf(F.col("ticket_text")))
    .withColumn("processed_at", F.current_timestamp())
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", checkpoint_path)
    .trigger(availableNow=True)  # Process all available data, then stop
    .toTable("default.streaming_tickets_silver")
)

streaming_query.awaitTermination()

silver_count = spark.table("default.streaming_tickets_silver").count()
print(f"\n  ✅ Streaming pipeline completed: {silver_count} tickets enriched")

spark.table("default.streaming_tickets_silver").groupBy("ai_category").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Latency Considerations for Streaming AI (I8)

# COMMAND ----------

# ============================================================================
# LATENCY COMPARISON: Normal transform vs AI-enriched transform
# ============================================================================

print("=" * 70)
print("  LATENCY: Normal transform vs AI-enriched transform")
print("=" * 70)

source = spark.table("default.streaming_tickets_source").limit(1000)

# Normal transform (filter + cast) — microseconds per row
start = time.time()
normal = source.withColumn("word_count", F.size(F.split(F.col("ticket_text"), " ")))
normal_count = normal.count()
normal_time = time.time() - start

# AI-enriched transform — much slower (simulated)
start = time.time()
enriched = source.withColumn("ai_category", streaming_classify_udf(F.col("ticket_text")))
enriched_count = enriched.count()
enriched_time = time.time() - start

print(f"\n  Normal transform (word count):  {normal_time:.2f}s for {normal_count} rows")
print(f"  AI-enriched transform:          {enriched_time:.2f}s for {enriched_count} rows")
print(f"  Slowdown factor:                {enriched_time/max(normal_time, 0.001):.1f}x")
print(f"\n  ⚠️ In production with real LLM calls, the slowdown is 100-1000x!")
print(f"     → Micro-batch sizing must account for LLM latency")
print(f"     → Trigger interval should be large enough for LLM calls to complete")
print(f"     → Consider: is real-time enrichment needed, or is batch sufficient?")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 5: Cost-Optimized Pipeline Patterns (I7, I15)

# COMMAND ----------

# ============================================================================
# COST OPTIMIZATION: Skip unchanged rows with Change Data Feed concept
# ============================================================================

print("=" * 70)
print("  COST OPTIMIZATION: Process only CHANGED rows (I7, I15)")
print("=" * 70)

# Simulate a table with 10,000 rows where only 500 changed
full_table_count = 10000
changed_count = 500
cost_per_row = 0.0001  # typical LLM call cost

full_reprocess_cost = full_table_count * cost_per_row
incremental_cost = changed_count * cost_per_row
savings = (1 - incremental_cost / full_reprocess_cost) * 100

print(f"""
  Scenario: Table with {full_table_count:,} rows, {changed_count} changed since last run

  ❌ NAIVE: Reprocess entire table every run
     Cost: {full_table_count:,} × ${cost_per_row} = ${full_reprocess_cost:.2f}

  ✅ SMART: Use Change Data Feed to identify changed rows only
     Cost: {changed_count} × ${cost_per_row} = ${incremental_cost:.2f}
     Savings: {savings:.0f}%!

  On full Databricks, use Change Data Feed (CDF):
    spark.readStream
        .option("readChangeFeed", "true")      # ← Only read changed rows!
        .option("startingVersion", last_version)
        .table("bronze_tickets")

  This is the SAME Delta feature data engineers already use for
  incremental ETL — applied here to avoid re-paying for AI on unchanged rows.
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # ✅ Key Takeaways
# MAGIC
# MAGIC | Concept | What We Did | Handbook Section |
# MAGIC |---------|------------|------------------|
# MAGIC | Model Serving | Explained unified endpoint abstraction | C1-C4 |
# MAGIC | External model proxy | Why wrap OpenAI/Anthropic through Databricks | C3 |
# MAGIC | Batch inference | Processed 10K tickets with retry/backoff | I7 |
# MAGIC | Partitioning | Controlled concurrency via repartition | I7 |
# MAGIC | Retry + backoff | Handled transient failures robustly | I7 |
# MAGIC | Idempotent processing | MERGE to skip already-processed rows | I7 |
# MAGIC | Streaming AI | readStream → AI UDF → writeStream pipeline | I8 |
# MAGIC | Latency awareness | Showed LLM latency vs normal transforms | I8 |
# MAGIC | Cost optimization | CDF for incremental processing | I7, I15 |
# MAGIC | Batch vs real-time | When to use each mode | C5 |
# MAGIC
# MAGIC **Next**: Notebook 10 — LLM Internals & Responsible AI.
