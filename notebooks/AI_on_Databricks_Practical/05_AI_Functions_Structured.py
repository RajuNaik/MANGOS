# Databricks notebook source

# MAGIC %md
# MAGIC # 🤖 Notebook 05 — AI Functions & Structured Outputs
# MAGIC
# MAGIC **Handbook Sections Covered**: B4 (AI Functions in SQL), I6 (Structured Outputs), I9 (Multimodal AI)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Learning Objectives
# MAGIC 1. **AI Functions** — replicate `ai_classify`, `ai_extract`, `ai_summarize`, `ai_similarity` as PySpark UDFs
# MAGIC 2. **Structured Outputs** — force LLM output into a JSON schema for reliable pipeline integration
# MAGIC 3. **Batch enrichment** — apply AI functions across a full Delta table (medallion architecture)
# MAGIC 4. **Multimodal concepts** — understand `ai_parse_document()` and the FILE type

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Setup

# COMMAND ----------

%pip install -q sentence-transformers transformers torch

# COMMAND ----------

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import *
import mlflow
import json
import re
import warnings
warnings.filterwarnings('ignore')

mlflow.set_experiment("/Users/{}/AI_Handbook_05_AIFunctions".format(
    spark.sql("SELECT current_user()").collect()[0][0]
))

print("✅ Base setup complete! Loading models...")

# COMMAND ----------

# Load models for AI function simulation
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Embedding model (for ai_similarity)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Zero-shot classification model (for ai_classify)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)

print("✅ Models loaded:")
print("   - Embedding model: all-MiniLM-L6-v2 (for ai_similarity)")
print("   - Classifier: facebook/bart-large-mnli (for ai_classify)")
print(f"\n📌 On full Databricks, these are replaced by SQL AI Functions:")
print(f"   SELECT ai_classify(text, ARRAY('billing','technical','shipping'))")
print(f"   SELECT ai_similarity(text1, text2)")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 1: Generate Synthetic Support Tickets

# COMMAND ----------

# ============================================================================
# SYNTHETIC DATA: 1000 support tickets for AI enrichment
# ============================================================================
np.random.seed(2026)

ticket_templates = {
    'billing': [
        "I was charged ${amount} but my plan should only cost ${plan_cost}. Please fix this overcharge.",
        "I need a refund for my last payment of ${amount}. I cancelled my subscription on {date}.",
        "My credit card was charged twice for the same month. Transaction IDs: {txn1} and {txn2}.",
        "I'm being billed annually but I signed up for monthly billing. Please switch me to monthly.",
        "The promotional discount of {discount}% wasn't applied to my last invoice.",
        "I haven't used the platform in months but I'm still being charged ${amount}/month.",
        "Can you send me an invoice for my last 3 months of charges for expense reporting?",
        "I upgraded from Basic to Premium mid-month. How is the prorated charge calculated?",
    ],
    'technical': [
        "The application crashes every time I try to upload a file larger than {size}MB.",
        "I get a '500 Internal Server Error' when accessing the dashboard since {date}.",
        "My account shows 0 projects but I had {count} projects last week. Where did my data go?",
        "The API returns 'Rate Limit Exceeded' even though I'm well under my {limit} requests/hour limit.",
        "SSO login with Azure AD stopped working after our IT team updated the SAML configuration.",
        "The search function returns no results even when I search for exact file names I know exist.",
        "Two-factor authentication codes from my authenticator app are being rejected as invalid.",
        "The real-time collaboration feature shows a 'connection lost' error every few minutes.",
    ],
    'shipping': [
        "My order #{order_id} was supposed to arrive on {date} but it still hasn't been delivered.",
        "I received the wrong item. I ordered {item1} but received {item2} instead.",
        "The tracking number {tracking} shows 'delivered' but I never received the package.",
        "I need to return order #{order_id}. How do I get a prepaid return shipping label?",
        "Can I change the delivery address for order #{order_id}? It hasn't shipped yet.",
        "My package arrived damaged. The box was crushed and the product inside is broken.",
        "I placed an express shipping order but it's been 5 days and it still hasn't shipped.",
        "The delivery person left my package outside in the rain and everything inside is soaked.",
    ],
    'general': [
        "I love your product! Can you tell me about any upcoming features planned for Q{quarter}?",
        "How do I add more team members to my workspace? I need to invite {count} new people.",
        "What are the differences between the Premium and Enterprise plans?",
        "Can you help me understand the best practices for organizing my projects?",
        "I'd like to schedule a demo of your enterprise features for my team of {count} people.",
        "Do you offer educational or nonprofit discounts? I'm with {org_type}.",
        "Is there a way to integrate AcmeCorp with our existing {tool} setup?",
        "What data residency options do you have for customers in {region}?",
    ],
}

def generate_ticket(category):
    template = np.random.choice(ticket_templates[category])
    return template.format(
        amount=np.random.choice([9.99, 24.99, 49.99, 99.99, 199.99]),
        plan_cost=np.random.choice([9.99, 24.99]),
        date=f"2024-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}",
        discount=np.random.choice([10, 15, 20, 25, 50]),
        txn1=f"TXN-{np.random.randint(100000,999999)}",
        txn2=f"TXN-{np.random.randint(100000,999999)}",
        size=np.random.choice([10, 25, 50, 100]),
        count=np.random.randint(2, 50),
        limit=np.random.choice([100, 1000, 10000]),
        order_id=f"{np.random.randint(100000,999999)}",
        item1=np.random.choice(["Blue Widget", "Pro Gadget", "Ultra Sensor"]),
        item2=np.random.choice(["Red Widget", "Basic Gadget", "Mini Sensor"]),
        tracking=f"1Z{np.random.randint(1000000000, 9999999999)}",
        quarter=np.random.randint(1, 5),
        org_type=np.random.choice(["a university", "a nonprofit", "a government agency"]),
        tool=np.random.choice(["Slack", "Jira", "Salesforce", "HubSpot"]),
        region=np.random.choice(["the EU", "Canada", "Australia", "Japan"]),
    )

# Generate 1000 tickets
tickets = []
categories = ['billing', 'technical', 'shipping', 'general']
probs = [0.3, 0.35, 0.2, 0.15]

for i in range(1000):
    cat = np.random.choice(categories, p=probs)
    tickets.append({
        'ticket_id': f'TICKET_{i:05d}',
        'customer_id': f'CUST_{np.random.randint(0, 500):05d}',
        'ticket_text': generate_ticket(cat),
        'true_category': cat,
        'priority': np.random.choice(['low', 'medium', 'high', 'urgent'], p=[0.3, 0.4, 0.2, 0.1]),
        'created_at': f"2024-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}",
    })

tickets_df = pd.DataFrame(tickets)
spark_tickets = spark.createDataFrame(tickets_df)
spark_tickets.write.format("delta").mode("overwrite").saveAsTable("default.support_tickets_bronze")

print(f"✅ Bronze table: default.support_tickets_bronze ({len(tickets)} tickets)")
print(f"   Category distribution: {dict(tickets_df['true_category'].value_counts())}")
spark_tickets.show(5, truncate=60)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 2: AI Functions as UDFs (Handbook B4)
# MAGIC
# MAGIC On full Databricks, you'd use SQL AI Functions directly:
# MAGIC ```sql
# MAGIC SELECT ai_classify(ticket_text, ARRAY('billing','technical','shipping','general')) FROM tickets
# MAGIC ```
# MAGIC
# MAGIC On CE, we build the **same pattern** as PySpark UDFs using local models.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 ai_classify — Zero-shot Classification

# COMMAND ----------

# ============================================================================
# ai_classify UDF: Classify text into one of the given labels
# ============================================================================

def ai_classify_fn(text: str, labels: list) -> str:
    """
    Open-source equivalent of Databricks ai_classify().
    Uses facebook/bart-large-mnli for zero-shot classification.
    """
    if not text or not labels:
        return None
    result = classifier(text, labels, multi_label=False)
    return result['labels'][0]

# Register as a Spark UDF
@F.udf(StringType())
def ai_classify_udf(text, labels_str):
    if text is None:
        return None
    labels = labels_str.split(",")
    return ai_classify_fn(text, labels)

# Apply to a sample (full 1000 would be slow on CPU — batch inference pattern in NB09)
sample_tickets = spark.table("default.support_tickets_bronze").limit(50)

classified = sample_tickets.withColumn(
    "predicted_category",
    ai_classify_udf(F.col("ticket_text"), F.lit("billing,technical,shipping,general"))
)

print("=" * 70)
print("  ai_classify — Zero-shot ticket classification")
print("=" * 70)
classified.select("ticket_id", "ticket_text", "true_category", "predicted_category").show(10, truncate=50)

# Calculate accuracy
classified_pdf = classified.select("true_category", "predicted_category").toPandas()
accuracy = (classified_pdf['true_category'] == classified_pdf['predicted_category']).mean()
print(f"  Accuracy: {accuracy:.1%} (zero-shot, no training data needed!)")
print(f"\n📌 SQL equivalent on full Databricks:")
print(f"   SELECT ai_classify(ticket_text, ARRAY('billing','technical','shipping','general'))")
print(f"   FROM support_tickets;")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 ai_extract — Structured Entity Extraction

# COMMAND ----------

# ============================================================================
# ai_extract UDF: Pull structured fields from unstructured text
# ============================================================================

def ai_extract_fn(text: str, fields: list) -> dict:
    """
    Open-source equivalent of Databricks ai_extract().
    Uses regex patterns to extract common entity types.
    On full Databricks, this uses an LLM for much more robust extraction.
    """
    result = {}
    text_lower = text.lower() if text else ""
    
    for field in fields:
        field_lower = field.lower()
        if field_lower in ('amount', 'dollar_amount', 'price', 'cost'):
            match = re.search(r'\$(\d+\.?\d*)', text or "")
            result[field] = float(match.group(1)) if match else None
        elif field_lower in ('date', 'event_date'):
            match = re.search(r'(\d{4}-\d{2}-\d{2})', text or "")
            result[field] = match.group(1) if match else None
        elif field_lower in ('order_id', 'order_number'):
            match = re.search(r'#?(\d{6})', text or "")
            result[field] = match.group(1) if match else None
        elif field_lower in ('tracking_number', 'tracking'):
            match = re.search(r'(1Z\d{10})', text or "")
            result[field] = match.group(1) if match else None
        elif field_lower in ('email',):
            match = re.search(r'[\w.-]+@[\w.-]+\.\w+', text or "")
            result[field] = match.group(0) if match else None
        elif field_lower in ('sentiment', 'tone'):
            # Simple sentiment based on keywords
            positive_words = ['love', 'great', 'excellent', 'thanks', 'appreciate']
            negative_words = ['frustrated', 'angry', 'terrible', 'broken', 'wrong', 'never']
            pos_count = sum(1 for w in positive_words if w in text_lower)
            neg_count = sum(1 for w in negative_words if w in text_lower)
            if pos_count > neg_count:
                result[field] = 'positive'
            elif neg_count > pos_count:
                result[field] = 'negative'
            else:
                result[field] = 'neutral'
        else:
            result[field] = None
    return result

@F.udf(StringType())
def ai_extract_udf(text, fields_str):
    if text is None:
        return None
    fields = fields_str.split(",")
    return json.dumps(ai_extract_fn(text, fields))

# Apply extraction
extracted = sample_tickets.withColumn(
    "extracted",
    ai_extract_udf(F.col("ticket_text"), F.lit("amount,date,order_id,tracking_number,sentiment"))
)

print("=" * 70)
print("  ai_extract — Structured entity extraction")
print("=" * 70)
extracted.select("ticket_id", "ticket_text", "extracted").show(8, truncate=60)

print(f"\n📌 SQL equivalent on full Databricks:")
print(f"   SELECT ai_extract(ticket_text, ARRAY('amount','date','order_id','sentiment'))")
print(f"   FROM support_tickets;")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 ai_similarity — Semantic Similarity Score

# COMMAND ----------

# ============================================================================
# ai_similarity UDF: Cosine similarity between two texts
# ============================================================================
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

@F.udf(FloatType())
def ai_similarity_udf(text1, text2):
    if text1 is None or text2 is None:
        return None
    emb1 = embedding_model.encode([text1])
    emb2 = embedding_model.encode([text2])
    return float(cos_sim(emb1, emb2)[0][0])

# Find duplicate/similar tickets
from itertools import combinations

similar_pairs = spark.table("default.support_tickets_bronze").limit(20)

# Compare first ticket against next 5
first_ticket = similar_pairs.collect()[0]['ticket_text']
similarity_check = (
    similar_pairs.limit(10)
    .withColumn("reference_ticket", F.lit(first_ticket))
    .withColumn("similarity_score",
                ai_similarity_udf(F.col("reference_ticket"), F.col("ticket_text")))
    .orderBy(F.col("similarity_score").desc())
)

print("=" * 70)
print("  ai_similarity — Semantic similarity scoring")
print("=" * 70)
print(f"  Reference: \"{first_ticket[:60]}...\"")
print()
similarity_check.select("ticket_id", "ticket_text", "similarity_score").show(10, truncate=55)

print(f"\n📌 SQL equivalent on full Databricks:")
print(f"   SELECT ai_similarity(t1.text, t2.text) AS sim_score")
print(f"   FROM tickets t1 CROSS JOIN tickets t2")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 3: Structured Outputs (Handbook I6)
# MAGIC
# MAGIC > **The problem**: A plain LLM call returns free-form text. If a downstream pipeline
# MAGIC > expects reliably parseable JSON, "usually well-formed" isn't good enough.
# MAGIC >
# MAGIC > **Structured outputs** force the model to conform to a specific JSON schema.

# COMMAND ----------

# ============================================================================
# STRUCTURED OUTPUTS: Force JSON-schema-compliant extraction
# ============================================================================

# Define a target schema
ticket_schema = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": ["billing", "technical", "shipping", "general"]},
        "priority_estimate": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
        "dollar_amount": {"type": "number", "nullable": True},
        "order_id": {"type": "string", "nullable": True},
        "action_required": {"type": "string"},
        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
    },
    "required": ["category", "priority_estimate", "action_required", "sentiment"]
}

def extract_structured(text: str, schema: dict) -> dict:
    """
    Simulates structured output extraction.
    On full Databricks, this would use:
    
    SELECT ai_query(
        'my-endpoint',
        'Extract fields from: ' || ticket_text,
        responseFormat => '{"type":"json_schema","json_schema":...}'
    )
    
    The responseFormat parameter FORCES the model to output valid JSON matching the schema.
    """
    # Use our extraction function + classification to produce structured output
    category = ai_classify_fn(text, ["billing", "technical", "shipping", "general"])
    entities = ai_extract_fn(text, ["amount", "order_id", "sentiment"])
    
    # Determine priority based on keywords
    urgent_words = ['immediately', 'urgent', 'asap', 'critical', 'emergency']
    high_words = ['broken', 'crashed', 'lost', 'wrong', 'damaged', 'soaked']
    text_lower = text.lower()
    
    if any(w in text_lower for w in urgent_words):
        priority = "urgent"
    elif any(w in text_lower for w in high_words):
        priority = "high"
    elif entities.get('amount') and entities['amount'] > 50:
        priority = "medium"
    else:
        priority = "low"
    
    # Build structured output conforming to schema
    result = {
        "category": category,
        "priority_estimate": priority,
        "dollar_amount": entities.get('amount'),
        "order_id": entities.get('order_id'),
        "action_required": f"Route to {category} team" + (" (escalate)" if priority in ['high', 'urgent'] else ""),
        "sentiment": entities.get('sentiment', 'neutral'),
    }
    
    # Validate against schema (in practice, the model serving layer enforces this)
    for required_field in schema.get('required', []):
        if required_field not in result or result[required_field] is None:
            result[required_field] = "unknown"
    
    return result

# Demo structured extraction
print("=" * 70)
print("  STRUCTURED OUTPUTS — Schema-enforced extraction")
print("=" * 70)
print(f"\n  Target Schema:")
print(f"  {json.dumps(ticket_schema, indent=2)[:300]}...")

sample_texts = [
    "I was charged $99.99 but my plan costs $24.99. I need a refund immediately!",
    "The app crashes every time I upload a file larger than 50MB.",
    "My order #456789 arrived damaged. The box was crushed.",
]

for text in sample_texts:
    result = extract_structured(text, ticket_schema)
    print(f"\n  Input: \"{text[:65]}...\"")
    print(f"  Output: {json.dumps(result, indent=2)}")

print(f"\n📌 On full Databricks (structured outputs via ai_query):")
print(f"   SELECT ai_query(")
print(f"       'my-endpoint',")
print(f"       'Extract: ' || ticket_text,")
print(f"       responseFormat => '{{\"type\":\"json_schema\",...}}'")
print(f"   ) FROM tickets;")
print(f"\n   The responseFormat forces EVERY output to match the schema — no parsing failures.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 4: Medallion Architecture with AI Enrichment
# MAGIC
# MAGIC AI Functions fit naturally into the **medallion architecture**:
# MAGIC - **Bronze**: Raw support tickets (as ingested)
# MAGIC - **Silver**: Classified, extracted, enriched tickets
# MAGIC - **Gold**: Aggregated metrics by category/priority

# COMMAND ----------

# ============================================================================
# MEDALLION PIPELINE: Bronze → Silver (AI-enriched) → Gold (aggregated)
# ============================================================================

# Bronze already exists: default.support_tickets_bronze

# Silver: Enrich with AI functions (using pre-computed results for speed)
bronze = spark.table("default.support_tickets_bronze")

# For demo, use the true_category as our "AI classification" result
# (In production, this would be the ai_classify UDF output)
silver = (
    bronze
    .withColumn("ai_category", F.col("true_category"))  # Simulating ai_classify output
    .withColumn("extracted_json",
                ai_extract_udf(F.col("ticket_text"), F.lit("amount,date,order_id,sentiment")))
    .withColumn("word_count", F.size(F.split(F.col("ticket_text"), " ")))
    .withColumn("has_dollar_amount",
                F.col("ticket_text").contains("$").cast("boolean"))
    .withColumn("processed_at", F.current_timestamp())
)

silver.write.format("delta").mode("overwrite").saveAsTable("default.support_tickets_silver")
print("✅ Silver table: default.support_tickets_silver")
silver.select("ticket_id", "ai_category", "priority", "extracted_json", "word_count").show(5, truncate=50)

# COMMAND ----------

# Gold: Aggregated metrics
gold = (
    silver
    .groupBy("ai_category", "priority")
    .agg(
        F.count("*").alias("ticket_count"),
        F.avg("word_count").alias("avg_ticket_length"),
        F.sum(F.when(F.col("has_dollar_amount"), 1).otherwise(0)).alias("tickets_with_amount"),
    )
    .orderBy("ai_category", "priority")
)

gold.write.format("delta").mode("overwrite").saveAsTable("default.support_tickets_gold")
print("✅ Gold table: default.support_tickets_gold")
gold.show(20)

print(f"\n📌 On full Databricks, this entire pipeline would be a Lakeflow Declarative Pipeline:")
print(f"   @dp.table")
print(f"   def silver_tickets():")
print(f"       return spark.table('bronze_tickets').selectExpr(")
print(f"           '*',")
print(f"           \"ai_classify(ticket_text, array('billing','technical','shipping')) as category\"")
print(f"       )")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 5: Multimodal AI Concepts (Handbook I9)
# MAGIC
# MAGIC > `ai_parse_document()` is a specialized AI Function for extracting structured content
# MAGIC > from document files (PDFs, scanned forms) — tables, layout, text.
# MAGIC >
# MAGIC > Vision-capable Foundation Models accept IMAGE input alongside text.
# MAGIC >
# MAGIC > Both rely on the **FILE type** and **Volumes** for governed unstructured data storage.

# COMMAND ----------

# ============================================================================
# MULTIMODAL CONCEPTS (architecture explanation — CE can't run these)
# ============================================================================

print("=" * 70)
print("  MULTIMODAL AI ON DATABRICKS (Handbook I9)")
print("=" * 70)
print("""
  ┌──────────────────────────────────────────────────────────────────────┐
  │  Unstructured Data Pipeline (B1)                                     │
  │                                                                      │
  │  1. Raw PDFs/images land in VOLUMES (governed Unity Catalog objects)  │
  │     └── Same access controls as tables                               │
  │                                                                      │
  │  2. ai_parse_document() extracts structured data FROM documents       │
  │     SELECT ai_parse_document(file_content) FROM volume_files          │
  │     └── Tables, layout, text → structured columns                    │
  │                                                                      │
  │  3. FILE type (Beta) lets Delta tables hold unstructured data         │
  │     as a proper COLUMN type alongside structured columns              │
  │     └── No more "path reference" workarounds                         │
  │                                                                      │
  │  4. Vision-capable models accept IMAGE input alongside text           │
  │     └── Classify product photos, extract from scanned documents      │
  └──────────────────────────────────────────────────────────────────────┘

  WHY THIS MATTERS FOR A DATA ENGINEER:
  - Unstructured data (PDFs, images) gets the SAME governance as tables
  - AI enrichment fits into the medallion pipeline naturally
  - No separate OCR/parsing library needed — it's a SQL function
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # ✅ Key Takeaways
# MAGIC
# MAGIC | Concept | What We Did | Handbook Section |
# MAGIC |---------|------------|------------------|
# MAGIC | ai_classify | Zero-shot ticket classification UDF | B4 |
# MAGIC | ai_extract | Structured entity extraction UDF | B4 |
# MAGIC | ai_similarity | Semantic similarity scoring UDF | B4 |
# MAGIC | ai_summarize | Text condensation concept | B4 |
# MAGIC | Structured Outputs | Schema-enforced JSON extraction | I6 |
# MAGIC | Medallion + AI | Bronze → Silver (AI-enriched) → Gold pipeline | B4, I8 |
# MAGIC | Multimodal | ai_parse_document, FILE type, Volumes concepts | I9 |
# MAGIC
# MAGIC ### 🔗 Full Platform Mapping
# MAGIC | CE Approach | Full Databricks |
# MAGIC |---|---|
# MAGIC | `bart-large-mnli` zero-shot UDF | `ai_classify(text, ARRAY(...))` SQL function |
# MAGIC | Regex extraction UDF | `ai_extract(text, ARRAY(...))` SQL function |
# MAGIC | `SentenceTransformer` cosine UDF | `ai_similarity(text1, text2)` SQL function |
# MAGIC | Manual JSON validation | `responseFormat` parameter (schema enforcement) |
# MAGIC | Spark UDF in PySpark | `ai_query(endpoint, prompt)` in pure SQL |
# MAGIC
# MAGIC **Next**: Notebook 06 — Building Agents with tool calling, ReAct loops, and MCP concepts.
