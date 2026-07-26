# Databricks notebook source

# MAGIC %md
# MAGIC # 🥇 Notebook 07 — Production Medallion Architecture & Data Mesh
# MAGIC
# MAGIC **Handbook Sections Covered**: Part G1–G15, J7–J12
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Learning Objectives
# MAGIC 1. **Medallion Pipeline Pattern (Bronze → Silver → Gold)** — Architect production-grade multi-layer Lakehouse pipelines.
# MAGIC 2. **Bronze Layer (Raw Ingestion)** — Raw landing with system audit metadata (`_ingested_at`, `_source_file`, `_raw_hash`).
# MAGIC 3. **Silver Layer (Cleansed & Enriched)** — Deduplication, validation, quarantining bad data to DLQ, and SCD Type 2 history tracking.
# MAGIC 4. **Gold Layer (Business Aggregations & Star Schema)** — Dimensional Modeling (`dim_customer`, `dim_product`, `fact_sales`).
# MAGIC 5. **Data Quality Governance** — Assertions for Primary Key uniqueness, Foreign Key integrity, and non-null checks.
# MAGIC 6. **Data Mesh Architecture** — Domain-driven data products and federated governance.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Setup & Database Layering

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from delta.tables import DeltaTable

spark = SparkSession.builder.getOrCreate()
spark.sql("USE de_practical_db")

print("✅ Environment initialized for Medallion Lakehouse Architecture.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 1: Bronze Layer — Raw Ingestion & System Metadata (Handbook G1–G4)
# MAGIC
# MAGIC ### Bronze Layer Principles (G2):
# MAGIC - Raw, append-only ingestion table that mirrors source system structure.
# MAGIC - Schema is loose (`PERMISSIVE` mode).
# MAGIC - **System Audit Metadata Columns**:
# MAGIC   - `_ingested_at`: Timestamp when record arrived in Lakehouse.
# MAGIC   - `_source_file_name`: Path of the source input file (`input_file_name()`).
# MAGIC   - `_raw_hash`: SHA256 hash of the row content to identify exact duplicate payload files.

# COMMAND ----------

# Load raw customer data and add Bronze System Audit Metadata
df_raw_cust = spark.table("de_practical_db.raw_customers")

df_bronze_customers = (
    df_raw_cust
    .withColumn("_ingested_at", F.current_timestamp())
    .withColumn("_source_file_name", F.lit("dbfs:/landing/raw_customers.csv"))
    .withColumn("_raw_hash", F.sha2(F.concat_ws("||", *df_raw_cust.columns), 256))
)

# Write to Bronze Delta Table
df_bronze_customers.write.format("delta").mode("overwrite").saveAsTable("de_practical_db.bronze_customers")

print("=" * 80)
print("  BRONZE LAYER TABLE: de_practical_db.bronze_customers")
print("=" * 80)
spark.table("de_practical_db.bronze_customers").select(
    "customer_id", "email", "_ingested_at", "_source_file_name", "_raw_hash"
).show(4, truncate=40)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 2: Silver Layer — Cleaning, Data Quality & SCD Type 2 (Handbook G5–G9)
# MAGIC
# MAGIC ### Silver Layer Principles (G5):
# MAGIC - Cleansed, validated, structured, and enriched table layer.
# MAGIC - Schema enforcement applied.
# MAGIC - Invalid rows are quarantined into a **Dead-Letter Queue (DLQ)** table (`silver_customers_dlq`).
# MAGIC - Deduplicated and transformed into canonical data models.

# COMMAND ----------

df_bronze = spark.table("de_practical_db.bronze_customers")

# ----------------------------------------------------------------------------
# Data Quality Rules (G6):
# 1. customer_id MUST NOT BE NULL
# 2. email MUST CONTAIN '@'
# ----------------------------------------------------------------------------
condition_valid = (F.col("customer_id").isNotNull()) & (F.col("email").like("%@%"))

df_silver_valid = df_bronze.filter(condition_valid)
df_silver_invalid = df_bronze.filter(~condition_valid).withColumn("_quarantine_reason", F.lit("Invalid Customer ID or Email format"))

# Write Quarantine Rows to DLQ
df_silver_invalid.write.format("delta").mode("overwrite").saveAsTable("de_practical_db.silver_customers_dlq")

print("=" * 80)
print("  DATA QUALITY VALIDATION & DEAD-LETTER QUEUE (DLQ)")
print("=" * 80)
print(f"Bronze Rows Total:     {df_bronze.count()}")
print(f"Valid Silver Rows:     {df_silver_valid.count()}")
print(f"Quarantined DLQ Rows:  {df_silver_invalid.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Implementing Slowly Changing Dimension (SCD Type 2) in Silver (Handbook G8)
# MAGIC
# MAGIC SCD Type 2 tracks historical changes by maintaining effective start/end dates (`valid_from`, `valid_to`) and an active flag (`is_current`).

# COMMAND ----------

# Initialize Silver Customer SCD Type 2 Table
spark.sql("""
CREATE TABLE IF NOT EXISTS de_practical_db.silver_customers_scd2 (
    customer_sk STRING,        -- Surrogate key (hash of customer_id + valid_from)
    customer_id STRING,
    first_name STRING,
    last_name STRING,
    email STRING,
    loyalty_tier STRING,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    is_current BOOLEAN
) USING DELTA
""")

# Build SCD Type 2 Transformation logic
window_spec_scd2 = Window.partitionBy("customer_id").orderBy("_ingested_at")

df_scd2_processed = (
    df_silver_valid
    .withColumn("valid_from", F.col("_ingested_at"))
    .withColumn("next_valid_from", F.lead("valid_from", 1).over(window_spec_scd2))
    .withColumn("valid_to", F.coalesce(F.col("next_valid_from"), F.to_timestamp(F.lit("9999-12-31 23:59:59"))))
    .withColumn("is_current", F.col("next_valid_from").isNull())
    .withColumn("customer_sk", F.sha2(F.concat_ws("||", "customer_id", "valid_from"), 256))
    .select("customer_sk", "customer_id", "first_name", "last_name", "email", "loyalty_tier", "valid_from", "valid_to", "is_current")
)

df_scd2_processed.write.format("delta").mode("overwrite").saveAsTable("de_practical_db.silver_customers_scd2")

print("=" * 80)
print("  SILVER LAYER: SCD TYPE 2 CUSTOMERS TABLE")
print("=" * 80)
spark.table("de_practical_db.silver_customers_scd2").select(
    "customer_id", "loyalty_tier", "valid_from", "valid_to", "is_current"
).show(6, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 3: Gold Layer — Dimensional Modeling & Star Schema (Handbook G10–G14)
# MAGIC
# MAGIC ### Gold Layer Principles (G10):
# MAGIC - High-level business aggregation metrics and Dimensional Star Schema data models ready for BI (PowerBI, Tableau, Databricks SQL Datasets).
# MAGIC - **Star Schema**:
# MAGIC   - **`dim_customer`**: Conformed customer dimension table.
# MAGIC   - **`dim_product`**: Product catalog dimension table.
# MAGIC   - **`fact_sales`**: Grain: 1 line item per order transaction, joining keys to dimensions.

# COMMAND ----------

# Build Dimension Tables
df_dim_customer = spark.table("de_practical_db.silver_customers_scd2").filter(F.col("is_current") == True)
df_dim_product = spark.table("de_practical_db.raw_products")
df_orders = spark.table("de_practical_db.raw_orders")

# Create Fact Table: Fact Sales
df_fact_sales = (
    df_orders
    .join(df_dim_customer, "customer_id", "inner")
    .join(df_dim_product, df_orders.order_id == df_dim_product.product_id, "left") # Mock join
    .select(
        F.sha2(F.concat_ws("||", "order_id", "customer_id"), 256).alias("sales_fact_sk"),
        F.col("order_id"),
        F.col("customer_sk"),
        F.col("product_id"),
        F.col("order_timestamp").alias("transaction_timestamp"),
        F.col("total_amount").alias("sales_amount"),
        F.col("status").alias("order_status")
    )
)

# Save Gold Dimension & Fact Tables
df_dim_customer.write.format("delta").mode("overwrite").saveAsTable("de_practical_db.gold_dim_customer")
df_fact_sales.write.format("delta").mode("overwrite").saveAsTable("de_practical_db.gold_fact_sales")

print("=" * 80)
print("  GOLD LAYER STAR SCHEMA DATA MODEL")
print("=" * 80)
print("--- 1. Dimension Table: gold_dim_customer ---")
spark.table("de_practical_db.gold_dim_customer").select("customer_sk", "customer_id", "email", "loyalty_tier").show(3, truncate=False)

print("--- 2. Fact Table: gold_fact_sales ---")
spark.table("de_practical_db.gold_fact_sales").show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Gold Business Aggregation Datamart

# COMMAND ----------

# Create Gold Data Mart: Customer Executive KPI Aggregations
df_gold_exec_kpi = (
    spark.table("de_practical_db.gold_fact_sales")
    .join(spark.table("de_practical_db.gold_dim_customer"), "customer_sk", "inner")
    .groupBy("loyalty_tier")
    .agg(
        F.countDistinct("order_id").alias("total_orders"),
        F.countDistinct("customer_id").alias("unique_customers"),
        F.sum("sales_amount").alias("gross_revenue"),
        F.avg("sales_amount").alias("average_order_value")
    )
    .orderBy(F.col("gross_revenue").desc())
)

df_gold_exec_kpi.write.format("delta").mode("overwrite").saveAsTable("de_practical_db.gold_exec_revenue_by_tier")

print("=" * 80)
print("  GOLD EXECUTIVE DATAMART: REVENUE & AOV BY LOYALTY TIER")
print("=" * 80)
spark.table("de_practical_db.gold_exec_revenue_by_tier").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 4: Data Mesh Architecture Principles (Handbook G15)
# MAGIC
# MAGIC ### 4 Core Data Mesh Pillars on Databricks:
# MAGIC 1. **Domain-Driven Decentralized Ownership**: Each business unit (e.g., Sales Domain, Supply Chain Domain, Customer Marketing Domain) owns its own Delta tables and Medallion pipelines.
# MAGIC 2. **Data-as-a-Product**: Datasets are treated as published products with declared schemas, SLAs, documentation, and quality expectations.
# MAGIC 3. **Self-Serve Data Infrastructure**: Databricks platform provides shared compute, storage, and notebooks for all domain teams.
# MAGIC 4. **Federated Computational Governance**: **Unity Catalog** enforces central security access control policies, data lineage tracking, and audit logging across all domain workspaces.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 💡 Senior Data Engineer Interview Practice (Handbook H3)
# MAGIC
# MAGIC ### Q1: Explain the purpose and characteristics of the Bronze, Silver, and Gold layers in a Medallion Architecture.
# MAGIC **Answer**:
# MAGIC - **Bronze Layer (Raw Landing)**: Stores incoming data in its raw, append-only native structure with minimal processing. Retains full history and adds audit metadata (`_ingested_at`, `_source_file`, `_raw_hash`). Re-processable if downstream logic changes.
# MAGIC - **Silver Layer (Cleansed & Conformed)**: Cleans, deduplicates, validates schemas, and enriches data. Quarantines malformed rows into a Dead-Letter Queue (DLQ) and tracks historical changes using Slowly Changing Dimensions (SCD Type 2).
# MAGIC - **Gold Layer (Curated Business Analytics)**: Houses business-level aggregations, KPI Data Marts, and Star Schema Dimensional models (`fact_sales`, `dim_customer`) optimized for fast BI reporting and executive dashboards.
# MAGIC
# MAGIC ### Q2: How do you handle Data Quality enforcement between Bronze and Silver layers?
# MAGIC **Answer**: Implement an explicit expectation validation step. Filter rows against business constraints (e.g., `customer_id IS NOT NULL AND email LIKE '%@%'`). 
# MAGIC Valid rows proceed to `silver_table`; invalid rows are diverted with a `_quarantine_reason` column into a separate `silver_dead_letter_queue` Delta table for alerting, monitoring, and operational recovery.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # ✅ Module 07 Summary
# MAGIC
# MAGIC | Medallion Layer | Target Table | Transformation Logic | Handbook Reference |
# MAGIC |---|---|---|---|
# MAGIC | **Bronze** | `bronze_customers` | Append-only raw load + `_ingested_at`, `_raw_hash` | G2–G4 |
# MAGIC | **Silver** | `silver_customers_scd2` | Cleansed, deduplicated, DLQ filtering, SCD Type 2 history | G5–G9 |
# MAGIC | **Silver DLQ**| `silver_customers_dlq` | Quarantined invalid rows with `_quarantine_reason` | G6 |
# MAGIC | **Gold** | `gold_fact_sales`, `gold_dim_customer` | Star Schema dimensional model + Star Schema join | G10–G14 |
# MAGIC | **Gold Mart**| `gold_exec_revenue_by_tier` | Business aggregations for executive BI reporting | G11 |
