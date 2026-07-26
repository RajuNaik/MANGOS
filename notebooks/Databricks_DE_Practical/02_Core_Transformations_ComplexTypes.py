# Databricks notebook source

# MAGIC %md
# MAGIC # 🛠️ Notebook 02 — Core Transformations & Complex Data Types
# MAGIC
# MAGIC **Handbook Sections Covered**: Part B7–B10, B13–B18, B21–B24, B26
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Learning Objectives
# MAGIC 1. **Robust Null Handling** — `coalesce`, `fillna`, `dropna`, and null-safe equality (`<=>`).
# MAGIC 2. **Python UDFs vs Vectorized Pandas UDFs** — Compare performance bottlenecks and Apache Arrow vectorization.
# MAGIC 3. **Complex Type Manipulation** — Arrays, Maps, Structs (`explode`, `posexplode`, `create_map`, `map_keys`, `collect_list`).
# MAGIC 4. **Enterprise Window Functions** — `lead`/`lag`, ranking (`row_number`, `rank`, `dense_rank`), cumulative sums, dedup.
# MAGIC 5. **Column Projection Efficiency** — `select` vs chaining multiple `withColumn` operations.
# MAGIC 6. **Column-Level PII Encryption** — Symmetric Fernet encryption for data privacy compliance.
# MAGIC 7. **JSON & Direct SQL Querying** — `from_json`, `to_json`, and `spark.sql()` direct DataFrame querying.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Setup & Data Loading

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType, 
    ArrayType, MapType, TimestampType, DateType
)
from pyspark.sql.window import Window
import pandas as pd
import numpy as np

spark = SparkSession.builder.getOrCreate()
spark.sql("USE de_practical_db")

# Load baseline tables created in Notebook 01
df_customers = spark.table("de_practical_db.raw_customers")
df_orders = spark.table("de_practical_db.raw_orders")
df_products = spark.table("de_practical_db.raw_products")
df_clickstream = spark.table("de_practical_db.raw_clickstream")

print("✅ Data loaded from de_practical_db baseline tables.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 1: Null Handling & Null-Safe Comparisons (Handbook B7, B17)
# MAGIC
# MAGIC In Spark SQL:
# MAGIC - `NULL == NULL` evaluates to **`NULL`** (Unknown), NOT `TRUE`!
# MAGIC - `col("a") == col("b")` will filter out rows where both columns are `null`.
# MAGIC - **Null-Safe Equality Operator (`<=>`)**: Evaluates `NULL <=> NULL` to **`TRUE`**.

# COMMAND ----------

# Create a sample DataFrame with Nulls
data_nulls = [
    (1, "ORD_1001", "COMPLETED", 150.0),
    (2, "ORD_1002", None, 200.0),
    (3, "ORD_1003", "PENDING", None),
    (4, None, None, None)
]
df_null_demo = spark.createDataFrame(data_nulls, ["id", "order_ref", "status", "amount"])

print("=" * 80)
print("  NULL HANDLING & COALESCE DEMONSTRATION")
print("=" * 80)

# Coalesce: Returns the first non-null value among arguments
df_coalesced = df_null_demo.select(
    "id",
    "status",
    F.coalesce(F.col("status"), F.lit("UNKNOWN_STATUS")).alias("status_cleaned"),
    F.coalesce(F.col("amount"), F.lit(0.0)).alias("amount_cleaned")
)
df_coalesced.show()

# Null-Safe Equality Check (<=>)
print("--- Null-Safe Equality Check (<=>) ---")
df_null_demo.select(
    "id",
    "status",
    (F.col("status") == F.lit(None)).alias("standard_equals_null"),
    (F.col("status").eqNullSafe(F.lit(None))).alias("null_safe_equals_null") # Equivalent to <=>
).show()

# Dropna & Fillna
print("--- Fillna & Dropna ---")
df_null_demo.fillna({"status": "N/A", "amount": -1.0}).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 2: Python UDFs vs Vectorized Pandas UDFs (Handbook B8, B16)
# MAGIC
# MAGIC ### Performance Warning (B8):
# MAGIC - **Standard Python UDF (`@udf`)**: Serializes JVM rows -> Python process -> executes row-by-row -> serializes back to JVM. **Extremely slow (10x-100x overhead)** and disables Catalyst Optimizer vectorization.
# MAGIC - **Pandas UDF (`@pandas_udf` / Vectorized UDF)**: Uses **Apache Arrow** to transfer data between JVM and Python in columnar memory batches. Executes operations as vectorized Pandas/Numpy code.

# COMMAND ----------

from pyspark.sql.functions import udf, pandas_udf
import time

# ----------------------------------------------------------------------------
# 1. Standard Python UDF (Row-by-Row Serialization)
# ----------------------------------------------------------------------------
@udf(returnType=StringType())
def calculate_discount_tier_py(spend):
    if spend is None:
        return "UNKNOWN"
    elif spend > 300.0:
        return "TIER_1_VIP"
    elif spend > 100.0:
        return "TIER_2_REGULAR"
    else:
        return "TIER_3_LOW"

# ----------------------------------------------------------------------------
# 2. Vectorized Pandas UDF (Apache Arrow Columnar Transfer)
# ----------------------------------------------------------------------------
@pandas_udf(StringType())
def calculate_discount_tier_pd(spend_series: pd.Series) -> pd.Series:
    conditions = [
        spend_series > 300.0,
        spend_series > 100.0,
        spend_series.notna()
    ]
    choices = ["TIER_1_VIP", "TIER_2_REGULAR", "TIER_3_LOW"]
    return pd.Series(np.select(conditions, choices, default="UNKNOWN"))

# Measure Execution Speed Difference
df_benchmark = df_customers.select("customer_id", "lifetime_spend")

start_time = time.time()
df_benchmark.withColumn("tier", calculate_discount_tier_py(F.col("lifetime_spend"))).write.mode("overwrite").format("noop").save()
py_udf_time = time.time() - start_time

start_time = time.time()
df_benchmark.withColumn("tier", calculate_discount_tier_pd(F.col("lifetime_spend"))).write.mode("overwrite").format("noop").save()
pd_udf_time = time.time() - start_time

print("=" * 80)
print("  UDF BENCHMARK: Python UDF vs Vectorized Pandas UDF (Apache Arrow)")
print("=" * 80)
print(f"Standard Python UDF Time:    {py_udf_time:.4f} seconds")
print(f"Vectorized Pandas UDF Time:  {pd_udf_time:.4f} seconds")
print(f"⚡ Pandas UDF Speedup:       {py_udf_time / max(pd_udf_time, 0.0001):.2f}x faster!")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 3: Complex Data Types: Arrays & Maps (Handbook B9, B23, B24)
# MAGIC
# MAGIC In modern Lakehouse architectures, semi-structured columns (`ArrayType`, `MapType`) are common in clickstream, IoT, and JSON payloads.

# COMMAND ----------

# Extract nested JSON array and explode
df_exploded = (
    df_clickstream
    .select("event_id", "customer_id", "page_events")
    .withColumn("single_page_event", F.explode("page_events")) # explode array into multiple rows
    .select(
        "event_id",
        "customer_id",
        F.col("single_page_event.page").alias("page_name"),
        F.col("single_page_event.action").alias("user_action"),
        F.col("single_page_event.duration_sec").alias("page_duration")
    )
)

print("=" * 80)
print("  EXPLODING NESTED ARRAYS (B23)")
print("=" * 80)
df_exploded.show(6, truncate=False)

# Map Creation & Manipulation (B24)
df_map_demo = df_customers.select(
    "customer_id",
    F.create_map(
        F.lit("tier"), F.col("loyalty_tier"),
        F.lit("email"), F.col("email")
    ).alias("customer_attributes_map")
)

print("--- MapType Creation & Access (B24) ---")
df_map_demo.select(
    "customer_id",
    "customer_attributes_map",
    F.col("customer_attributes_map")["tier"].alias("extracted_tier"),
    F.map_keys("customer_attributes_map").alias("map_keys")
).show(4, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 4: Enterprise Window Functions (Handbook B18)
# MAGIC
# MAGIC Window functions are critical for sessionization, running totals, and deduplication.
# MAGIC
# MAGIC Key Window Expressions:
# MAGIC - `row_number()`: Unique integer per row (1, 2, 3, 4) - Ideal for deduplication.
# MAGIC - `rank()`: Rank with gaps on ties (1, 2, 2, 4).
# MAGIC - `dense_rank()`: Rank without gaps on ties (1, 2, 2, 3).
# MAGIC - `lead(col, 1)` / `lag(col, 1)`: Access next / previous row value in partition.

# COMMAND ----------

# ============================================================================
# B18: Window Functions — Customer Order History Analysis
# ============================================================================

window_spec_cust = Window.partitionBy("customer_id").orderBy("order_timestamp")

df_orders_windowed = (
    df_orders
    .withColumn("order_rank", F.row_number().over(window_spec_cust))
    .withColumn("prev_order_amount", F.lag("total_amount", 1).over(window_spec_cust))
    .withColumn("next_order_timestamp", F.lead("order_timestamp", 1).over(window_spec_cust))
    .withColumn("running_total_spend", F.sum("total_amount").over(window_spec_cust))
)

print("=" * 80)
print("  WINDOW FUNCTIONS: Ranking, Lead/Lag, Running Total (B18)")
print("=" * 80)
df_orders_windowed.select(
    "customer_id", "order_id", "order_timestamp", 
    "total_amount", "order_rank", "prev_order_amount", "running_total_spend"
).show(10, truncate=False)

# ----------------------------------------------------------------------------
# Deduplication Pattern via Window Row_Number
# ----------------------------------------------------------------------------
print("--- Deduplication Pattern: Keep Most Recent Order per Customer ---")
window_latest = Window.partitionBy("customer_id").orderBy(F.col("order_timestamp").desc())
df_dedup = (
    df_orders
    .withColumn("row_num", F.row_number().over(window_latest))
    .filter(F.col("row_num") == 1)
    .drop("row_num")
)
print(f"Total Raw Orders: {df_orders.count()} | Deduplicated Latest Orders per Customer: {df_dedup.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 5: Projection Optimization: `select` vs Chaining `withColumn` (Handbook B14)
# MAGIC
# MAGIC ### Performance Trap (B14):
# MAGIC Chaining 20 `withColumn()` calls generates **20 separate projection stages** in the Catalyst logical plan. This inflates plan compilation time and stack memory usage.
# MAGIC
# MAGIC **Best Practice**: Replace multi-chained `withColumn` calls with a single `select()` or `selectExpr()`.

# COMMAND ----------

# ❌ POOR PATTERN (Chaining multiple withColumn calls)
df_poor_projection = (
    df_customers
    .withColumn("full_name", F.concat_ws(" ", "first_name", "last_name"))
    .withColumn("email_domain", F.split("email", "@")[1])
    .withColumn("is_vip", F.when(F.col("loyalty_tier") == "Platinum", True).otherwise(False))
    .withColumn("spend_rounded", F.round("lifetime_spend", 0))
)

# ✅ OPTIMAL PATTERN (Single select projection)
df_optimal_projection = df_customers.select(
    "*",
    F.concat_ws(" ", "first_name", "last_name").alias("full_name"),
    F.split("email", "@")[1].alias("email_domain"),
    (F.col("loyalty_tier") == "Platinum").alias("is_vip"),
    F.round("lifetime_spend", 0).alias("spend_rounded")
)

print("✅ Optimal single-select projection compiled cleanly!")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 6: Column-Level PII Encryption (Handbook B15)
# MAGIC
# MAGIC Compliance frameworks (GDPR, HIPAA, PCI-DSS) require encrypting PII (Personally Identifiable Information) columns at rest. We use `cryptography.fernet` symmetric encryption.

# COMMAND ----------

from cryptography.fernet import Fernet

# Generate a symmetric encryption key
encryption_key = Fernet.generate_key()
cipher_suite = Fernet(encryption_key)

# Vectorized Pandas UDFs for PII Encryption & Decryption
@pandas_udf(StringType())
def encrypt_pii_udf(text_series: pd.Series) -> pd.Series:
    return text_series.apply(lambda val: cipher_suite.encrypt(val.encode()).decode() if val else None)

@pandas_udf(StringType())
def decrypt_pii_udf(encrypted_series: pd.Series) -> pd.Series:
    return encrypted_series.apply(lambda val: cipher_suite.decrypt(val.encode()).decode() if val else None)

# Apply encryption to email column
df_encrypted = (
    df_customers
    .select("customer_id", "first_name", "last_name", "email")
    .withColumn("encrypted_email", encrypt_pii_udf(F.col("email")))
    .withColumn("decrypted_email", decrypt_pii_udf(F.col("encrypted_email")))
)

print("=" * 80)
print("  COLUMN-LEVEL SYMMETRIC PII ENCRYPTION (B15)")
print("=" * 80)
df_encrypted.select("customer_id", "email", "encrypted_email", "decrypted_email").show(4, truncate=40)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 7: Reusable Data Engineering Utilities (Handbook B26)

# COMMAND ----------

class DEUtils:
    """Enterprise Data Engineering Utility Methods."""
    
    @staticmethod
    def compare_schemas(df1, df2) -> bool:
        """Returns True if two DataFrames share identical schemas."""
        return df1.schema == df2.schema

    @staticmethod
    def assert_no_duplicates(df, primary_key_cols: list):
        """Raises ValueError if primary key columns contain duplicate entries."""
        dup_count = df.groupBy(primary_key_cols).count().filter(F.col("count") > 1).count()
        if dup_count > 0:
            raise ValueError(f"⚠️ Primary Key Violation! Found {dup_count} duplicate keys in {primary_key_cols}")
        print(f"✅ Primary Key Validation Passed: No duplicates in {primary_key_cols}")

# Execute validation check
DEUtils.assert_no_duplicates(df_dedup, ["customer_id"])

# Save clean transformed table to Delta
df_orders_windowed.write.format("delta").mode("overwrite").saveAsTable("de_practical_db.silver_orders_transformed")
print("✅ Saved de_practical_db.silver_orders_transformed Delta table.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 💡 Senior Data Engineer Interview Practice (Handbook H3)
# MAGIC
# MAGIC ### Q1: Why are standard Python UDFs considered an anti-pattern in PySpark? What is the technical alternative?
# MAGIC **Answer**: Standard Python UDFs treat the Spark execution engine as a black box. Each row must be serialized from JVM memory, sent over an IPC socket to a Python worker process, executed line-by-line in pure Python, and serialized back to the JVM. This destroys Catalyst optimizer pushdowns and vectorization, causing 10x-100x slowdowns.
# MAGIC **Alternative**: Use native PySpark SQL functions (`pyspark.sql.functions`). If custom Python logic is strictly required, use Vectorized Pandas UDFs (`@pandas_udf`), which utilize **Apache Arrow** for zero-copy in-memory columnar transfer and C-optimized vectorized execution.
# MAGIC
# MAGIC ### Q2: Explain the performance difference between chaining multiple `withColumn()` calls versus using a single `select()`.
# MAGIC **Answer**: Each `withColumn()` invocation creates a new projection expression node in the Catalyst logical plan. Chaining 20 `withColumn()` calls creates 20 nested `Project` operators. This inflates Spark driver memory during query planning, slows down Catalyst optimization, and can trigger `StackOverflowError`. A single `select()` or `selectExpr()` evaluates all expressions within a single projection operator.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # ✅ Module 02 Summary
# MAGIC
# MAGIC | Topic | Key Function / Expression | Handbook Reference |
# MAGIC |---|---|---|
# MAGIC | Null Safety | `coalesce()`, `<=>` (`eqNullSafe`) | B7 |
# MAGIC | Vectorized UDFs | `@pandas_udf` + Apache Arrow | B8, B16 |
# MAGIC | Array & Map Unnesting | `explode()`, `create_map()`, `map_keys()` | B9, B23, B24 |
# MAGIC | Window Analytical Functions | `lead()`, `lag()`, `row_number()`, `sum().over()` | B18 |
# MAGIC | Plan Optimization | Single `select()` over chained `withColumn()` | B14 |
# MAGIC | Data Privacy Compliance | Cryptography Fernet Column Encryption | B15 |
