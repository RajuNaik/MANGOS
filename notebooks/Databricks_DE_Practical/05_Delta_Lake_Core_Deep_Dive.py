# Databricks notebook source

# MAGIC %md
# MAGIC # 🧱 Notebook 05 — Delta Lake Core Engine & Deep Dive
# MAGIC
# MAGIC **Handbook Sections Covered**: Part D1–D27, K11–K15
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Learning Objectives
# MAGIC 1. **Delta Transaction Log Architecture** — Inspect `_delta_log` JSON commits, checkpoints, and ACID guarantees.
# MAGIC 2. **Full DML Support** — `UPDATE`, `DELETE`, and `MERGE INTO` (SCD Type 1 & SCD Type 2 implementation).
# MAGIC 3. **Schema Enforcement & Evolution** — Prevent accidental column corruption (`mergeSchema` vs `overwriteSchema`).
# MAGIC 4. **Time Travel & Auditing** — Query historical table versions (`VERSION AS OF`) and perform `RESTORE TABLE`.
# MAGIC 5. **Storage Optimization & Clustering** — `OPTIMIZE`, `Z-ORDER BY`, Liquid Clustering, and `VACUUM`.
# MAGIC 6. **Cloning & Change Data Feed (CDF)** — Shallow vs Deep Clones, and CDC tracking via `table_changes()`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Setup & Table Initialization

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from delta.tables import DeltaTable
import os

spark = SparkSession.builder.getOrCreate()
spark.sql("USE de_practical_db")

print("✅ Setup complete for Delta Lake Deep Dive.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 1: Delta Transaction Log (`_delta_log`) & ACID Guarantees (Handbook D1–D5)
# MAGIC
# MAGIC ### How Delta Lake Works Under the Hood:
# MAGIC 1. **Storage Layer**: Raw Parquet data files stored on cheap object storage (S3/ADLS/DBFS).
# MAGIC 2. **Transaction Log (`_delta_log/`)**: A directory containing ordered JSON commit files (`00000000000000000000.json`, `00000000000000000001.json`).
# MAGIC 3. **Checkpoint Parquet Files**: Every 10 commits, Delta writes a `.checkpoint.parquet` file consolidating the table state, enabling readers to parse table metadata instantly.
# MAGIC 4. **Optimistic Concurrency Control (OCC)**: Ensures ACID isolation. If two jobs try to write to the table simultaneously, Delta checks if the files modified overlap. If not, both succeed; if yes, the second writer retries automatically.

# COMMAND ----------

# Create a sample Delta table
table_path = "/tmp/de_practical_delta_demo"

data_init = [
    (1, "PROD_A", "Electronics", 150.0),
    (2, "PROD_B", "Apparel", 45.0),
    (3, "PROD_C", "Home", 85.0)
]
df_init = spark.createDataFrame(data_init, ["product_id", "product_name", "category", "price"])

df_init.write.format("delta").mode("overwrite").save(table_path)
spark.sql(f"CREATE TABLE IF NOT EXISTS de_practical_db.delta_demo USING DELTA LOCATION '{table_path}'")

print("=" * 80)
print("  DELTA TRANSACTION LOG DIRECTORY INVOCATION (_delta_log)")
print("=" * 80)
log_dir = os.path.join(table_path, "_delta_log")
for file_name in sorted(os.listdir(log_dir)):
    print(f"📄 Delta Log File: {file_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 2: Delta DML: `UPDATE`, `DELETE`, and `MERGE INTO` (Handbook D6–D10)
# MAGIC
# MAGIC Plain Parquet data lakes cannot update individual rows (files are immutable).
# MAGIC **Delta Lake enables DML by rewriting ONLY the affected Parquet files** and recording the change in the `_delta_log`.

# COMMAND ----------

delta_table = DeltaTable.forPath(spark, table_path)

# ----------------------------------------------------------------------------
# 1. UPDATE Operation (D6)
# ----------------------------------------------------------------------------
print("--- 1. Executing Delta UPDATE (10% price increase on Electronics) ---")
delta_table.update(
    condition = "category = 'Electronics'",
    set = { "price": "price * 1.10" }
)

# ----------------------------------------------------------------------------
# 2. DELETE Operation (D7)
# ----------------------------------------------------------------------------
print("--- 2. Executing Delta DELETE (Remove product_id = 2) ---")
delta_table.delete("product_id = 2")

print("\nTable state after UPDATE & DELETE:")
spark.read.format("delta").load(table_path).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Implementing Upsert (`MERGE INTO`) & SCD Type 2 (Handbook D8, D9)
# MAGIC
# MAGIC - **SCD Type 1 (Overwrite)**: Updates matching records in-place and inserts new records.
# MAGIC - **SCD Type 2 (History Tracking)**: Keeps historical records by setting `is_current = False` and `valid_to = current_timestamp()`, while inserting a new row with `is_current = True`.

# COMMAND ----------

# ============================================================================
# D8: MERGE INTO — SCD Type 1 Implementation
# ============================================================================
incoming_updates = [
    (1, "PROD_A", "Electronics", 175.0), # Updated price
    (4, "PROD_D", "Beauty", 35.0)        # New record
]
df_updates = spark.createDataFrame(incoming_updates, ["product_id", "product_name", "category", "price"])

(
    delta_table.alias("target")
    .merge(
        df_updates.alias("source"),
        "target.product_id = source.product_id"
    )
    .whenMatchedUpdate(set={
        "product_name": "source.product_name",
        "category": "source.category",
        "price": "source.price"
    })
    .whenNotMatchedInsert(values={
        "product_id": "source.product_id",
        "product_name": "source.product_name",
        "category": "source.category",
        "price": "source.price"
    })
    .execute()
)

print("=" * 80)
print("  MERGE INTO (SCD TYPE 1 UPSERT) RESULTS")
print("=" * 80)
spark.read.format("delta").load(table_path).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 3: Schema Enforcement & Schema Evolution (Handbook D11, D12)
# MAGIC
# MAGIC - **Schema Enforcement (Default)**: Prevents writing a DataFrame whose schema does NOT match the target Delta table, preventing data corruption.
# MAGIC - **Schema Evolution (`mergeSchema`)**: Automatically alters the Delta table schema to add new incoming columns when `option("mergeSchema", "true")` is set.

# COMMAND ----------

# 1. Test Schema Enforcement (Expected to Fail)
df_bad_schema = spark.createDataFrame([(5, "PROD_E", "Sports", 99.0, "EXTRA_UNEXPECTED_COL")], 
                                       ["product_id", "product_name", "category", "price", "new_attribute"])

print("Testing Schema Enforcement (Attempting to write extra column without mergeSchema)...")
try:
    df_bad_schema.write.format("delta").mode("append").save(table_path)
except Exception as e:
    print("✅ Schema Enforcement Blocked Write as Expected!")
    print(f"   Error: {str(e)[:120]}...")

# 2. Test Schema Evolution (mergeSchema = True)
print("\n--- Testing Schema Evolution (mergeSchema = True) ---")
(
    df_bad_schema
    .write
    .format("delta")
    .mode("append")
    .option("mergeSchema", "true")
    .save(table_path)
)

print("✅ Delta Table Schema Evolved Successfully:")
spark.read.format("delta").load(table_path).printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 4: Time Travel & History Audit (Handbook D13, D14)
# MAGIC
# MAGIC Every write/update/delete creates a new transaction commit. Delta stores table history metadata, allowing audit queries and instant point-in-time recovery.

# COMMAND ----------

print("=" * 80)
print("  DELTA TABLE TRANSACTION HISTORY (DESCRIBE HISTORY)")
print("=" * 80)
history_df = delta_table.history()
history_df.select("version", "timestamp", "operation", "operationParameters").show(10, truncate=False)

# Query Table Version 0 (Initial Insert State before updates/deletes)
print("--- Time Travel: Querying Version 0 ---")
df_v0 = spark.read.format("delta").option("versionAsOf", 0).load(table_path)
df_v0.show()

# Restore Table to Version 0
print("--- Restoring Table to Version 0 ---")
spark.sql(f"RESTORE TABLE de_practical_db.delta_demo TO VERSION AS OF 0")
print("Restored Table State:")
spark.read.format("delta").load(table_path).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 5: Storage Optimization: `OPTIMIZE`, `Z-ORDER`, Liquid Clustering & `VACUUM` (Handbook D15–D20)
# MAGIC
# MAGIC ### 1. `OPTIMIZE` (Bin-Packing) (D15)
# MAGIC Merges thousands of tiny files (~KB/MBs) into optimal **1 GB Parquet files**. Fixes the "Small File Problem".
# MAGIC
# MAGIC ### 2. `Z-ORDER BY (col)` (D16)
# MAGIC Organizes data along a space-filling Z-curve. Co-locates similar data in the same files to maximize **Data Skipping** during query execution.
# MAGIC
# MAGIC ### 3. Liquid Clustering (D17)
# MAGIC Modern replacement for Hive partitioning and Z-Order (introduced in Databricks Runtime 13.3+). Automatically clusters data dynamically without fixed directory hierarchies or rigid partition keys.
# MAGIC Syntax: `CREATE TABLE ... CLUSTER BY (col1, col2)`.
# MAGIC
# MAGIC ### 4. `VACUUM` (D18)
# MAGIC Garbage-collects unreferenced, tombstoned Parquet data files older than the retention threshold (default 7 days).

# COMMAND ----------

# Execute OPTIMIZE with Z-ORDER BY
spark.sql("OPTIMIZE de_practical_db.delta_demo ZORDER BY (product_id)")
print("✅ OPTIMIZE with Z-ORDER BY completed.")

# Demonstrate VACUUM (Set retention safety check to false for demonstration)
spark.conf.set("spark.databricks.delta.vacuum.parallelDelete.enabled", "true")
spark.sql("SET spark.databricks.delta.vacuum.parallelDelete.enabled = true")

print("Executing VACUUM (Retain 168 hours / 7 days default)...")
# Note: Dry run shows files that would be deleted without removing them
vacuum_dry = delta_table.vacuum(168)
print("✅ VACUUM dry run completed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 6: Delta Clones & Change Data Feed (CDF) (Handbook D21–D27)
# MAGIC
# MAGIC ### 6.1 Shallow Clone vs Deep Clone (D21, D22)
# MAGIC - **Shallow Clone (`SHALLOW CLONE`)**: Copies ONLY transaction log metadata. Shares existing Parquet data files. Instant creation! Ideal for staging testing environments.
# MAGIC - **Deep Clone (`DEEP CLONE`)**: Copies metadata AND makes a physical copy of all underlying Parquet data files. Independent production backup.

# COMMAND ----------

# Create Shallow & Deep Clones via SQL
spark.sql("CREATE TABLE IF NOT EXISTS de_practical_db.delta_shallow_clone SHALLOW CLONE de_practical_db.delta_demo")
spark.sql("CREATE TABLE IF NOT EXISTS de_practical_db.delta_deep_clone DEEP CLONE de_practical_db.delta_demo")

print("✅ Shallow and Deep Clones Created Successfully:")
print("Shallow Clone Count:", spark.table("de_practical_db.delta_shallow_clone").count())
print("Deep Clone Count:   ", spark.table("de_practical_db.delta_deep_clone").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2 Delta Change Data Feed (CDF / CDC) (Handbook D25–D27)
# MAGIC
# MAGIC Delta CDF logs row-level change events (`insert`, `update_preimage`, `update_postimage`, `delete`) to allow downstream micro-batch pipelines to process incremental changes efficiently.

# COMMAND ----------

# Enable CDF on a new table
spark.sql("""
CREATE TABLE IF NOT EXISTS de_practical_db.cdf_demo (
    id INT,
    val STRING
) USING DELTA
TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# Perform DML Operations
spark.sql("INSERT INTO de_practical_db.cdf_demo VALUES (1, 'Initial_A'), (2, 'Initial_B')")
spark.sql("UPDATE de_practical_db.cdf_demo SET val = 'Updated_A' WHERE id = 1")
spark.sql("DELETE FROM de_practical_db.cdf_demo WHERE id = 2")

print("=" * 80)
print("  DELTA CHANGE DATA FEED (CDF) LOG OUTPUT (table_changes)")
print("=" * 80)
df_cdf = spark.read.format("delta").option("readChangeFeed", "true").option("startingVersion", 0).table("de_practical_db.cdf_demo")

df_cdf.select("id", "val", "_change_type", "_commit_version", "_commit_timestamp").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 💡 Senior Data Engineer Interview Practice (Handbook H3)
# MAGIC
# MAGIC ### Q1: Explain the internal structure of the `_delta_log` directory. How does Delta Lake maintain ACID transactions on object storage?
# MAGIC **Answer**: Delta Lake writes ordered JSON commit logs (`000000.json`, `000001.json`) to the `_delta_log` directory inside the table location. Each commit file records discrete transaction operations (e.g., `add` file, `remove` file, update `metaData`).
# MAGIC To maintain ACID guarantees:
# MAGIC - **Atomicity**: Writes create a single JSON commit file atomically.
# MAGIC - **Consistency**: Readers construct the valid table snapshot by reading JSON commit logs sequentially.
# MAGIC - **Isolation**: Uses **Optimistic Concurrency Control (OCC)**. If two writers attempt to commit at version $V$, the second writer checks if files modified overlap; if not, it automatically retries as version $V+1$.
# MAGIC - **Performance**: Every 10 commits, Delta generates a `.checkpoint.parquet` file that aggregates all previous commits, avoiding reading thousands of historical JSON files.
# MAGIC
# MAGIC ### Q2: Compare `OPTIMIZE Z-ORDER BY` with `Liquid Clustering`.
# MAGIC **Answer**:
# MAGIC - `OPTIMIZE Z-ORDER BY`: Multi-dimensional clustering along a Z-curve. Highly effective for static columns, but requires manual re-execution, is computationally expensive on large tables, and cannot dynamically adjust clustering keys as query patterns change.
# MAGIC - `Liquid Clustering`: Modern dynamic clustering mechanism (Databricks Runtime 13.3+). It decouples physical layout from directory partitioning, supports **incremental clustering**, allows changing clustering keys without rewriting the table, and optimizes writes in real-time.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # ✅ Module 05 Summary
# MAGIC
# MAGIC | Feature | Implementation Syntax | Handbook Reference |
# MAGIC |---|---|---|
# MAGIC | Transaction Log | `_delta_log/*.json` + `*.checkpoint.parquet` | D1–D5 |
# MAGIC | DML / Upsert | `MERGE INTO target USING source ON key WHEN MATCHED...` | D6–D10 |
# MAGIC | Schema Evolution | `.option("mergeSchema", "true")` | D11, D12 |
# MAGIC | Time Travel | `.option("versionAsOf", N)` / `RESTORE TABLE` | D13, D14 |
# MAGIC | File Optimization | `OPTIMIZE table ZORDER BY (col)` / Liquid Clustering | D15–D17 |
# MAGIC | Garbage Collection | `VACUUM table RETAIN 168 HOURS` | D18 |
# MAGIC | Clones | `SHALLOW CLONE` (Metadata only) vs `DEEP CLONE` (Data copy) | D21, D22 |
# MAGIC | CDC / CDF | `TBLPROPERTIES (delta.enableChangeDataFeed = true)` + `table_changes()` | D25–D27 |
