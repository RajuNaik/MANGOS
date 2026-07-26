# Databricks notebook source

# MAGIC %md
# MAGIC # ⚡ Notebook 04 — Performance Tuning, Data Skew Mitigation & Caching
# MAGIC
# MAGIC **Handbook Sections Covered**: Part C1–C5, C7, C8, C11, C13, C15–C18, K6–K10
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Learning Objectives
# MAGIC 1. **Partitioning Mechanics** — `repartition()` vs `coalesce()`, shuffle partition tuning.
# MAGIC 2. **Disk Storage Partitioning vs Bucketing** — `partitionBy()` vs `bucketBy()` when writing tables.
# MAGIC 3. **Caching & Lineage Truncation** — `cache()` vs `persist()` vs `checkpoint()`, Delta Cache vs Spark Cache.
# MAGIC 4. **Data Skew Detection & Salting Technique** — Understand skewed keys, worker stragglers, and implement Salting.
# MAGIC 5. **AQE Skew Join Optimization** — Automatic skew detection and partition splitting configurations.
# MAGIC 6. **Memory Tuning & OOM Avoidance** — Driver OOM vs Executor OOM, `collect()` anti-patterns.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Setup & Environment Configuration

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.storage import StorageLevel
import os
import time

spark = SparkSession.builder.getOrCreate()
spark.sql("USE de_practical_db")

# Load baseline tables
df_customers = spark.table("de_practical_db.raw_customers")
df_orders = spark.table("de_practical_db.raw_orders")

print("✅ Setup complete for Performance Tuning & Skew Mitigation.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 1: Partitioning Mechanics: `repartition` vs `coalesce` (Handbook C1, C2, C3)
# MAGIC
# MAGIC | Feature | `repartition(N)` | `coalesce(N)` |
# MAGIC |---|---|---|
# MAGIC | **Transformation Type** | **Wide Transformation** (Triggers full network shuffle) | **Narrow Transformation** (Combines adjacent partitions without shuffle) |
# MAGIC | **Use Case** | **Increase** partitions or re-balance data evenly across cluster | **Decrease** partitions (e.g., reduce 1,000 small files to 10 files before writing) |
# MAGIC | **Partition Count** | Can increase OR decrease partition count | Can ONLY decrease partition count |

# COMMAND ----------

print("=" * 80)
print("  REPARTITION VS COALESCE DEMONSTRATION")
print("=" * 80)

print(f"Original Orders Partition Count: {df_orders.rdd.getNumPartitions()}")

# 1. Repartition to INCREASE partitions (Wide Transformation)
df_repartitioned = df_orders.repartition(16)
print(f"After repartition(16):          {df_repartitioned.rdd.getNumPartitions()} partitions (Shuffle triggered)")

# 2. Coalesce to DECREASE partitions (Narrow Transformation)
df_coalesced = df_repartitioned.coalesce(2)
print(f"After coalesce(2):             {df_coalesced.rdd.getNumPartitions()} partitions (Zero shuffle)")

# Demonstrate shuffle partition tuning
print("\n--- Tuning spark.sql.shuffle.partitions ---")
print(f"Current shuffle partitions: {spark.conf.get('spark.sql.shuffle.partitions')}")
# Recommended rule of thumb: 200 for medium data, 2000+ for TB-scale data, 4-8 for tiny local/test data
spark.conf.set("spark.sql.shuffle.partitions", "8")
print(f"Updated shuffle partitions for test run: {spark.conf.get('spark.sql.shuffle.partitions')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 2: Storage Partitioning (`partitionBy`) vs Bucketing (`bucketBy`) (Handbook C4)
# MAGIC
# MAGIC When writing data to disk (S3/ADLS/DBFS):
# MAGIC - **`partitionBy("col")`**: Creates a physical directory structure (`/loyalty_tier=Gold/`). Best for **low-cardinality** columns (e.g., `date`, `region`, `status`). 
# MAGIC   - ⚠️ **Cardinality Trap**: Partitioning by a high-cardinality column like `customer_id` creates 1,000,000 tiny directories ("Small File Problem"), destroying metadata read performance.
# MAGIC - **`bucketBy(N, "col")`**: Hashes data into a fixed number of bucket files ($N$) inside a single directory. Pre-sorts and pre-shuffles data on disk by join key, **eliminating shuffle during future joins!**

# COMMAND ----------

output_path_partitioned = "/tmp/de_practical_partitioned"

# Save using partitionBy on low-cardinality column
(
    df_customers
    .write
    .mode("overwrite")
    .partitionBy("loyalty_tier")
    .format("parquet")
    .save(output_path_partitioned)
)

print("=" * 80)
print("  PHYSICAL DISK DIRECTORY STRUCTURE (partitionBy)")
print("=" * 80)
for root, dirs, files in os.walk(output_path_partitioned):
    for dir_name in dirs:
        print(f"📁 Partition Directory: {os.path.join(root, dir_name)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 3: Cache vs Persist vs Checkpoint & Delta Cache (Handbook C5, C11)
# MAGIC
# MAGIC ### 1. Spark Cache & Persist (C5)
# MAGIC - **`df.cache()`**: Shortcut for `persist(StorageLevel.MEMORY_AND_DISK)` (deserialized in memory, spills to disk if full).
# MAGIC - **`df.persist(StorageLevel.MEMORY_ONLY_SER)`**: Serialized in memory (smaller RAM footprint, slight CPU overhead to deserialize).
# MAGIC - **Lazy Evaluation**: `cache()` and `persist()` do NOT store data immediately. Data is cached **only when the first action is executed!**
# MAGIC
# MAGIC ### 2. Checkpoint (`df.checkpoint()`) (C5)
# MAGIC - Cuts the RDD/DataFrame DAG lineage history and writes the raw DataFrame to reliable storage (DBFS/S3).
# MAGIC - Essential for recursive algorithms or massive 50+ stage pipelines to avoid `StackOverflowError`.
# MAGIC
# MAGIC ### 3. Delta Cache (Disk Cache) (C11)
# MAGIC - Automatic on Databricks clusters. Stores local decompressed copies of remote Delta files on worker local NVMe SSD drives.

# COMMAND ----------

print("=" * 80)
print("  CACHE vs PERSIST vs CHECKPOINT DEMONSTRATION")
print("=" * 80)

# Configure checkpoint directory
spark.sparkContext.setCheckpointDir("/tmp/de_practical_checkpoints")

# 1. Cache
df_cached = df_customers.filter(F.col("lifetime_spend") > 50.0).cache()
print(f"Is df_cached in memory before action? {df_cached.is_cached}")
df_cached.count() # Action triggers caching into RAM
print("✅ Action executed — Data now cached in Spark Memory.")

# 2. Persist with specific Storage Level
df_persisted = df_orders.persist(StorageLevel.MEMORY_AND_DISK_SER)
df_persisted.count()
print("✅ Persisted with MEMORY_AND_DISK_SER.")

# 3. Checkpoint (Truncate Lineage)
df_checkpointed = df_orders.checkpoint()
print("✅ Checkpoint written — Lineage DAG cut successfully.")

# Clean up cache
df_cached.unpersist()
df_persisted.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 4: Data Skew Identification & Salting Technique (Handbook C8)
# MAGIC
# MAGIC ### What is Data Skew?
# MAGIC In a distributed join or aggregation, **data skew** occurs when one or a few join keys contain 80%-90% of all rows (e.g., `NULL` values, default values, or a massive client like `CUST_000001`).
# MAGIC
# MAGIC **Symptom**: 99 worker tasks finish in 5 seconds, but 1 worker task gets stuck at 99% for 2 hours (the "Straggler Task").
# MAGIC
# MAGIC ### Salting Technique (C8):
# MAGIC 1. Append a random integer `[0, N-1]` (the "salt") to the skewed join key on the heavy DataFrame.
# MAGIC 2. Explode the lookup/dimension DataFrame by duplicating each key `N` times with salts `[0, N-1]`.
# MAGIC 3. Join on `(join_key, salt_key)`. The skewed rows are evenly distributed across `N` tasks, eliminating the bottleneck!

# COMMAND ----------

# ============================================================================
# C8: Generate Synthetic Skewed Dataset
# ============================================================================
np.random.seed(2026)

# Create 100,000 order records where 85% of records have customer_id = 'CUST_SKEWED'
skewed_orders = []
for i in range(1, 50001):
    cid = "CUST_SKEWED" if np.random.random() < 0.85 else f"CUST_{np.random.randint(1, 500):06d}"
    skewed_orders.append((f"ORD_SKEW_{i:06d}", cid, float(np.random.uniform(10.0, 500.0))))

df_skewed_orders = spark.createDataFrame(skewed_orders, ["order_id", "customer_id", "amount"])

print("=" * 80)
print("  DATA SKEW DEMONSTRATION: Key Distribution")
print("=" * 80)
df_skewed_orders.groupBy("customer_id").count().orderBy(F.col("count").desc()).show(5)

# COMMAND ----------

# ============================================================================
# C8: Implement Salting Technique to Fix Skewed Join
# ============================================================================
SALT_FACTOR = 4 # Split skewed key into 4 sub-keys

# Step 1: Add salt to skewed left table
df_salted_orders = df_skewed_orders.withColumn(
    "salt", 
    F.floor(F.rand() * SALT_FACTOR)
).withColumn(
    "salted_customer_id", 
    F.concat_ws("_", "customer_id", "salt")
)

# Step 2: Replicate customer dimension table SALT_FACTOR times
salt_array = F.array([F.lit(i) for i in range(SALT_FACTOR)])
df_salted_customers = (
    df_customers
    .withColumn("salt_array", salt_array)
    .withColumn("salt", F.explode("salt_array"))
    .withColumn("salted_customer_id", F.concat_ws("_", "customer_id", "salt"))
    .drop("salt_array", "salt")
)

# Step 3: Perform Join on salted key (salted_customer_id)
df_salted_join = df_salted_orders.join(
    df_salted_customers,
    "salted_customer_id",
    "inner"
)

print("=" * 80)
print("  SALTED JOIN EXECUTED SUCCESSFULLY")
print("=" * 80)
print(f"Total Salted Joined Rows: {df_salted_join.count()}")
print("\nSalted Partition Key Distribution (Notice how CUST_SKEWED is split evenly across 4 keys!):")
df_salted_orders.groupBy("salted_customer_id").count().filter(F.col("salted_customer_id").contains("CUST_SKEWED")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 5: Automatic Skew Mitigation via AQE (Handbook C8, C14)
# MAGIC
# MAGIC On Databricks, **AQE Skew Join** automatically detects skewed partitions during a Sort-Merge Join and splits them into smaller sub-partitions without requiring manual salting code!
# MAGIC
# MAGIC ```python
# MAGIC # AQE Skew Join Configurations (Handbook C14):
# MAGIC spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
# MAGIC spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5")
# MAGIC spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256MB")
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 6: Memory Tuning & Out-of-Memory (OOM) Debugging (Handbook C16, C17)
# MAGIC
# MAGIC ### 1. Driver OOM (`java.lang.OutOfMemoryError: Java heap space`)
# MAGIC - **Root Cause**: Calling `df.collect()` or `df.toPandas()` on a multi-gigabyte DataFrame. `collect()` pulls ALL distributed partitions onto the single Driver node JVM memory.
# MAGIC - **Fix**: Never use `collect()` on large DataFrames. Use `df.take(N)`, `df.limit(N)`, or write directly to storage.
# MAGIC
# MAGIC ### 2. Executor OOM (`Container killed by YARN for exceeding memory limits`)
# MAGIC - **Root Cause**: Data skew (one partition exceeds executor RAM), massive `groupBy` on high-cardinality keys, or un-broadcasted Cartesian cross joins.
# MAGIC - **Fix**: Increase executor memory, fix data skew via salting, increase `spark.sql.shuffle.partitions`, or replace row-by-row Python UDFs.

# COMMAND ----------

# Demonstrating SAFE iteration vs DANGEROUS collect()
print("=" * 80)
print("  SAFE DATA ITERATION PATTERNS")
print("=" * 80)

# ✅ SAFE: Fetch only top 5 rows to driver
sample_rows = df_customers.limit(5).collect()
print(f"Fetched {len(sample_rows)} rows safely using limit(5).collect()")

# ✅ SAFE: Iterate lazily over large DataFrame without loading all in RAM
for row in df_customers.select("customer_id", "email").toLocalIterator():
    # Process row by row
    pass
print("✅ Successfully iterated using toLocalIterator() without driver memory strain.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 💡 Senior Data Engineer Interview Practice (Handbook H3)
# MAGIC
# MAGIC ### Q1: Explain how the Salting Technique resolves severe data skew during a distributed join.
# MAGIC **Answer**: Data skew happens when a single join key (e.g., `NULL` or a super-user ID) contains millions of rows. During a shuffle join, Spark hashes keys to assign rows to worker partitions; thus, all rows for that skewed key land on a **single executor core**, while other cores sit idle.
# MAGIC **Salting Solution**:
# MAGIC 1. On the skewed large table, add a random integer column `salt` ranging from `0` to `N-1`, creating a new composite key `concat(join_key, "_", salt)`. This evenly spreads the skewed key's rows across `N` separate partitions and cores.
# MAGIC 2. On the lookup dimension table, explode each row `N` times with salts `0` to `N-1`.
# MAGIC 3. Join on the composite key `(join_key, salt)`. All partitions execute in parallel in $1/N$-th of the original time!
# MAGIC
# MAGIC ### Q2: Compare `repartition(N)` and `coalesce(N)`. When must you use `repartition` over `coalesce`?
# MAGIC **Answer**: `repartition(N)` performs a full **wide-dependency network shuffle** to distribute data uniformly across $N$ partitions. `coalesce(N)` is a **narrow transformation** that merges existing adjacent partitions on the same worker node without network shuffling.
# MAGIC - Use `coalesce(N)` when **decreasing** the number of partitions (e.g., consolidating output files before writing).
# MAGIC - You MUST use `repartition(N)` when **increasing** the number of partitions or when you need to fix severe partition size imbalance (data skew across partitions).

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # ✅ Module 04 Summary
# MAGIC
# MAGIC | Topic | Recommended Strategy | Handbook Reference |
# MAGIC |---|---|---|
# MAGIC | Partitioning | Use `coalesce` to shrink, `repartition` to balance | C1, C2 |
# MAGIC | Storage Layout | `partitionBy` for low cardinality; `bucketBy` for frequent join keys | C4 |
# MAGIC | Caching Strategy | `cache()` / `persist()` for re-used DFs; `checkpoint()` to cut deep DAGs | C5, C11 |
# MAGIC | Skew Mitigation | Salting keys manually OR configuring AQE Skew Join | C8, C14 |
# MAGIC | OOM Avoidance | Avoid `collect()`, use `toLocalIterator()` / `take()`, balance shuffle partitions | C16, C17 |
