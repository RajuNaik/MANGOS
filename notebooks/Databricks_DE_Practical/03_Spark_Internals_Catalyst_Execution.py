# Databricks notebook source

# MAGIC %md
# MAGIC # 🧠 Notebook 03 — Spark Internals, Catalyst Optimizer & Execution Plans
# MAGIC
# MAGIC **Handbook Sections Covered**: Part A5–A8, C6, C9, C10, C12, C14, J1–J6, K1–K5
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Learning Objectives
# MAGIC 1. **Catalyst Optimizer Deep Dive** — Understand Parsed, Analyzed, Optimized, Physical Plans, and Whole-Stage Java Codegen.
# MAGIC 2. **Adaptive Query Execution (AQE)** — Dynamic partition coalescing, dynamic join switching, and dynamic skew join handling.
# MAGIC 3. **Execution Plan Decoupling** — Interpret `explain(extended=True)`, `explain(mode="cost")`, `ShuffleExchange`, and `BroadcastExchange`.
# MAGIC 4. **Narrow vs Wide Dependencies** — Understand shuffle boundaries, stages, and tasks.
# MAGIC 5. **Join Engine Mechanics** — Compare Sort-Merge Join (SMJ) vs Broadcast Hash Join (BHJ).
# MAGIC 6. **RDD vs DataFrame vs Dataset** — Memory representation, Tungsten binary format, and JVM Encoders.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Setup & Data Preparation

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

spark = SparkSession.builder.getOrCreate()
spark.sql("USE de_practical_db")

# Load baseline tables
df_customers = spark.table("de_practical_db.raw_customers")
df_orders = spark.table("de_practical_db.raw_orders")
df_products = spark.table("de_practical_db.raw_products")

print("✅ Setup initialized for Spark Internals & Catalyst Analysis.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 1: The 5 Phases of Catalyst Optimizer (Handbook C6)
# MAGIC
# MAGIC When you write PySpark or Spark SQL code, Spark does NOT execute it immediately. It passes through **5 distinct Catalyst phases**:
# MAGIC
# MAGIC ```text
# MAGIC PySpark / SQL Code
# MAGIC        │
# MAGIC        ▼
# MAGIC 1. Parsed Logical Plan (Unresolved AST - checks syntax only)
# MAGIC        │
# MAGIC        ▼  <-- Catalog / Metastore (Resolves table & column names, types)
# MAGIC 2. Analyzed Logical Plan (Resolved Logical Plan)
# MAGIC        │
# MAGIC        ▼  <-- Rule-Based Optimizer (Filter Pushdown, Projection Pruning, Constant Folding)
# MAGIC 3. Optimized Logical Plan
# MAGIC        │
# MAGIC        ▼  <-- Cost-Based Optimizer (CBO - estimates data size/cardinality)
# MAGIC 4. Physical Plans (Generates multiple physical execution strategies)
# MAGIC        │
# MAGIC        ▼  <-- Selects Best Physical Plan based on cost
# MAGIC 5. Code Generation (Tungsten Whole-Stage Java Bytecode Generation)
# MAGIC ```

# COMMAND ----------

# Build a multi-step query to inspect Catalyst phases
df_complex_query = (
    df_orders.filter(F.col("status") == "COMPLETED")
    .join(df_customers, "customer_id", "inner")
    .filter(F.col("loyalty_tier") == "Gold")
    .groupBy("customer_id", "first_name", "last_name")
    .agg(
        F.count("order_id").alias("total_completed_orders"),
        F.sum("total_amount").alias("gold_spend")
    )
    .filter(F.col("gold_spend") > 500.0)
)

print("=" * 80)
print("  CATALYST OPTIMIZER EXPLAIN (EXTENDED = TRUE)")
print("=" * 80)
# Print Parsed, Analyzed, Optimized, and Physical Plans
df_complex_query.explain(extended=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Key Catalyst Optimizations Demonstrated above:
# MAGIC 1. **Filter Pushdown (`PredicatePushdown`)**: The filter `status == 'COMPLETED'` and `loyalty_tier == 'Gold'` are pushed directly down to the file scan layer (`FileScan delta`), avoiding reading unnecessary rows from storage!
# MAGIC 2. **Column Projection Pruning (`ColumnPruning`)**: Unused columns (e.g., `phone`, `email`, `registration_date`) are dropped before joining, reducing RAM and network shuffle load.
# MAGIC 3. **Whole-Stage Java Codegen (`*(1) HashAggregate`, `*(2) BroadcastHashJoin`)**: Asterisks `*(N)` indicate that multiple operations are compiled into a single clean C++/Java loops in memory via Project Tungsten.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 2: Adaptive Query Execution (AQE) (Handbook C6, C14)
# MAGIC
# MAGIC **AQE (enabled by default in Spark 3.0+)** re-optimizes the query plan at **runtime** using runtime statistics gathered during stage execution.
# MAGIC
# MAGIC ### 3 Major AQE Features:
# MAGIC 1. **Dynamic Coalescing of Shuffle Partitions**: Automatically combines thousands of tiny empty shuffle partitions into optimal ~64MB - 128MB partitions, eliminating the "small file / empty task" problem.
# MAGIC 2. **Dynamic Switching of Join Strategies**: If a join side turns out to be smaller than the broadcast threshold *after filtering at runtime*, AQE dynamically converts a Sort-Merge Join to a **Broadcast Hash Join**.
# MAGIC 3. **Dynamic Skew Join Handling**: Automatically detects skewed partitions at runtime and splits them into sub-partitions to prevent worker stragglers.

# COMMAND ----------

# Inspect active AQE settings
print("=" * 80)
print("  ADAPTIVE QUERY EXECUTION (AQE) CONFIGURATIONS")
print("=" * 80)
print(f"AQE Enabled:                       {spark.conf.get('spark.sql.adaptive.enabled')}")
print(f"AQE Coalesce Partitions Enabled:   {spark.conf.get('spark.sql.adaptive.coalescePartitions.enabled')}")
print(f"AQE Dynamic Join Switching:        {spark.conf.get('spark.sql.adaptive.autoBroadcastJoinThreshold')}")
print(f"AQE Skew Join Enabled:             {spark.conf.get('spark.sql.adaptive.skewJoin.enabled')}")

# Demonstrate Cost-Based Optimization (CBO) & AQE mode
print("\n--- Cost-Based Plan Analysis ---")
df_complex_query.explain(mode="cost")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 3: Narrow vs Wide Dependencies & Stage Boundaries (Handbook A7, C3)
# MAGIC
# MAGIC - **Narrow Dependency**: Each partition of the parent RDD/DataFrame is used by at most ONE partition of the child DataFrame (e.g., `map()`, `filter()`, `select()`). Executed in **parallel within the SAME stage** without network IO.
# MAGIC - **Wide Dependency**: Multiple child partitions depend on data from parent partitions (e.g., `groupBy()`, `join()`, `distinct()`, `repartition()`). Creates a **Shuffle Boundary** and forces the driver to create a **NEW Stage**.

# COMMAND ----------

# ----------------------------------------------------------------------------
# Narrow vs Wide Dependency Demonstration
# ----------------------------------------------------------------------------
df_narrow = df_customers.filter(F.col("lifetime_spend") > 100.0).select("customer_id", "email") # Narrow

df_wide = df_customers.groupBy("loyalty_tier").agg(F.avg("lifetime_spend")) # Wide (Shuffle required)

print("=" * 80)
print("  NARROW DEPENDENCY PHYSICAL PLAN (No Shuffle Exchange)")
print("=" * 80)
df_narrow.explain()

print("\n" + "=" * 80)
print("  WIDE DEPENDENCY PHYSICAL PLAN (Notice ShuffleExchange Operator!)")
print("=" * 80)
df_wide.explain()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 4: Join Mechanics: Sort-Merge Join (SMJ) vs Broadcast Hash Join (BHJ) (Handbook C7, C9)
# MAGIC
# MAGIC ### 1. Broadcast Hash Join (BHJ) (Handbook C7)
# MAGIC - **Mechanism**: Driver collects the small table, builds a Hash Table in memory, and broadcasts it to ALL worker nodes. Workers join locally **without any shuffle of the large table!**
# MAGIC - **Performance**: Fastest join strategy ($O(N)$ time complexity, zero network shuffle for large table).
# MAGIC - **Threshold**: Controlled by `spark.sql.autoBroadcastJoinThreshold` (default 10 MB). Can be explicitly requested via `F.broadcast()`.
# MAGIC
# MAGIC ### 2. Sort-Merge Join (SMJ) (Handbook C9)
# MAGIC - **Mechanism**: 
# MAGIC   1. **Shuffle Phase**: Both tables are re-partitioned across cluster nodes based on the join key hash.
# MAGIC   2. **Sort Phase**: Data inside each partition on worker nodes is sorted by join key.
# MAGIC   3. **Merge Phase**: Workers iterate through both sorted streams in parallel to merge matching keys.
# MAGIC - **Performance**: Scalable for petabyte-scale tables, but heavy network I/O and disk spilling if memory is constrained.

# COMMAND ----------

# Force Broadcast Hash Join (BHJ) using F.broadcast() hint
df_bhj = df_orders.join(F.broadcast(df_products), df_orders.order_id == df_products.product_id, "inner")

# Force Sort-Merge Join (SMJ) by disabling broadcast threshold
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
df_smj = df_orders.join(df_customers, "customer_id", "inner")

print("=" * 80)
print("  PHYSICAL PLAN: BROADCAST HASH JOIN (BHJ)")
print("=" * 80)
df_bhj.explain()

print("\n" + "=" * 80)
print("  PHYSICAL PLAN: SORT-MERGE JOIN (SMJ)")
print("=" * 80)
df_smj.explain()

# Restore default broadcast threshold (10MB)
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 10 * 1024 * 1024)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 5: RDD vs DataFrame vs Dataset & Tungsten Binary Format (Handbook A6)
# MAGIC
# MAGIC | Feature | RDD (Resilient Distributed Dataset) | DataFrame | Dataset (Scala/Java) |
# MAGIC |---|---|---|---|
# MAGIC | Data Type | Unstructured JVM Objects (`RDD[T]`) | Generic `Row` objects backed by Catalyst | Strongly-typed domain objects (`Dataset[T]`) |
# MAGIC | Optimization | None (Developer writes execution logic) | Full Catalyst & Tungsten Optimization | Catalyst & Tungsten Optimization |
# MAGIC | Serialization | Heavy Java Serialization (Kryo/Java) | **Tungsten Off-Heap Binary Format** | Encoders (Fast byte code) |
# MAGIC | GC Impact | **High** (Millions of JVM objects trigger GC pauses) | **Zero GC** (Operates directly on raw byte arrays) | Moderate |

# COMMAND ----------

# Convert DataFrame to RDD and compare execution
rdd_cust = df_customers.rdd

print("=" * 80)
print("  RDD vs DATAFRAME COMPARISON")
print("=" * 80)
print(f"DataFrame Count: {df_customers.count()}")
print(f"RDD First Record: {rdd_cust.first()}")

# Demonstrate RDD Map-Reduce Transformation
rdd_spend_sum = rdd_cust.map(lambda r: (r.loyalty_tier, r.lifetime_spend)).reduceByKey(lambda a, b: a + b)
print("RDD ReduceByKey Output (Loyalty Tier Spend):", rdd_spend_sum.collect())

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 💡 Senior Data Engineer Interview Practice (Handbook H3)
# MAGIC
# MAGIC ### Q1: Explain how Adaptive Query Execution (AQE) dynamically optimizes join strategies at runtime.
# MAGIC **Answer**: In static query planning, if a table's initial file size on disk is 50 MB, Spark chooses a Sort-Merge Join (SMJ) because 50 MB > 10 MB broadcast threshold. However, if a SQL filter (e.g., `WHERE date = '2026-07-26'`) reduces the actual data read to only 2 MB, static planning cannot change the plan.
# MAGIC With **AQE**, after the initial scan and filter stage completes, AQE checks the actual runtime size of the intermediate dataset (2 MB). Since 2 MB < 10 MB, AQE dynamically switches the physical plan from a Sort-Merge Join to a **Broadcast Hash Join**, skipping the expensive shuffle and sort phases entirely.
# MAGIC
# MAGIC ### Q2: What is Whole-Stage Java Code Generation (Tungsten Codegen) and how do you spot it in an execution plan?
# MAGIC **Answer**: Whole-Stage Codegen collapses multiple physical plan operators (like Filter, Project, and HashAggregate) into a single, clean Java loop byte-code representation. This eliminates virtual function calls and keeps intermediate data inside CPU L1/L2 registers rather than pushing data up/down JVM object stacks.
# MAGIC In `df.explain()`, Whole-Stage Codegen is identified by an asterisk next to the operator name, such as `*(1) Filter`, `*(1) HashAggregate`, or `*(2) BroadcastHashJoin`.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # ✅ Module 03 Summary
# MAGIC
# MAGIC | Concept | Mechanism & Details | Handbook Reference |
# MAGIC |---|---|---|
# MAGIC | Catalyst Optimizer | 5-phase pipeline: Parsed -> Analyzed -> Optimized -> Physical -> Codegen | C6, J1 |
# MAGIC | AQE | Runtime dynamic coalescing, join switching, skew handling | C6, C14 |
# MAGIC | Dependency Types | Narrow (In-stage, no shuffle) vs Wide (`ShuffleExchange`, Stage boundary) | A7, C3 |
# MAGIC | Join Mechanics | BHJ (No shuffle, memory broadcast) vs SMJ (Shuffle + Sort + Merge) | C7, C9 |
# MAGIC | Memory & Codegen | Tungsten off-heap byte format, Whole-Stage Java Codegen (`*(N)`) | A6, A8, J2 |
