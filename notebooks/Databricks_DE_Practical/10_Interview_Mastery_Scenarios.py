# Databricks notebook source

# MAGIC %md
# MAGIC # 🎓 Notebook 10 — Enterprise Interview Mastery & Advanced Coding Scenarios
# MAGIC
# MAGIC **Handbook Sections Covered**: Part J1–J25, H1–H5, K1–K20
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Learning Objectives
# MAGIC 1. **25+ Senior/Principal DE Scenario Challenges** — Solve complex interview problems using PySpark & Spark SQL.
# MAGIC 2. **Gaps & Islands Problem (Sessionization)** — Identify user session boundaries from clickstream logs.
# MAGIC 3. **High-Performance Pivot / Unpivot** — Perform cross-tabulation without PySpark `pivot()` performance bottlenecks.
# MAGIC 4. **Complex Analytical Windowing** — Running cumulative totals, consecutive activity streak detection, and `dense_rank()`.
# MAGIC 5. **Production Incident Debugging** — Diagnose OOM errors, GC pauses, executor loss, and task stragglers.
# MAGIC 6. **Lakehouse Architecture System Design** — Architect a petabyte-scale Databricks platform for FAANG / Fortune 500 interviews.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Setup & Data Preparation

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

spark = SparkSession.builder.getOrCreate()
spark.sql("USE de_practical_db")

print("✅ Setup complete for Enterprise Interview Mastery Scenarios.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Challenge 1: The Gaps & Islands Problem (Sessionization) (Handbook J8)
# MAGIC
# MAGIC ### Scenario:
# MAGIC You have raw user clickstream page views. A new session starts if a user is inactive for **> 15 minutes** between consecutive clicks. Calculate the `session_id`, `session_start_time`, `session_end_time`, and total clicks per session.

# COMMAND ----------

# Generate Clickstream Event Data
click_data = [
    ("USER_A", "2026-07-26 10:00:00"),
    ("USER_A", "2026-07-26 10:05:00"), # +5 mins (Same session)
    ("USER_A", "2026-07-26 10:10:00"), # +5 mins (Same session)
    ("USER_A", "2026-07-26 10:35:00"), # +25 mins (> 15 min gap! NEW SESSION)
    ("USER_A", "2026-07-26 10:40:00"), # +5 mins (Same session 2)
    ("USER_B", "2026-07-26 11:00:00"),
    ("USER_B", "2026-07-26 11:02:00")
]
df_clicks = spark.createDataFrame(click_data, ["user_id", "click_time"]).withColumn("click_time", F.to_timestamp("click_time"))

# Step 1: Calculate time difference from previous click (lag)
w_user = Window.partitionBy("user_id").orderBy("click_time")

df_session_step1 = (
    df_clicks
    .withColumn("prev_click_time", F.lag("click_time", 1).over(w_user))
    .withColumn("time_diff_sec", F.coalesce(F.col("click_time").cast("long") - F.col("prev_click_time").cast("long"), F.lit(0)))
)

# Step 2: Flag 1 if new session (> 15 mins = 900 seconds), else 0
df_session_step2 = df_session_step1.withColumn(
    "is_new_session", 
    F.when(F.col("time_diff_sec") > 900, 1).otherwise(0)
)

# Step 3: Cumulative Sum of flags creates Session ID (The Island identifier!)
df_session_step3 = df_session_step2.withColumn(
    "session_num", 
    F.sum("is_new_session").over(w_user)
).withColumn("session_id", F.concat_ws("_", "user_id", "session_num"))

# Step 4: Group By session_id to extract metrics
df_final_sessions = (
    df_session_step3
    .groupBy("user_id", "session_id")
    .agg(
        F.min("click_time").alias("session_start"),
        F.max("click_time").alias("session_end"),
        F.count("click_time").alias("click_count"),
        ((F.max("click_time").cast("long") - F.min("click_time").cast("long")) / 60).alias("session_duration_mins")
    )
    .orderBy("user_id", "session_start")
)

print("=" * 80)
print("  GAPS & ISLANDS SOLUTION: USER SESSIONIZATION")
print("=" * 80)
df_final_sessions.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Challenge 2: High-Performance Pivot without `pivot()` Bottlenecks (Handbook J12)
# MAGIC
# MAGIC ### Performance Note:
# MAGIC PySpark's built-in `.pivot()` requires an extra **two-pass shuffle aggregation** to determine distinct pivot values if not supplied explicitly.
# MAGIC **High-Performance Alternative**: Use a single `groupBy` with conditional aggregations (`max(when(...))`).

# COMMAND ----------

df_sales_raw = spark.createDataFrame([
    ("2026-01", "Electronics", 15000.0),
    ("2026-01", "Apparel", 8000.0),
    ("2026-02", "Electronics", 18000.0),
    ("2026-02", "Apparel", 9500.0)
], ["month", "category", "revenue"])

print("=" * 80)
print("  HIGH-PERFORMANCE CONDITIONAL PIVOT (SINGLE PASS)")
print("=" * 80)

# Optimal Conditional Aggregation Pivot
df_pivoted_fast = (
    df_sales_raw
    .groupBy("month")
    .agg(
        F.sum(F.when(F.col("category") == "Electronics", F.col("revenue")).otherwise(0.0)).alias("electronics_revenue"),
        F.sum(F.when(F.col("category") == "Apparel", F.col("revenue")).otherwise(0.0)).alias("apparel_revenue")
    )
    .orderBy("month")
)
df_pivoted_fast.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Challenge 3: Consecutive Active Days Streak Detection (Handbook J15)
# MAGIC
# MAGIC ### Scenario:
# MAGIC Find all customers who logged in for **at least 3 consecutive days**.

# COMMAND ----------

logins = [
    ("CUST_1", "2026-07-01"),
    ("CUST_1", "2026-07-02"),
    ("CUST_1", "2026-07-03"), # Streak of 3!
    ("CUST_1", "2026-07-05"),
    ("CUST_2", "2026-07-01"),
    ("CUST_2", "2026-07-03")  # Not consecutive
]
df_logins = spark.createDataFrame(logins, ["customer_id", "login_date"]).withColumn("login_date", F.to_date("login_date"))

# Technique: Date Subtraction Grouping
# (login_date - row_number()) will be CONSTANT for consecutive dates!
w_login = Window.partitionBy("customer_id").orderBy("login_date")

df_streaks = (
    df_logins
    .withColumn("rn", F.row_number().over(w_login))
    .withColumn("date_group", F.expr("date_sub(login_date, rn)"))
    .groupBy("customer_id", "date_group")
    .agg(
        F.count("login_date").alias("consecutive_days"),
        F.min("login_date").alias("streak_start"),
        F.max("login_date").alias("streak_end")
    )
    .filter(F.col("consecutive_days") >= 3)
)

print("=" * 80)
print("  CONSECUTIVE DAYS LOGIN STREAK DETECTION (>= 3 DAYS)")
print("=" * 80)
df_streaks.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Challenge 4: Finding Top N Spending Customers per Region via `dense_rank()` (Handbook J5)

# COMMAND ----------

df_cust_spend = spark.table("de_practical_db.raw_customers")

w_region = Window.partitionBy("loyalty_tier").orderBy(F.col("lifetime_spend").desc())

df_top_spenders = (
    df_cust_spend
    .withColumn("rank", F.dense_rank().over(w_region))
    .filter(F.col("rank") <= 2) # Top 2 per loyalty tier
    .select("loyalty_tier", "customer_id", "first_name", "last_name", "lifetime_spend", "rank")
)

print("=" * 80)
print("  TOP 2 SPENDING CUSTOMERS PER LOYALTY TIER (DENSE_RANK)")
print("=" * 80)
df_top_spenders.show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 5: Production Debugging Scenario Guide (Handbook J19–J25, K1–K10)
# MAGIC
# MAGIC ### Production Bug 1: Straggler Tasks (1 Task Stuck at 99%) (Handbook J20)
# MAGIC - **Symptom**: Spark job reaches 199/200 tasks in 10 seconds, but task #200 hangs for 3 hours.
# MAGIC - **Root Cause**: Severe Data Skew on a single join/aggregation key.
# MAGIC - **Resolution**: Implement **Salting** on the join key (Notebook 04) OR enable **AQE Skew Join** (`spark.sql.adaptive.skewJoin.enabled = true`).
# MAGIC
# MAGIC ### Production Bug 2: Garbage Collection (GC) Pauses & Executor Loss (Handbook J22)
# MAGIC - **Symptom**: `ExecutorLostFailure: Slave lost` or `GC time took 45% of total task time`.
# MAGIC - **Root Cause**: Excessive creation of short-lived JVM objects in memory (e.g., calling standard Python UDFs inside loops, or caching massive deserialized `MEMORY_ONLY` DataFrames).
# MAGIC - **Resolution**: Use native `pyspark.sql.functions`, convert to Vectorized `@pandas_udf`, switch storage level to `MEMORY_AND_DISK_SER` or off-heap Tungsten memory.
# MAGIC
# MAGIC ### Production Bug 3: Driver Out-Of-Memory (`Java heap space`) (Handbook J24)
# MAGIC - **Symptom**: Driver process crashes with `java.lang.OutOfMemoryError`.
# MAGIC - **Root Cause**: A developer ran `df.collect()` or `df.toPandas()` on a 500 GB DataFrame.
# MAGIC - **Resolution**: Audit codebase for `collect()`, enforce `limit(100)`, or stream results using `toLocalIterator()`.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 💡 System Design Interview: Petabyte-Scale Architecture (Handbook H1–H5, J25)
# MAGIC
# MAGIC ### System Design Question:
# MAGIC **"Design an end-to-end, enterprise Data Lakehouse platform on Databricks for a Fortune 500 Omnichannel Retailer processing 10 TB of daily streaming and batch transactions."**
# MAGIC
# MAGIC ```text
# MAGIC RAW DATA SOURCES              LAKEHOUSE ARCHITECTURE (DATABRICKS)                CONSUMPTION LAYER
# MAGIC ─────────────────          ─────────────────────────────────────────            ───────────────────
# MAGIC
# MAGIC  POS Stores (CSV/JDBC) ──┐
# MAGIC                          │    BRONZE LAYER           SILVER LAYER           GOLD LAYER
# MAGIC  Web Clickstream (JSON) ──┼─▶  bronze_orders  ───────▶ silver_orders ──────▶ gold_fact_sales ────▶ BI Dashboards
# MAGIC                          │    (Append-Only            (Cleaned, Dedupped,   (Star Schema         (PowerBI / DBSQL)
# MAGIC  IoT Sensors (Streaming)─┘     Audit Metadata)         SCD Type 2)            Data Marts)
# MAGIC                                      │                      │                      │
# MAGIC                                      └──────────────────────┴──────────────────────┘
# MAGIC                                                              │
# MAGIC                                                   UNITY CATALOG GOVERNANCE
# MAGIC                                                   • 3-Level Namespace (catalog.schema.table)
# MAGIC                                                   • Column Masking & Row-Level Security
# MAGIC                                                   • Automatic Data Lineage & System Audit Tables
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # ✅ Course Completion & Master Summary
# MAGIC
# MAGIC Congratulations! You have completed the **Databricks & PySpark Data Engineering Enterprise Practical Course** based on `Databricks_COMPLETE_HANDBOOK_2026.txt`.
# MAGIC
# MAGIC ### Summary of Completed Curriculum Modules:
# MAGIC 1. **Notebook 01**: Foundations, Multi-Format Ingestion, Schemas, Malformed Records DLQ.
# MAGIC 2. **Notebook 02**: Core Transformations, Null Safety, Vectorized Pandas UDFs, Window Functions, PII Encryption.
# MAGIC 3. **Notebook 03**: Spark Internals, Catalyst 5-Phase Optimizer, AQE, Execution Plans, Join Engine Mechanics.
# MAGIC 4. **Notebook 04**: Performance Tuning, Partitioning (`repartition` vs `coalesce`), Salting Data Skew, Memory Cache.
# MAGIC 5. **Notebook 05**: Delta Lake Core, `_delta_log` ACID Log, `MERGE INTO`, Time Travel, `OPTIMIZE`, Liquid Clustering, CDF.
# MAGIC 6. **Notebook 06**: Structured Streaming, `availableNow` Trigger, Watermarking, Stream-Static Joins, `foreachBatch`.
# MAGIC 7. **Notebook 07**: Production Medallion Architecture (Bronze -> Silver -> Gold), SCD Type 2, Star Schema Modeling.
# MAGIC 8. **Notebook 08**: Lakeflow Pipelines (DLT), Data Quality Expectations Framework, Lakeflow Jobs Orchestration.
# MAGIC 9. **Notebook 09**: Unity Catalog 3-Level Namespace, Managed vs External Tables/Volumes, Security Masking & Grants.
# MAGIC 10. **Notebook 10**: Enterprise Interview Mastery, Gaps & Islands, High-Performance Pivot, Straggler Debugging, System Design.
