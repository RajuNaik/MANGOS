# Databricks notebook source

# MAGIC %md
# MAGIC # ⚙️ Notebook 08 — Lakeflow Pipelines, Orchestration & Parameterization
# MAGIC
# MAGIC **Handbook Sections Covered**: Part F1–F4, F16–F20, B25, H1–H5
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Learning Objectives
# MAGIC 1. **Lakeflow Pipelines (formerly DLT)** — Declarative ETL framework, Live Tables vs Streaming Live Tables.
# MAGIC 2. **Data Quality Expectations Framework** — Implement `@dlt.expect` (Warn), `@dlt.expect_or_drop` (Drop), and `@dlt.expect_or_fail` (Fail).
# MAGIC 3. **Lakeflow Jobs & Orchestration** — Task DAG dependencies, failure notifications, retries, and timeout parameters.
# MAGIC 4. **Notebook Parameterization via Widgets** — `dbutils.widgets` for dynamic parameter injection from ADF / Airflow / Lakeflow Jobs.
# MAGIC 5. **Modern Operational Data (2026)** — Lakehouse IQ, Lakebase, and Operational Delta tables.
# MAGIC 6. **CE Practical Simulation** — Code-first Expectation Decorator framework running seamlessly on Community Edition.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Setup & Widget Parameterization (Handbook B25)

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()
spark.sql("USE de_practical_db")

# ============================================================================
# B25: Databricks Notebook Widgets for Parameterization
# ============================================================================
# Create Widgets (Text & Dropdown parameters for Lakeflow Jobs orchestration)
dbutils.widgets.text("env", "DEVELOPMENT", "Environment")
dbutils.widgets.dropdown("batch_size", "1000", ["100", "500", "1000", "5000"], "Batch Size Limit")
dbutils.widgets.text("processing_date", "2026-07-26", "Processing Date (YYYY-MM-DD)")

# Retrieve Widget Parameter Values
env_param = dbutils.widgets.get("env")
batch_size_param = int(dbutils.widgets.get("batch_size"))
processing_date_param = dbutils.widgets.get("processing_date")

print("=" * 80)
print("  NOTEBOOK WIDGET PARAMETERS INJECTED FROM ORCHESTRATOR (B25)")
print("=" * 80)
print(f"Target Environment: {env_param}")
print(f"Batch Size Limit:   {batch_size_param}")
print(f"Processing Date:    {processing_date_param}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 1: Lakeflow Pipelines (formerly Delta Live Tables / DLT) (Handbook F1–F4)
# MAGIC
# MAGIC ### What is Lakeflow Pipelines / DLT?
# MAGIC - A **declarative framework** for building reliable data pipelines.
# MAGIC - You define WHAT data transformations you want using simple Python/SQL decorators (`@dlt.table`), and Databricks manages the HOW (cluster sizing, task orchestration, lineage graph, data quality enforcement, and automatic compaction/vacuuming).
# MAGIC
# MAGIC ### 3 Data Quality Expectations (Handbook F3):
# MAGIC 1. **`@dlt.expect("rule", "condition")`**: **WARN** — Logs quality metric failures to event log, but permits invalid records into table.
# MAGIC 2. **`@dlt.expect_or_drop("rule", "condition")`**: **DROP** — Silently drops invalid records from target table while logging counts.
# MAGIC 3. **`@dlt.expect_or_fail("rule", "condition")`**: **FAIL** — Immediately halts and fails the entire pipeline if a single record fails constraint.

# COMMAND ----------

# ============================================================================
# DLT Pipeline Code Pattern (Production DLT Syntax)
# ============================================================================
# Note: On full Databricks DLT runtime, you import dlt directly:
# import dlt
#
# @dlt.table(
#     name="dlt_bronze_orders",
#     comment="Raw ingested orders table with DLT tracking"
# )
# @dlt.expect("valid_order_id", "order_id IS NOT NULL")
# def dlt_bronze_orders():
#     return spark.readStream.format("delta").table("de_practical_db.raw_orders")
#
# @dlt.table(
#     name="dlt_silver_orders_clean",
#     comment="Cleansed orders dropping invalid amounts"
# )
# @dlt.expect_or_drop("positive_total_amount", "total_amount > 0.0")
# @dlt.expect_or_fail("valid_status", "status IN ('COMPLETED', 'PENDING', 'CANCELLED', 'REFUNDED')")
# def dlt_silver_orders_clean():
#     return dlt.read_stream("dlt_bronze_orders")

print("✅ DLT Production Code Pattern Defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 2: Code-First Expectation Simulator Framework for CE
# MAGIC
# MAGIC Since DLT UI is unavailable on Databricks Community Edition, we implement an executable **Python Decorator Expectation Simulator Framework** that enforces DLT expectations (`WARN`, `DROP`, `FAIL`) directly in PySpark.

# COMMAND ----------

class DLTExpectationSimulator:
    """Simulates Lakeflow / DLT Expectations on standard PySpark DataFrames."""
    
    @staticmethod
    def enforce_expectations(df, warn_rules: dict = None, drop_rules: dict = None, fail_rules: dict = None):
        total_input_rows = df.count()
        df_result = df
        
        # 1. Check FAIL Rules
        if fail_rules:
            for rule_name, condition in fail_rules.items():
                failed_count = df_result.filter(f"NOT ({condition})").count()
                if failed_count > 0:
                    raise ValueError(f"❌ DLT EXPECT_OR_FAIL VIOLATION! Rule '{rule_name}' failed on {failed_count} rows. Pipeline halted!")
                print(f"✅ DLT Expect_Or_Fail Rule '{rule_name}' Passed.")

        # 2. Check WARN Rules
        if warn_rules:
            for rule_name, condition in warn_rules.items():
                warn_count = df_result.filter(f"NOT ({condition})").count()
                print(f"⚠️ DLT Expect (Warn) Rule '{rule_name}': {warn_count} non-compliant rows logged to audit log.")

        # 3. Check DROP Rules
        if drop_rules:
            for rule_name, condition in drop_rules.items():
                before_drop = df_result.count()
                df_result = df_result.filter(condition)
                dropped_count = before_drop - df_result.count()
                print(f"✂️ DLT Expect_Or_Drop Rule '{rule_name}': Dropped {dropped_count} invalid rows.")

        print(f"\n📊 Summary: Input Rows = {total_input_rows} | Output Clean Rows = {df_result.count()}")
        return df_result

# COMMAND ----------

# Test DLT Simulator on Orders Data
df_raw_orders = spark.table("de_practical_db.raw_orders")

print("=" * 80)
print("  EXECUTING LAKEFLOW PIPELINES EXPECTATIONS SIMULATOR")
print("=" * 80)

warn_constraints = {
    "warn_null_customer": "customer_id IS NOT NULL"
}
drop_constraints = {
    "drop_negative_amount": "total_amount > 0.0"
}
fail_constraints = {
    "fail_invalid_order_id": "order_id IS NOT NULL"
}

df_lakeflow_silver_orders = DLTExpectationSimulator.enforce_expectations(
    df_raw_orders,
    warn_rules=warn_constraints,
    drop_rules=drop_constraints,
    fail_rules=fail_constraints
)

df_lakeflow_silver_orders.write.format("delta").mode("overwrite").saveAsTable("de_practical_db.lakeflow_silver_orders")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 3: Lakeflow Jobs & DAG Task Orchestration (Handbook F16–F20)
# MAGIC
# MAGIC ### Lakeflow Jobs (formerly Databricks Jobs):
# MAGIC - The native workflow orchestrator in Databricks.
# MAGIC - Configures multi-task DAGs (Directed Acyclic Graphs) where Task B depends on Task A (`depends_on`).
# MAGIC - **Task Types**: Notebooks, DLT Pipelines, Python Scripts, SQL Queries, dbt models, Jar tasks.
# MAGIC - **Production Best Practices**:
# MAGIC   - Set **Timeouts** (`timeout_seconds`) to prevent runaway infinite loops.
# MAGIC   - Configure **Retries** (`max_retries = 2`, `min_retry_interval_millis = 30000`).
# MAGIC   - Set up **Email / Slack / PagerDuty Notifications** on Failure (`on_failure`).

# COMMAND ----------

# Simulated Orchestrator DAG Workflow Output
print("=" * 80)
print("  LAKEFLOW JOBS MULTI-TASK DAG ORCHESTRATION CONFIGURATION")
print("=" * 80)

job_dag_spec = {
    "job_name": "production_retail_medallion_pipeline_daily",
    "trigger": {"cron_schedule": "0 0 2 * * ?"}, # Daily at 2 AM
    "tasks": [
        {
            "task_key": "Task_1_Ingest_Bronze",
            "notebook_path": "01_Foundations_Ingestion_Schemas",
            "timeout_seconds": 1800,
            "max_retries": 2
        },
        {
            "task_key": "Task_2_Transform_Silver",
            "depends_on": [{"task_key": "Task_1_Ingest_Bronze"}],
            "notebook_path": "07_Medallion_Architecture_DataMesh",
            "timeout_seconds": 3600
        },
        {
            "task_key": "Task_3_Aggregate_Gold",
            "depends_on": [{"task_key": "Task_2_Transform_Silver"}],
            "notebook_path": "07_Medallion_Architecture_DataMesh",
            "timeout_seconds": 1800
        }
    ],
    "email_notifications": {
        "on_failure": ["data-eng-oncall@retailcorp.com"]
    }
}

print(json.dumps(job_dag_spec, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 4: Modern Operational Data (2026 Edition) (Handbook H1–H5)
# MAGIC
# MAGIC ### 1. Operational Delta / Lakebase (H1, H2)
# MAGIC - Historically, Lakehouses were designed for batch analytical reads, while operational apps (like transactional web stores) required low-latency row lookups from PostgreSQL/MySQL.
# MAGIC - **Operational Delta (Lakebase)**: Provides sub-second, low-latency point-lookups and transactional updates directly against Delta tables, bridging OLTP and OLAP workloads.
# MAGIC
# MAGIC ### 2. Lakehouse IQ & ZeroOps (H3, H4)
# MAGIC - **Lakehouse IQ**: Semantic AI engine that understands enterprise schemas, lineage, queries, and business jargon to auto-tune query plans and power natural language queries.
# MAGIC - **ZeroOps**: Autonomous table maintenance where Databricks automatically runs `OPTIMIZE`, `VACUUM`, and index clustering in the background without scheduled maintenance jobs.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 💡 Senior Data Engineer Interview Practice (Handbook H3)
# MAGIC
# MAGIC ### Q1: Compare Delta Live Tables (Lakeflow Pipelines) with traditional Apache Airflow DAG orchestration.
# MAGIC **Answer**:
# MAGIC - **Traditional Orchestration (Airflow / Lakeflow Jobs)**: Imperative task orchestration. You explicitly define task execution order (`Task A >> Task B`), manage cluster compute provisioning, write manual error handling, and manually code table maintenance jobs.
# MAGIC - **Lakeflow Pipelines (DLT)**: Declarative data pipeline framework. You define data relationships using `@dlt.table` and `@dlt.expect` quality rules. DLT automatically infers task dependencies, builds the DAG, auto-scales compute clusters, enforces data quality expectations, tracks data lineage, and auto-maintains tables (`OPTIMIZE` / `VACUUM`).
# MAGIC
# MAGIC ### Q2: Explain the behavior of DLT expectation handling modes: `expect`, `expect_or_drop`, and `expect_or_fail`.
# MAGIC **Answer**:
# MAGIC 1. **`@dlt.expect("rule", "condition")`**: Logs quality metric failures to the DLT event log dashboard, but permits invalid rows to be written into the target table.
# MAGIC 2. **`@dlt.expect_or_drop("rule", "condition")`**: Automatically filters out and discards invalid rows from the target table while recording the count of dropped rows.
# MAGIC 3. **`@dlt.expect_or_fail("rule", "condition")`**: Instantly halts and fails the entire pipeline execution upon encountering the first invalid record, preventing any partial dataset writes.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # ✅ Module 08 Summary
# MAGIC
# MAGIC | Feature | Implementation Method | Handbook Reference |
# MAGIC |---|---|---|
# MAGIC | Pipeline Framework | Lakeflow Pipelines (DLT) `@dlt.table` declarative syntax | F1, F2 |
# MAGIC | Data Expectations | `@dlt.expect` (Warn), `expect_or_drop` (Drop), `expect_or_fail` (Fail) | F3 |
# MAGIC | Job Orchestration | Lakeflow Jobs DAG task dependencies (`depends_on`) & retries | F16, F17 |
# MAGIC | Parameter Injection| Notebook Widgets (`dbutils.widgets.text`, `get`) | B25 |
# MAGIC | Operational Data | Operational Delta (Lakebase) & ZeroOps autonomous tuning | H1–H5 |
