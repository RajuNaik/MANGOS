# Databricks notebook source

# MAGIC %md
# MAGIC # 🏢 Notebook 01 — Foundations, Data Ingestion & Schema Enforcement
# MAGIC
# MAGIC **Handbook Sections Covered**: Part A (A1–A10), B1–B6, B11, B12, B19, B20
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Learning Objectives
# MAGIC 1. **Distributed Computing & Spark Architecture** — Hadoop vs Spark 4.0, Driver/Executor model, On-heap vs Off-heap memory.
# MAGIC 2. **Data Lakehouse Paradigm** — DWH vs Data Lake vs Data Lakehouse (Delta Lake).
# MAGIC 3. **Synthetic Dataset Foundation** — Generate realistic Fortune 500 Omnichannel Retail & Supply Chain raw data files.
# MAGIC 4. **Multi-Format Data Ingestion** — Read CSV, JSON, Parquet, JDBC (simulated), and multi-sheet Excel files.
# MAGIC 5. **Malformed Record Strategies** — Handle corrupt data using `PERMISSIVE`, `DROPMALFORMED`, `FAILFAST`, and `_corrupt_record`.
# MAGIC 6. **Strict Schema Definition** — Programmatic schemas (`StructType`, `ArrayType`, `MapType`) to prevent schema drift.
# MAGIC 7. **Compression & Storage Formats** — Compare Snappy vs Gzip vs Parquet vs Delta.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Setup & Import Dependencies

# COMMAND ----------

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, LongType, DoubleType,
    BooleanType, TimestampType, DateType, ArrayType, MapType
)

# Initialize / verify Spark session
spark = SparkSession.builder.getOrCreate()

# Create dedicated schema for practical exercises
spark.sql("CREATE DATABASE IF NOT EXISTS de_practical_db")
spark.sql("USE de_practical_db")

print("✅ Setup complete! Connected to Spark version:", spark.version)
print("✅ Active database: de_practical_db")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 1: Distributed Computing & Databricks Architecture (Handbook A1–A10)
# MAGIC
# MAGIC ### 1.1 Why Distributed Computing? (A1)
# MAGIC - **Hadoop MapReduce (Legacy)**: Wrote intermediate results to disk after every step. Fault-tolerant, but slow due to I/O bottlenecks.
# MAGIC - **Apache Spark 4.0**: Keeps intermediate data in-memory (RAM) across pipeline stages, achieving 10x–100x performance improvements.
# MAGIC
# MAGIC ### 1.2 Spark Architecture (A5)
# MAGIC - **Driver Node**: The coordinator. Builds the Directed Acyclic Graph (DAG), splits work into **Stages** (shuffle boundaries) and **Tasks** (one per partition), and schedules tasks to executors.
# MAGIC - **Worker Nodes / Executors**: JVM processes that execute tasks in parallel on data partitions.
# MAGIC - **Core Rule**: **1 CPU Core = 1 Task = 1 Data Partition** processed at any given instant.
# MAGIC
# MAGIC ### 1.3 Memory Layout: On-Heap vs Off-Heap (A8)
# MAGIC - **On-Heap Memory**: Standard JVM heap space. Managed by Garbage Collection (GC). GC pauses can degrade performance during large shuffles.
# MAGIC - **Off-Heap Memory**: Allocated outside JVM GC control (via `sun.misc.Unsafe`). Used by Project Tungsten and Databricks **Photon Engine** for direct C++ memory management, eliminating GC overhead.

# COMMAND ----------

# Display active cluster configuration parameters
print("=" * 80)
print("  SPARK CLUSTER MEMORY & EXECUTION CONFIGURATION")
print("=" * 80)
print(f"Spark Driver Memory:            {spark.conf.get('spark.driver.memory', 'Default')}")
print(f"Default Parallelism:            {spark.sparkContext.defaultParallelism}")
print(f"Default Shuffle Partitions:     {spark.conf.get('spark.sql.shuffle.partitions')}")
print(f"Adaptive Query Execution (AQE): {spark.conf.get('spark.sql.adaptive.enabled', 'true')}")
print(f"Photon Engine Enabled:          {spark.conf.get('spark.databricks.powertrain', 'N/A on CE')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 2: Generating Synthetic Enterprise Datasets
# MAGIC
# MAGIC We will create raw input files on DBFS / local storage representing raw ingestion files for a Fortune 500 Retail enterprise:
# MAGIC 1. `raw_customers.csv` (Customer master file with headers and messy values)
# MAGIC 2. `raw_orders_corrupt.csv` (Orders file with intentionally malformed lines)
# MAGIC 3. `raw_clickstream.json` (Nested JSON events from web/mobile app)
# MAGIC 4. `raw_products.parquet` (Product catalog stored in Parquet format)
# MAGIC 5. `raw_supplier_sheets.xlsx` (Multi-sheet Excel workbook for supplier inventory)

# COMMAND ----------

# Setup local file system directory for raw mock inputs
raw_data_dir = "/tmp/de_practical_raw"
os.makedirs(raw_data_dir, exist_ok=True)

np.random.seed(2026)
n_records = 1000

# ----------------------------------------------------------------------------
# 1. Raw Customers CSV
# ----------------------------------------------------------------------------
customers_data = []
for i in range(1, n_records + 1):
    customers_data.append({
        "customer_id": f"CUST_{i:06d}",
        "first_name": np.random.choice(["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Linda", "Michael"]),
        "last_name": np.random.choice(["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]),
        "email": f"user_{i}@retailcorp.com" if i % 10 != 0 else f"invalid_email_{i}",
        "phone": f"+1-555-{np.random.randint(100, 999):03d}-{np.random.randint(1000, 9999):04d}",
        "registration_date": (datetime(2023, 1, 1) + timedelta(days=int(np.random.uniform(0, 500)))).strftime("%Y-%m-%d"),
        "loyalty_tier": np.random.choice(["Bronze", "Silver", "Gold", "Platinum"], p=[0.5, 0.3, 0.15, 0.05]),
        "lifetime_spend": round(float(np.random.exponential(scale=250.0)), 2)
    })

df_cust_pd = pd.DataFrame(customers_data)
csv_cust_path = os.path.join(raw_data_dir, "customers.csv")
df_cust_pd.to_csv(csv_cust_path, index=False)

# ----------------------------------------------------------------------------
# 2. Raw Orders CSV with Corrupt/Malformed Rows (for B2 testing)
# ----------------------------------------------------------------------------
corrupt_csv_path = os.path.join(raw_data_dir, "orders_corrupt.csv")
with open(corrupt_csv_path, "w") as f:
    f.write("order_id,customer_id,order_timestamp,total_amount,payment_method,status\n")
    for i in range(1, 501):
        if i == 25:
            # Corrupt row: Too few columns
            f.write(f"ORD_000025,CUST_000010,2024-03-01 10:00:00\n")
        elif i == 75:
            # Corrupt row: Non-numeric string in decimal column
            f.write(f"ORD_000075,CUST_000020,2024-03-01 11:30:00,CORRUPT_AMOUNT,CreditCard,COMPLETED\n")
        elif i == 150:
            # Corrupt row: Extra columns
            f.write(f"ORD_000150,CUST_000030,2024-03-02 14:15:00,199.99,PayPal,COMPLETED,EXTRA_FIELD_1,EXTRA_FIELD_2\n")
        else:
            ts = (datetime(2024, 1, 1) + timedelta(hours=i*2)).strftime("%Y-%m-%d %H:%M:%S")
            amt = round(float(np.random.uniform(10.0, 500.0)), 2)
            method = np.random.choice(["CreditCard", "DebitCard", "PayPal", "ApplePay"])
            status = np.random.choice(["COMPLETED", "PENDING", "CANCELLED", "REFUNDED"], p=[0.75, 0.1, 0.1, 0.05])
            f.write(f"ORD_{i:06d},CUST_{(i%100)+1:06d},{ts},{amt},{method},{status}\n")

# ----------------------------------------------------------------------------
# 3. Raw Nested Clickstream JSON
# ----------------------------------------------------------------------------
json_clickstream_path = os.path.join(raw_data_dir, "clickstream.json")
with open(json_clickstream_path, "w") as f:
    for i in range(1, 301):
        event = {
            "event_id": f"EVT_{i:08d}",
            "session_id": f"SESS_{np.random.randint(1000, 9999)}",
            "customer_id": f"CUST_{np.random.randint(1, 100):06d}",
            "timestamp": (datetime(2024, 2, 1) + timedelta(minutes=i*5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "device": {
                "type": np.random.choice(["mobile", "desktop", "tablet"]),
                "os": np.random.choice(["iOS", "Android", "Windows", "macOS"]),
                "ip_address": f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}"
            },
            "page_events": [
                {"page": "home", "action": "view", "duration_sec": int(np.random.uniform(2, 30))},
                {"page": "product_detail", "action": "click", "duration_sec": int(np.random.uniform(5, 120))}
            ]
        }
        f.write(json.dumps(event) + "\n")

# ----------------------------------------------------------------------------
# 4. Parquet Product Catalog
# ----------------------------------------------------------------------------
products_data = []
categories = ["Electronics", "Apparel", "Home & Kitchen", "Beauty", "Sports"]
for i in range(1, 201):
    products_data.append({
        "product_id": f"PROD_{i:04d}",
        "product_name": f"Enterprise Product {i}",
        "category": np.random.choice(categories),
        "unit_price": round(float(np.random.uniform(5.0, 1000.0)), 2),
        "in_stock": bool(np.random.choice([True, False], p=[0.85, 0.15]))
    })
df_prod_pd = pd.DataFrame(products_data)
parquet_prod_path = os.path.join(raw_data_dir, "products.parquet")
df_prod_pd.to_parquet(parquet_prod_path, index=False)

print(f"✅ Generated synthetic raw dataset files in: {raw_data_dir}")
print(f"   1. Customers CSV:         {csv_cust_path}")
print(f"   2. Corrupt Orders CSV:    {corrupt_csv_path}")
print(f"   3. Clickstream JSON:      {json_clickstream_path}")
print(f"   4. Products Parquet:      {parquet_prod_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 3: Reading CSV Files Properly (Handbook B1, B12, B20)
# MAGIC
# MAGIC ### Key Rule in Production:
# MAGIC **NEVER use `inferSchema=True` in production pipelines!**
# MAGIC - `inferSchema=True` requires **two passes** over the entire dataset (Pass 1 to read and guess data types, Pass 2 to load the data).
# MAGIC - On a 1 TB file, `inferSchema=True` doubles your read I/O cost and execution time.
# MAGIC - Always supply an explicit `StructType` schema.

# COMMAND ----------

# ============================================================================
# B11 & B1: Define Explicit Schema for Customers CSV
# ============================================================================
customer_schema = StructType([
    StructField("customer_id", StringType(), False),
    StructField("first_name", StringType(), True),
    StructField("last_name", StringType(), True),
    StructField("email", StringType(), True),
    StructField("phone", StringType(), True),
    StructField("registration_date", DateType(), True),
    StructField("loyalty_tier", StringType(), True),
    StructField("lifetime_spend", DoubleType(), True)
])

# Read CSV using explicit schema
df_customers = (
    spark.read
    .format("csv")
    .option("header", "true")
    .option("dateFormat", "yyyy-MM-dd")
    .schema(customer_schema)
    .load(f"file://{csv_cust_path}")
)

print("=" * 80)
print("  PROPERLY INGESTED CUSTOMERS DATAFRAME (Explicit Schema)")
print("=" * 80)
df_customers.printSchema()
df_customers.show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 4: Handling Malformed Records (Handbook B2)
# MAGIC
# MAGIC When reading raw incoming files from external vendors or IoT streams, corrupted or malformed rows will occur. PySpark provides 3 `mode` options for `read.option("mode", ...)`:
# MAGIC
# MAGIC | Mode | Behavior | Use Case |
# MAGIC |---|---|---|
# MAGIC | `PERMISSIVE` (Default) | Sets corrupted fields to `null` and captures the full raw malformed line into a `_corrupt_record` column. | Production Quarantine / Dead-Letter-Queue (DLQ). |
# MAGIC | `DROPMALFORMED` | Silently ignores and drops any line that does not conform to the schema. | Non-critical analytics where dropped rows are acceptable. |
# MAGIC | `FAILFAST` | Immediately throws a `RuntimeException` and aborts execution upon encountering the first corrupt row. | Strict financial/compliance feeds where zero corruption is allowed. |

# COMMAND ----------

# ============================================================================
# B2: Mode 1 — PERMISSIVE with _corrupt_record Quarantine Column
# ============================================================================

# To use _corrupt_record, the field MUST be included in the StructType schema!
order_schema_permissive = StructType([
    StructField("order_id", StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("order_timestamp", TimestampType(), True),
    StructField("total_amount", DoubleType(), True),
    StructField("payment_method", StringType(), True),
    StructField("status", StringType(), True),
    StructField("_corrupt_record", StringType(), True)  # Dedicated quarantine column
])

df_orders_permissive = (
    spark.read
    .format("csv")
    .option("header", "true")
    .option("mode", "PERMISSIVE")
    .schema(order_schema_permissive)
    .load(f"file://{corrupt_csv_path}")
)

print("=" * 80)
print("  PERMISSIVE MODE: Clean Records + Quarantined Corrupt Records")
print("=" * 80)

# Filter valid records vs dead-letter queue records
df_valid_orders = df_orders_permissive.filter(F.col("_corrupt_record").isNull())
df_corrupt_orders = df_orders_permissive.filter(F.col("_corrupt_record").isNotNull())

print(f"Total Records Read: {df_orders_permissive.count()}")
print(f"Valid Records:      {df_valid_orders.count()}")
print(f"Corrupt Records:    {df_corrupt_orders.count()}")

print("\n--- QUARANTINED DEAD-LETTER ROWS (_corrupt_record) ---")
df_corrupt_orders.select("_corrupt_record").show(truncate=False)

# COMMAND ----------

# ============================================================================
# B2: Mode 2 — DROPMALFORMED
# ============================================================================
df_orders_drop = (
    spark.read
    .format("csv")
    .option("header", "true")
    .option("mode", "DROPMALFORMED")
    .schema(StructType(order_schema_permissive.fields[:-1])) # Exclude _corrupt_record
    .load(f"file://{corrupt_csv_path}")
)

print(f"DROPMALFORMED Mode Count: {df_orders_drop.count()} rows (automatically dropped corrupt lines)")

# ============================================================================
# B2: Mode 3 — FAILFAST
# ============================================================================
print("\nTesting FAILFAST mode (expected to raise exception)...")
try:
    df_orders_fail = (
        spark.read
        .format("csv")
        .option("header", "true")
        .option("mode", "FAILFAST")
        .schema(StructType(order_schema_permissive.fields[:-1]))
        .load(f"file://{corrupt_csv_path}")
    )
    df_orders_fail.count() # Trigger action
except Exception as e:
    print("✅ FAILFAST triggered as expected!")
    print(f"   Error message snippet: {str(e)[:150]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 5: Reading Nested JSON & Parquet Formats (Handbook B22, B23)
# MAGIC
# MAGIC ### 5.1 Nested Struct & Array Schema Definition
# MAGIC JSON clickstream data frequently contains nested objects (`StructType`) and arrays (`ArrayType`).

# COMMAND ----------

# Explicit schema for nested Clickstream JSON
clickstream_schema = StructType([
    StructField("event_id", StringType(), False),
    StructField("session_id", StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("device", StructType([
        StructField("type", StringType(), True),
        StructField("os", StringType(), True),
        StructField("ip_address", StringType(), True)
    ]), True),
    StructField("page_events", ArrayType(
        StructType([
            StructField("page", StringType(), True),
            StructField("action", StringType(), True),
            StructField("duration_sec", IntegerType(), True)
        ])
    ), True)
])

df_clickstream = (
    spark.read
    .format("json")
    .schema(clickstream_schema)
    .load(f"file://{json_clickstream_path}")
)

print("=" * 80)
print("  NESTED JSON CLICKSTREAM DATAFRAME")
print("=" * 80)
df_clickstream.printSchema()
df_clickstream.select(
    "event_id", 
    "device.type", 
    "device.os", 
    F.col("page_events")[0]["page"].alias("first_page")
).show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Reading Parquet Files (Handbook B12)
# MAGIC Parquet is a columnar binary format that stores schema metadata inside the file footprint (footer metadata).
# MAGIC Thus, reading Parquet files does NOT require `inferSchema` — Spark reads the metadata instantaneously.

# COMMAND ----------

df_products = spark.read.format("parquet").load(f"file://{parquet_prod_path}")

print("=" * 80)
print("  PARQUET PRODUCTS DATAFRAME (Schema Auto-Extracted from Parquet Footer)")
print("=" * 80)
df_products.printSchema()
df_products.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 6: Multi-Sheet Excel & JDBC Simulation (Handbook B3, B19)
# MAGIC
# MAGIC ### 6.1 Multi-Sheet Excel Files (B19)
# MAGIC Spark does not natively read `.xlsx` files out-of-the-box. In production, Data Engineers use `openpyxl` / `pandas` or commercial Spark packages to parse individual Excel sheets into PySpark DataFrames.

# COMMAND ----------

# Generate mock multi-sheet Excel file
excel_path = os.path.join(raw_data_dir, "supplier_data.xlsx")
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    pd.DataFrame({"supplier_id": ["SUP_01", "SUP_02"], "name": ["Acme Logistics", "Global Freight"]}).to_excel(writer, sheet_name="Suppliers", index=False)
    pd.DataFrame({"warehouse_id": ["WH_US_1", "WH_EU_1"], "capacity_sqft": [500000, 350000]}).to_excel(writer, sheet_name="Warehouses", index=False)

def read_excel_sheet(excel_file_path: str, sheet_name: str) -> pd.DataFrame:
    """Reads a specific sheet from an Excel workbook and converts to PySpark DataFrame."""
    pdf = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    return spark.createDataFrame(pdf)

df_suppliers = read_excel_sheet(excel_path, "Suppliers")
df_warehouses = read_excel_sheet(excel_path, "Warehouses")

print("=" * 80)
print("  MULTI-SHEET EXCEL READ VIA PYSPARK + OPENPYXL (B19)")
print("=" * 80)
print("Sheet 'Suppliers':")
df_suppliers.show()
print("Sheet 'Warehouses':")
df_warehouses.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2 JDBC Database Connection Simulation (Handbook B3)
# MAGIC On Databricks, reading from SQL Server, PostgreSQL, or Oracle uses `spark.read.format("jdbc")`.
# MAGIC
# MAGIC ```python
# MAGIC # Production Pattern for JDBC Read (Handbook B3):
# MAGIC df_jdbc = (
# MAGIC     spark.read
# MAGIC     .format("jdbc")
# MAGIC     .option("url", "jdbc:postgresql://db-hostname.company.com:5432/retail_db")
# MAGIC     .option("dbtable", "(SELECT customer_id, lifetime_spend FROM dbo.customers WHERE is_active = 1) AS subquery")
# MAGIC     .option("user", dbutils.secrets.get(scope="keyvault-scope", key="db-user"))
# MAGIC     .option("password", dbutils.secrets.get(scope="keyvault-scope", key="db-password"))
# MAGIC     .option("numPartitions", "8")          # Parallel JDBC fetch channels
# MAGIC     .option("partitionColumn", "customer_id_int") # Numeric column for partitioning
# MAGIC     .option("lowerBound", "1")
# MAGIC     .option("upperBound", "1000000")
# MAGIC     .load()
# MAGIC )
# MAGIC ```

# COMMAND ----------

# Save clean baseline tables into Managed Delta Tables in de_practical_db
df_customers.write.format("delta").mode("overwrite").saveAsTable("de_practical_db.raw_customers")
df_valid_orders.write.format("delta").mode("overwrite").saveAsTable("de_practical_db.raw_orders")
df_products.write.format("delta").mode("overwrite").saveAsTable("de_practical_db.raw_products")
df_clickstream.write.format("delta").mode("overwrite").saveAsTable("de_practical_db.raw_clickstream")

print("✅ Managed Delta Baseline Tables Initialized in Metastore:")
spark.sql("SHOW TABLES IN de_practical_db").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 💡 Senior Data Engineer Interview Practice (Handbook H3)
# MAGIC
# MAGIC ### Q1: Why should `inferSchema=True` never be used when ingesting large CSV files in production?
# MAGIC **Answer**: `inferSchema=True` scans the entire dataset **twice**. The first pass reads all rows to infer data types for every column; the second pass parses and builds the DataFrame. On petabyte-scale data, this doubles storage I/O and pipeline latency. Additionally, automatic inference can introduce silent schema drift if incoming data types vary across batch runs.
# MAGIC
# MAGIC ### Q2: Explain the difference between `PERMISSIVE`, `DROPMALFORMED`, and `FAILFAST` read modes. How do you implement a Dead-Letter Queue (DLQ) pattern?
# MAGIC **Answer**:
# MAGIC - `PERMISSIVE`: Keeps all rows, populates invalid fields as `null`, and writes the exact raw unparseable string to a designated `_corrupt_record` column.
# MAGIC - `DROPMALFORMED`: Quietly discards corrupt rows during ingestion.
# MAGIC - `FAILFAST`: Instantly throws a runtime exception and aborts execution if malformed data is encountered.
# MAGIC - **DLQ Pattern**: Use `PERMISSIVE` with `_corrupt_record` defined in `StructType`. Filter rows where `_corrupt_record IS NOT NULL` and write them to a separate Delta quarantine table (`bronze_dead_letter_queue`) for alerting and auditing, while allowing clean rows (`_corrupt_record IS NULL`) to proceed downstream.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # ✅ Module 01 Summary
# MAGIC
# MAGIC | Topic | Implementation & Best Practice | Handbook Reference |
# MAGIC |---|---|---|
# MAGIC | Distributed Architecture | Hadoop disk I/O vs Spark in-memory execution | A1, A2, A5 |
# MAGIC | Memory Management | On-heap JVM vs Off-heap C++ (Tungsten/Photon) | A8 |
# MAGIC | Schema Enforcement | Explicit `StructType` vs dangerous `inferSchema` | B1, B11 |
# MAGIC | Malformed Records | `PERMISSIVE` + `_corrupt_record` for Dead-Letter Quarantine | B2 |
# MAGIC | File Formats | CSV, JSON (Nested Structs/Arrays), Parquet footers | B12, B22, B23 |
# MAGIC | Excel & JDBC | `openpyxl` conversion, partitioned parallel JDBC reads | B3, B19 |
