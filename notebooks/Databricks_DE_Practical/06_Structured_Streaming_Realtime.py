# Databricks notebook source

# MAGIC %md
# MAGIC # 🌊 Notebook 06 — Structured Streaming & Real-Time Lakehouse Engine
# MAGIC
# MAGIC **Handbook Sections Covered**: Part F5–F15, J13–J18, K16–K20
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Learning Objectives
# MAGIC 1. **Structured Streaming Architecture** — Unbounded DataFrames, Micro-batching, Checkpointing, and Write-Ahead Logs (WAL).
# MAGIC 2. **Output Modes & Triggers** — Compare `append`, `update`, `complete`, and `availableNow=True` (cost-effective batch streaming).
# MAGIC 3. **Event-Time Processing & Watermarking** — Handle out-of-order and late-arriving IoT data using `.withWatermark()`.
# MAGIC 4. **Stream-Static & Stream-Stream Joins** — Enrich live streams with static dimension lookup tables.
# MAGIC 5. **`foreachBatch` Micro-Batch Sink** — Execute custom micro-batch logic, including `MERGE INTO` Delta upserts.
# MAGIC 6. **Stateful Streaming Operations** — Manage streaming state memory and avoid memory leaks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Setup & Data Directory Initialization

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, TimestampType, IntegerType
)
import os
import shutil
import time

spark = SparkSession.builder.getOrCreate()
spark.sql("USE de_practical_db")

# Setup streaming directories
stream_base_dir = "/tmp/de_practical_streaming"
stream_source_dir = os.path.join(stream_base_dir, "raw_source")
stream_checkpoint_dir = os.path.join(stream_base_dir, "checkpoint")

# Clean previous streaming test directories
for d in [stream_source_dir, stream_checkpoint_dir]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

print("✅ Streaming directory environment initialized:", stream_base_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 1: Structured Streaming Architecture & Mechanics (Handbook F5–F8)
# MAGIC
# MAGIC ### Key Concepts:
# MAGIC - **Unbounded DataFrame**: Structured Streaming treats incoming data streams as an infinitely growing table.
# MAGIC - **Exactly-Once Semantics**: Achieved through **Replayable Sources** + **Idempotent Sinks** + **State Checkpointing** (`_checkpointLocation`).
# MAGIC - **Checkpointing (`checkpointLocation`)**: Logs the exact byte offset processing state of each source partition. If a cluster crashes, Spark resumes from the last committed offset.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 2: Generating Streaming Source Data (IoT Warehouse Sensors)
# MAGIC
# MAGIC We will write JSON files into `stream_source_dir` to simulate a continuous feed of warehouse IoT temperature/humidity sensor events.

# COMMAND ----------

# Schema for incoming IoT Sensor readings
sensor_schema = StructType([
    StructField("sensor_id", StringType(), True),
    StructField("warehouse_id", StringType(), True),
    StructField("reading_timestamp", TimestampType(), True),
    StructField("temperature_celsius", DoubleType(), True),
    StructField("humidity_pct", DoubleType(), True)
])

# Write Initial File (Batch 1) into streaming source directory
json_batch1 = """{"sensor_id":"SENS_101","warehouse_id":"WH_US_1","reading_timestamp":"2026-07-26 10:00:00","temperature_celsius":22.5,"humidity_pct":45.0}
{"sensor_id":"SENS_102","warehouse_id":"WH_US_1","reading_timestamp":"2026-07-26 10:01:00","temperature_celsius":28.9,"humidity_pct":52.0}
{"sensor_id":"SENS_103","warehouse_id":"WH_EU_1","reading_timestamp":"2026-07-26 10:02:00","temperature_celsius":18.2,"humidity_pct":40.0}"""

with open(os.path.join(stream_source_dir, "batch_01.json"), "w") as f:
    f.write(json_batch1)

print(f"✅ Generated Batch 1 JSON file in streaming landing zone: {stream_source_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 3: Reading & Writing Stream with Triggers (`availableNow`) (Handbook F9, F10)
# MAGIC
# MAGIC ### Trigger Options (Handbook F9):
# MAGIC 1. `Trigger.ProcessingTime("10 seconds")`: Standard micro-batch processing every 10s.
# MAGIC 2. **`Trigger.AvailableNow()`** (*Databricks Best Practice for Batch Streaming*): Processes all available landed data in optimal micro-batches and **automatically terminates the stream**. Avoids paying for 24/7 idle cluster compute!

# COMMAND ----------

# Create Streaming Read DataFrame
df_stream_raw = (
    spark.readStream
    .format("json")
    .schema(sensor_schema)
    .option("maxFilesPerTrigger", 1) # Control rate limit per micro-batch
    .load(f"file://{stream_source_dir}")
)

print(f"Is df_stream_raw a Streaming DataFrame? {df_stream_raw.isStreaming}")

# Simple Transformation: Filter high-temperature alerts
df_alerts = df_stream_raw.filter(F.col("temperature_celsius") > 25.0)

# Write Stream to Delta Sink using Trigger.AvailableNow()
query_alerts = (
    df_alerts
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", f"file://{stream_checkpoint_dir}/alerts")
    .trigger(availableNow=True) # Process all available data, then stop
    .table("de_practical_db.iot_temperature_alerts")
)

query_alerts.awaitTermination()
print("✅ Streaming micro-batch completed via trigger(availableNow=True).")

# Verify output in Delta table
print("\n--- High Temperature Alerts Delta Table ---")
spark.table("de_practical_db.iot_temperature_alerts").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 4: Event-Time Processing & Watermarking (`withWatermark`) (Handbook F11, F12)
# MAGIC
# MAGIC - **Event Time**: The timestamp embedded when the event was generated at the IoT device (`reading_timestamp`).
# MAGIC - **Processing Time**: The wall-clock time when the Spark cluster received the data.
# MAGIC - **Watermarking**: Tells Spark how long to wait for **late-arriving data** before dropping it from streaming aggregation state memory.
# MAGIC
# MAGIC ```python
# MAGIC # Watermark Definition (Handbook F11):
# MAGIC df.withWatermark("event_time_column", "delay_threshold")
# MAGIC # Example: "10 minutes" delay threshold means Spark keeps state for events up to 10 minutes late.
# MAGIC # Events arriving > 10 minutes late are dropped automatically to prevent OOM state leakage.
# MAGIC ```

# COMMAND ----------

# Add Late-Arriving File (Batch 2) with simulated out-of-order timestamp
json_batch2 = """{"sensor_id":"SENS_101","warehouse_id":"WH_US_1","reading_timestamp":"2026-07-26 10:05:00","temperature_celsius":23.1,"humidity_pct":46.0}
{"sensor_id":"SENS_101","warehouse_id":"WH_US_1","reading_timestamp":"2026-07-26 09:40:00","temperature_celsius":31.0,"humidity_pct":60.0}""" # 25 mins late!

with open(os.path.join(stream_source_dir, "batch_02.json"), "w") as f:
    f.write(json_batch2)

# Stream Aggregation with Watermarking
df_watermarked_agg = (
    spark.readStream
    .format("json")
    .schema(sensor_schema)
    .load(f"file://{stream_source_dir}")
    .withWatermark("reading_timestamp", "10 minutes") # 10 minute watermark threshold
    .groupBy(
        F.window("reading_timestamp", "5 minutes"), # 5-minute tumbling window
        "warehouse_id"
    )
    .agg(
        F.avg("temperature_celsius").alias("avg_temp"),
        F.max("temperature_celsius").alias("max_temp"),
        F.count("sensor_id").alias("reading_count")
    )
)

query_agg = (
    df_watermarked_agg
    .writeStream
    .format("delta")
    .outputMode("complete") # Complete mode requires window aggregation
    .option("checkpointLocation", f"file://{stream_checkpoint_dir}/agg")
    .trigger(availableNow=True)
    .table("de_practical_db.iot_warehouse_window_stats")
)

query_agg.awaitTermination()

print("=" * 80)
print("  WATERMARKED 5-MINUTE TUMBLING WINDOW AGGREGATION")
print("=" * 80)
spark.table("de_practical_db.iot_warehouse_window_stats").select("window.start", "window.end", "warehouse_id", "avg_temp", "reading_count").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 5: Stream-Static Join (Handbook F13)
# MAGIC
# MAGIC Joining a live real-time stream (IoT Sensor events) with a static dimension lookup table (`raw_products` or warehouse metadata).

# COMMAND ----------

# Static Lookup Data
df_static_warehouses = spark.createDataFrame([
    ("WH_US_1", "New York Logistics Hub", "North America"),
    ("WH_EU_1", "Frankfurt Fulfillment Center", "Europe")
], ["warehouse_id", "warehouse_name", "region"])

# Stream DataFrame
df_stream_sensors = (
    spark.readStream
    .format("json")
    .schema(sensor_schema)
    .load(f"file://{stream_source_dir}")
)

# Stream-Static Join
df_enriched_stream = df_stream_sensors.join(
    df_static_warehouses,
    "warehouse_id",
    "inner"
)

query_join = (
    df_enriched_stream
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", f"file://{stream_checkpoint_dir}/join")
    .trigger(availableNow=True)
    .table("de_practical_db.iot_enriched_readings")
)

query_join.awaitTermination()

print("=" * 80)
print("  STREAM-STATIC JOIN ENRICHED OUTPUT")
print("=" * 80)
spark.table("de_practical_db.iot_enriched_readings").select("sensor_id", "warehouse_name", "region", "temperature_celsius").show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 6: Custom Micro-Batch Sinks via `foreachBatch` (Handbook F14)
# MAGIC
# MAGIC When standard `writeStream` sinks (`append`/`complete`) are insufficient, `foreachBatch` allows running arbitrary PySpark code on every micro-batch DataFrame, such as executing a Delta **`MERGE INTO`** upsert or writing to multiple destinations simultaneously.

# COMMAND ----------

# Target Upsert Table
spark.sql("""
CREATE TABLE IF NOT EXISTS de_practical_db.iot_latest_sensor_state (
    sensor_id STRING,
    warehouse_id STRING,
    last_reading_timestamp TIMESTAMP,
    latest_temperature DOUBLE
) USING DELTA
""")

def upsert_to_delta_microbatch(microbatch_df, batch_id):
    """
    Executes Delta MERGE INTO for each streaming micro-batch.
    Ensures state updates happen idempotently without duplicating rows.
    """
    print(f"--- Processing Micro-Batch ID: {batch_id} ---")
    delta_target = DeltaTable.forName(spark, "de_practical_db.iot_latest_sensor_state")
    
    # Deduplicate microbatch rows before merging
    window_spec = Window.partitionBy("sensor_id").orderBy(F.col("reading_timestamp").desc())
    microbatch_dedup = (
        microbatch_df
        .withColumn("rn", F.row_number().over(window_spec))
        .filter(F.col("rn") == 1)
        .drop("rn")
    )
    
    (
        delta_target.alias("target")
        .merge(
            microbatch_dedup.alias("source"),
            "target.sensor_id = source.sensor_id"
        )
        .whenMatchedUpdate(set={
            "last_reading_timestamp": "source.reading_timestamp",
            "latest_temperature": "source.temperature_celsius"
        })
        .whenNotMatchedInsert(values={
            "sensor_id": "source.sensor_id",
            "warehouse_id": "source.warehouse_id",
            "last_reading_timestamp": "source.reading_timestamp",
            "latest_temperature": "source.temperature_celsius"
        })
        .execute()
    )

# Run foreachBatch Stream
query_foreach = (
    df_stream_sensors
    .writeStream
    .option("checkpointLocation", f"file://{stream_checkpoint_dir}/foreach")
    .trigger(availableNow=True)
    .foreachBatch(upsert_to_delta_microbatch)
    .start()
)

query_foreach.awaitTermination()

print("=" * 80)
print("  FOREACHBATCH DELTA MERGE UPSERT RESULTS")
print("=" * 80)
spark.table("de_practical_db.iot_latest_sensor_state").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 💡 Senior Data Engineer Interview Practice (Handbook H3)
# MAGIC
# MAGIC ### Q1: Explain how Structured Streaming achieves Exactly-Once Processing guarantees end-to-end.
# MAGIC **Answer**: Exactly-Once Processing requires three distinct components:
# MAGIC 1. **Replayable Source**: The data source must support re-reading data from specified offsets (e.g., Apache Kafka offset replay, Delta Change Data Feed, or cloud storage directory listings).
# MAGIC 2. **State Checkpointing & Write-Ahead Logs (WAL)**: Spark logs source offset ranges to `checkpointLocation/offsets` BEFORE processing each micro-batch. If a crash occurs, Spark re-reads the exact uncommitted offset range.
# MAGIC 3. **Idempotent Sink**: The output target must support idempotent writes (e.g., Delta Lake ACID transactions or `foreachBatch` with Delta `MERGE INTO`). If a micro-batch is re-executed after a failure, the idempotent sink ensures duplicate rows are overwritten rather than duplicated.
# MAGIC
# MAGIC ### Q2: What is the purpose of `withWatermark()` in stateful streaming? What happens if data arrives past the watermark threshold?
# MAGIC **Answer**: In stateful operations (like windowed aggregations or stream-stream joins), Spark maintains intermediate state memory in the cluster state store. Without watermarking, Spark would have to keep state indefinitely because late data could arrive anytime, causing an eventual **Out-Of-Memory (OOM)** crash.
# MAGIC `withWatermark("timestamp_col", "10 minutes")` establishes a sliding threshold: `Watermark = Max(EventTime) - 10 minutes`.
# MAGIC Any record whose `reading_timestamp` is **older than the watermark threshold is discarded automatically** and excluded from state memory and aggregation results.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # ✅ Module 06 Summary
# MAGIC
# MAGIC | Concept | Implementation Method | Handbook Reference |
# MAGIC |---|---|---|
# MAGIC | Fault Tolerance | `checkpointLocation` + Replayable Source + Delta Sink | F5, F8 |
# MAGIC | Batch Streaming | `trigger(availableNow=True)` | F9 |
# MAGIC | Watermarking | `.withWatermark("event_time", "10 minutes")` | F11, F12 |
# MAGIC | Stream Enrichment | Stream-Static Join (`stream_df.join(static_df, "key")`) | F13 |
# MAGIC | Custom Sinks | `writeStream.foreachBatch(upsert_func)` | F14 |
