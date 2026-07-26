# Databricks notebook source

# MAGIC %md
# MAGIC # 🏗️ Notebook 03 — Feature Store & AutoML
# MAGIC
# MAGIC **Handbook Sections Covered**: A5 (Feature Store concept), I1 (AutoML), I2 (Feature Store API), I3 (Feature Serving)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Learning Objectives
# MAGIC 1. **Feature Store** — create feature tables with primary keys, write/update features
# MAGIC 2. **Point-in-time correctness** — understand WHY this matters (training-serving skew / label leakage)
# MAGIC 3. **FeatureLookup** — build training sets with automatic point-in-time joins
# MAGIC 4. **AutoML** — run automated model selection and get a generated notebook
# MAGIC 5. **Feature Serving** — understand the real-time serving concept

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Setup

# COMMAND ----------

%pip install -q databricks-feature-engineering 2>/dev/null || echo "Feature Engineering client not available on CE — we will simulate the API patterns"

# COMMAND ----------

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import datetime, timedelta
import mlflow
import warnings
warnings.filterwarnings('ignore')

mlflow.set_experiment("/Users/{}/AI_Handbook_03_FeatureStore".format(
    spark.sql("SELECT current_user()").collect()[0][0]
))

print("✅ Setup complete!")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 1: Understanding Features (Handbook A5)
# MAGIC
# MAGIC A **feature** is a derived input column that a model actually trains/predicts on.
# MAGIC
# MAGIC The **Feature Store** solves a critical problem: the SAME feature logic must be computed
# MAGIC consistently in TWO very different contexts:
# MAGIC
# MAGIC | Context | Mode | Example |
# MAGIC |---------|------|---------|
# MAGIC | **Training** | Bulk/historical, looking BACK in time | "What was this customer's avg spend as of *last month*?" |
# MAGIC | **Serving** | Single record, real-time, low-latency | "What is this customer's avg spend *right now*?" |
# MAGIC
# MAGIC Without a Feature Store, two teams often re-implement the same feature differently,
# MAGIC causing **training-serving skew** — the model was trained on feature values computed one way,
# MAGIC but serves predictions using slightly different feature values computed another way.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Generate Synthetic E-Commerce Data

# COMMAND ----------

# ============================================================================
# SYNTHETIC DATA: A realistic e-commerce dataset for feature engineering
# ============================================================================
np.random.seed(2026)

# --- Customers ---
n_customers = 1000
customer_ids = [f'CUST_{i:05d}' for i in range(n_customers)]
signup_dates = [datetime(2022, 1, 1) + timedelta(days=int(np.random.uniform(0, 700))) for _ in range(n_customers)]

customers_data = pd.DataFrame({
    'customer_id': customer_ids,
    'signup_date': signup_dates,
    'region': np.random.choice(['US-East', 'US-West', 'Europe', 'Asia-Pacific'], n_customers, p=[0.35, 0.25, 0.25, 0.15]),
    'plan': np.random.choice(['free', 'basic', 'premium', 'enterprise'], n_customers, p=[0.4, 0.3, 0.2, 0.1]),
    'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '56+'], n_customers),
})

# --- Transactions (with timestamps for point-in-time features) ---
transactions_list = []
for i, cid in enumerate(customer_ids):
    n_txns = np.random.poisson(12) + 1
    base_date = signup_dates[i]
    for j in range(n_txns):
        txn_date = base_date + timedelta(
            days=int(np.random.uniform(1, 600)),
            hours=int(np.random.uniform(0, 23)),
            minutes=int(np.random.uniform(0, 59))
        )
        transactions_list.append({
            'transaction_id': f'TXN_{len(transactions_list):07d}',
            'customer_id': cid,
            'event_timestamp': txn_date,
            'amount': round(np.random.lognormal(mean=3.2, sigma=0.9), 2),
            'product_category': np.random.choice(['electronics', 'clothing', 'home', 'food', 'sports']),
            'channel': np.random.choice(['web', 'mobile_app', 'in_store'], p=[0.5, 0.35, 0.15]),
        })

transactions_df = pd.DataFrame(transactions_list)
transactions_df['event_timestamp'] = pd.to_datetime(transactions_df['event_timestamp'])

# --- Sessions (web/app engagement data) ---
sessions_list = []
for i, cid in enumerate(customer_ids):
    n_sessions = np.random.poisson(20) + 1
    base_date = signup_dates[i]
    for j in range(n_sessions):
        session_date = base_date + timedelta(days=int(np.random.uniform(1, 600)))
        sessions_list.append({
            'customer_id': cid,
            'session_date': session_date,
            'duration_minutes': max(0.5, np.random.exponential(scale=8)),
            'pages_viewed': np.random.poisson(5) + 1,
            'device': np.random.choice(['desktop', 'mobile', 'tablet'], p=[0.4, 0.45, 0.15]),
        })

sessions_df = pd.DataFrame(sessions_list)

# Save as Delta tables
spark.createDataFrame(customers_data).write.format("delta").mode("overwrite").saveAsTable("default.ecom_customers")
spark.createDataFrame(transactions_df).write.format("delta").mode("overwrite").saveAsTable("default.ecom_transactions")
spark.createDataFrame(sessions_df).write.format("delta").mode("overwrite").saveAsTable("default.ecom_sessions")

print(f"✅ Synthetic e-commerce data created:")
print(f"   Customers:    {len(customers_data):,} rows")
print(f"   Transactions: {len(transactions_df):,} rows")
print(f"   Sessions:     {len(sessions_df):,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 3: Feature Engineering — Building Feature Tables (Handbook I2)
# MAGIC
# MAGIC On full Databricks, you'd use the **Feature Engineering Client**:
# MAGIC ```python
# MAGIC from databricks.feature_engineering import FeatureEngineeringClient
# MAGIC fe = FeatureEngineeringClient()
# MAGIC fe.create_table(name="catalog.schema.customer_features", primary_keys=["customer_id"], df=features_df)
# MAGIC ```
# MAGIC
# MAGIC On CE, we'll build the features using Spark and save as Delta tables with a clear
# MAGIC primary key — the same underlying pattern, just without the managed Feature Store wrapper.

# COMMAND ----------

# ============================================================================
# FEATURE TABLE 1: Customer Transaction Features
# These are the features a data engineer builds and maintains via Lakeflow
# ============================================================================

# Read from Delta tables (governed data sources)
txns = spark.table("default.ecom_transactions")
custs = spark.table("default.ecom_customers")

# Reference date for "current" features
reference_date = F.lit("2024-01-15").cast("timestamp")

# Compute rolling features from transactions
customer_txn_features = (
    txns
    .groupBy("customer_id")
    .agg(
        # Spend features
        F.count("*").alias("total_orders"),
        F.sum("amount").alias("total_spend"),
        F.avg("amount").alias("avg_order_value"),
        F.stddev("amount").alias("spend_volatility"),
        F.max("amount").alias("max_order_value"),
        F.min("amount").alias("min_order_value"),
        
        # Recency features
        F.max("event_timestamp").alias("last_order_timestamp"),
        F.min("event_timestamp").alias("first_order_timestamp"),
        
        # Diversity features
        F.countDistinct("product_category").alias("unique_categories"),
        F.countDistinct("channel").alias("unique_channels"),
    )
    # Derived features
    .withColumn("days_since_last_order",
                F.datediff(reference_date, F.col("last_order_timestamp")))
    .withColumn("customer_tenure_days",
                F.datediff(reference_date, F.col("first_order_timestamp")))
    .withColumn("orders_per_month",
                F.when(F.col("customer_tenure_days") > 0,
                       F.col("total_orders") / (F.col("customer_tenure_days") / 30.0))
                .otherwise(0))
    .withColumn("spend_per_order_ratio",
                F.when(F.col("total_orders") > 0,
                       F.col("total_spend") / F.col("total_orders"))
                .otherwise(0))
    .fillna(0)
)

# Save as a feature table
customer_txn_features.write.format("delta").mode("overwrite").saveAsTable("default.customer_txn_features")

print("✅ Feature Table: default.customer_txn_features")
print(f"   Primary Key: customer_id")
print(f"   Features: {len(customer_txn_features.columns) - 1}")  # minus the PK
customer_txn_features.select(
    "customer_id", "total_orders", "avg_order_value", "days_since_last_order",
    "orders_per_month", "unique_categories"
).show(5, truncate=False)

# COMMAND ----------

# ============================================================================
# FEATURE TABLE 2: Customer Session/Engagement Features  
# ============================================================================

sessions = spark.table("default.ecom_sessions")

customer_session_features = (
    sessions
    .groupBy("customer_id")
    .agg(
        F.count("*").alias("total_sessions"),
        F.avg("duration_minutes").alias("avg_session_duration"),
        F.sum("duration_minutes").alias("total_time_on_site"),
        F.avg("pages_viewed").alias("avg_pages_per_session"),
        F.sum("pages_viewed").alias("total_pages_viewed"),
        F.max("session_date").alias("last_session_date"),
    )
    .withColumn("days_since_last_session",
                F.datediff(reference_date, F.col("last_session_date")))
    .withColumn("engagement_score",
                F.col("avg_session_duration") * F.col("avg_pages_per_session") / 10.0)
    .fillna(0)
)

customer_session_features.write.format("delta").mode("overwrite").saveAsTable("default.customer_session_features")

print("✅ Feature Table: default.customer_session_features")
print(f"   Primary Key: customer_id")
customer_session_features.select(
    "customer_id", "total_sessions", "avg_session_duration", "engagement_score"
).show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 4: Point-in-Time Correctness (Handbook A5 — CRITICAL Concept)
# MAGIC
# MAGIC > **The bug**: You want to predict "will this customer churn in March 2024?"
# MAGIC > but you accidentally use their *current* feature values (which include data from
# MAGIC > *after* March 2024). The model sees the future → artificially good training metrics
# MAGIC > → terrible real-world performance. This is called **label leakage**.
# MAGIC >
# MAGIC > **The fix**: Point-in-time joins — for each label, join features AS THEY WERE at
# MAGIC > the label's timestamp, not as they are now.

# COMMAND ----------

# ============================================================================
# POINT-IN-TIME CORRECTNESS: Why it matters and how Feature Store handles it
# ============================================================================

# Create labels: "did this customer churn?" at specific evaluation dates
# Each label has a customer_id + an event_timestamp (WHEN we're asking the question)
label_dates = [datetime(2023, 7, 1), datetime(2023, 10, 1), datetime(2024, 1, 1)]
labels_list = []

for cid in customer_ids[:200]:  # Subset for demo
    for label_date in label_dates:
        # Synthetic churn label
        churned = int(np.random.random() < 0.25)  # 25% churn rate
        labels_list.append({
            'customer_id': cid,
            'event_timestamp': label_date,
            'churned': churned,
        })

labels_df = pd.DataFrame(labels_list)
labels_spark = spark.createDataFrame(labels_df)

print(f"✅ Labels dataset: {len(labels_df)} rows")
print(f"   Each row = 'did customer X churn as of date Y?'")
labels_spark.show(6, truncate=False)

# COMMAND ----------

# ============================================================================
# THE WRONG WAY: Join features as they are NOW (leaks future information)
# ============================================================================
print("❌ THE WRONG WAY — joining current features to historical labels")
print("   This gives the model access to FUTURE information when training on past labels!")
print()

wrong_join = (
    labels_spark
    .join(spark.table("default.customer_txn_features"), on="customer_id", how="inner")
)

print("   Example: For a label from 2023-07-01, the features include data from ALL TIME")
print("   (including transactions AFTER 2023-07-01 — the model is 'cheating')")
wrong_join.select("customer_id", "event_timestamp", "total_orders", "days_since_last_order").show(4)

# COMMAND ----------

# ============================================================================
# THE RIGHT WAY: Point-in-time correct join (Handbook A5, I2)
# Only use feature values KNOWN AT the label's timestamp
# ============================================================================
print("✅ THE RIGHT WAY — point-in-time correct features")
print("   For each label, compute features using ONLY data available BEFORE that date")
print()

# On full Databricks, this is done automatically by FeatureLookup:
#   from databricks.feature_engineering import FeatureLookup
#   training_set = fe.create_training_set(
#       df=labels_spark,
#       feature_lookups=[
#           FeatureLookup(
#               table_name="catalog.schema.customer_features",
#               lookup_key="customer_id",
#               timestamp_lookup_key="event_timestamp"   # <-- THIS is what makes it point-in-time correct
#           )
#       ],
#       label="churned"
#   )

# On CE, we implement the point-in-time join manually:
txns = spark.table("default.ecom_transactions")

def compute_pit_features(labels_df, txns_df):
    """
    Point-in-time feature computation:
    For each (customer_id, event_timestamp) label, compute features using
    ONLY transactions that occurred BEFORE event_timestamp.
    """
    # Join labels with transactions, filtering to only PAST transactions
    pit_joined = (
        labels_df.alias("l")
        .join(txns_df.alias("t"),
              (F.col("l.customer_id") == F.col("t.customer_id")) &
              (F.col("t.event_timestamp") < F.col("l.event_timestamp")),
              how="left")
    )
    
    # Compute features from the filtered (past-only) transactions
    pit_features = (
        pit_joined
        .groupBy(
            F.col("l.customer_id").alias("customer_id"),
            F.col("l.event_timestamp").alias("label_timestamp"),
            F.col("l.churned").alias("churned")
        )
        .agg(
            F.count("t.transaction_id").alias("pit_total_orders"),
            F.coalesce(F.sum("t.amount"), F.lit(0)).alias("pit_total_spend"),
            F.coalesce(F.avg("t.amount"), F.lit(0)).alias("pit_avg_order_value"),
            F.max("t.event_timestamp").alias("pit_last_order"),
        )
        .withColumn("pit_days_since_last_order",
                    F.datediff(F.col("label_timestamp"), F.col("pit_last_order")))
        .fillna(0)
    )
    
    return pit_features

pit_training_data = compute_pit_features(labels_spark, txns)

print("Point-in-time correct training data:")
pit_training_data.select(
    "customer_id", "label_timestamp", "pit_total_orders", 
    "pit_avg_order_value", "pit_days_since_last_order", "churned"
).show(8, truncate=False)

print("\n💡 Notice: the SAME customer has DIFFERENT feature values for different label dates!")
print("   This is because features are computed using only data available AT that point in time.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Full Databricks Feature Store API (Handbook I2)
# MAGIC
# MAGIC On full Databricks, ALL of the above is handled for you by the Feature Engineering client:
# MAGIC
# MAGIC ```python
# MAGIC from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
# MAGIC fe = FeatureEngineeringClient()
# MAGIC
# MAGIC # 1. Create feature table (a Delta table with a declared primary key)
# MAGIC fe.create_table(
# MAGIC     name="catalog.schema.customer_features",
# MAGIC     primary_keys=["customer_id"],
# MAGIC     df=features_df,
# MAGIC     description="Rolling 30-day customer spend/activity features"
# MAGIC )
# MAGIC
# MAGIC # 2. Create training set with automatic point-in-time join
# MAGIC training_set = fe.create_training_set(
# MAGIC     df=labels_df,
# MAGIC     feature_lookups=[
# MAGIC         FeatureLookup(
# MAGIC             table_name="catalog.schema.customer_features",
# MAGIC             lookup_key="customer_id",
# MAGIC             timestamp_lookup_key="event_timestamp"
# MAGIC         )
# MAGIC     ],
# MAGIC     label="churned"
# MAGIC )
# MAGIC training_df = training_set.load_df()
# MAGIC ```
# MAGIC
# MAGIC The `timestamp_lookup_key` parameter is what makes it point-in-time correct — the Feature Store
# MAGIC automatically joins the feature values that were valid AT each label's timestamp.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 5: AutoML — Automated Model Selection (Handbook I1)
# MAGIC
# MAGIC > AutoML automatically tries multiple algorithms and hyperparameter combinations
# MAGIC > against your dataset. It gives you back **generated notebook code** — a starting
# MAGIC > point, not a black box.

# COMMAND ----------

# ============================================================================
# AUTOML: On full Databricks, this is a single API call
# ============================================================================
# On full Databricks:
#   from databricks import automl
#   summary = automl.classify(
#       dataset=training_df,
#       target_col="churned",
#       timeout_minutes=30,
#       primary_metric="f1"
#   )
#   # Returns: best model, generated notebook, trial results
#   print(summary.best_trial)

# On CE, let's simulate AutoML's behavior: try multiple algorithms and compare
print("=" * 70)
print("  SIMULATED AutoML — trying multiple algorithms automatically")
print("=" * 70)

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# Prepare features for AutoML
pit_pdf = pit_training_data.toPandas()
feature_cols = ['pit_total_orders', 'pit_total_spend', 'pit_avg_order_value', 'pit_days_since_last_order']
X_auto = pit_pdf[feature_cols].fillna(0)
y_auto = pit_pdf['churned']

X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
    X_auto, y_auto, test_size=0.2, random_state=42, stratify=y_auto
)

# AutoML candidate models (similar to what Databricks AutoML would try)
automl_candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest (100)": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
    "Random Forest (200)": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    "GBM (tuned)": GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
}

automl_results = []
for name, model in automl_candidates.items():
    with mlflow.start_run(run_name=f"AutoML_{name}") as run:
        mlflow.set_tag("experiment_type", "automl_simulation")
        mlflow.log_param("algorithm", name)
        
        # Cross-validation score (what AutoML uses internally)
        cv_scores = cross_val_score(model, X_train_a, y_train_a, cv=5, scoring='f1')
        
        # Train on full training set and evaluate on test
        model.fit(X_train_a, y_train_a)
        y_pred_a = model.predict(X_test_a)
        
        test_f1 = f1_score(y_test_a, y_pred_a)
        test_acc = accuracy_score(y_test_a, y_pred_a)
        
        mlflow.log_metric("cv_f1_mean", cv_scores.mean())
        mlflow.log_metric("cv_f1_std", cv_scores.std())
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.sklearn.log_model(model, "model")
        
        automl_results.append({
            'algorithm': name,
            'cv_f1_mean': round(cv_scores.mean(), 4),
            'cv_f1_std': round(cv_scores.std(), 4),
            'test_f1': round(test_f1, 4),
            'test_accuracy': round(test_acc, 4),
            'run_id': run.info.run_id,
        })

automl_df = pd.DataFrame(automl_results).sort_values('cv_f1_mean', ascending=False)
print("\n" + "=" * 90)
print("  AutoML RESULTS — All candidates ranked by cross-validated F1")
print("=" * 90)
print(automl_df.to_string(index=False))

best_automl = automl_df.iloc[0]
print(f"\n🏆 AutoML winner: {best_automl['algorithm']}")
print(f"   CV F1: {best_automl['cv_f1_mean']:.4f} ± {best_automl['cv_f1_std']:.4f}")
print(f"   Test F1: {best_automl['test_f1']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### AutoML Generated Notebook (Concept)
# MAGIC
# MAGIC On full Databricks, AutoML doesn't just give you a model — it generates a **full, editable notebook**
# MAGIC with all the code to reproduce the best result. This is the key differentiator from other AutoML tools:
# MAGIC you get a **starting point**, not a black box.
# MAGIC
# MAGIC The generated notebook includes:
# MAGIC - Data preprocessing code
# MAGIC - Feature engineering steps
# MAGIC - The winning algorithm with tuned hyperparameters
# MAGIC - Evaluation metrics and charts
# MAGIC - MLflow tracking for reproducibility

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 6: Feature Serving (Handbook I3)
# MAGIC
# MAGIC > Feature Serving = real-time, low-latency REST endpoint for looking up a specific
# MAGIC > entity's feature values at INFERENCE time (e.g., "give me customer X's current features
# MAGIC > right now, so I can score this transaction for fraud").
# MAGIC >
# MAGIC > This is distinct from the batch/offline Feature Store above, which is for TRAINING.

# COMMAND ----------

# ============================================================================
# FEATURE SERVING: Simulated real-time feature lookup
# ============================================================================
# On full Databricks:
#   fe.create_feature_spec(name="catalog.schema.customer_feature_spec", features=[...])
#   fe.serve_feature_spec(name="catalog.schema.customer_feature_spec")
#   # Then call via REST endpoint for real-time feature lookups

# On CE, simulate with a simple lookup function:

# Load the feature table into a dictionary for fast lookup (simulates online store)
feature_table = spark.table("default.customer_txn_features").toPandas().set_index('customer_id')

def serve_features(customer_id: str) -> dict:
    """
    Simulates a Feature Serving endpoint.
    On full Databricks, this would be a low-latency REST API backed by
    Lakebase (an online feature store).
    """
    if customer_id in feature_table.index:
        row = feature_table.loc[customer_id]
        return {
            'customer_id': customer_id,
            'total_orders': int(row.get('total_orders', 0)),
            'avg_order_value': round(float(row.get('avg_order_value', 0)), 2),
            'days_since_last_order': int(row.get('days_since_last_order', 0)),
            'orders_per_month': round(float(row.get('orders_per_month', 0)), 3),
            'unique_categories': int(row.get('unique_categories', 0)),
        }
    else:
        return {'error': f'Customer {customer_id} not found'}

# Simulate real-time serving requests
print("=" * 70)
print("  FEATURE SERVING — Real-time feature lookup simulation")
print("=" * 70)
for cid in ['CUST_00001', 'CUST_00050', 'CUST_00999']:
    features = serve_features(cid)
    print(f"\n  Request: GET /features?customer_id={cid}")
    print(f"  Response: {features}")

print(f"\n📌 On full Databricks:")
print(f"   - Feature Serving = managed REST endpoint")
print(f"   - Backed by Lakebase (low-latency online store)")
print(f"   - Same feature definitions as offline Feature Store")
print(f"   - A senior DE decides which features need online serving vs. offline only")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # ✅ Key Takeaways
# MAGIC
# MAGIC | Concept | What We Did | Handbook Section |
# MAGIC |---------|------------|------------------|
# MAGIC | Feature engineering | Built transaction + session features from raw data | A5 |
# MAGIC | Feature tables | Saved as Delta tables with primary keys | I2 |
# MAGIC | Point-in-time correctness | Showed the WRONG join vs the RIGHT join | A5, I2 |
# MAGIC | Label leakage | Demonstrated how using future data corrupts training | A5 |
# MAGIC | FeatureLookup (concept) | Explained automatic PIT joins on full Databricks | I2 |
# MAGIC | AutoML | Tried 8 algorithms, found best by CV F1 | I1 |
# MAGIC | Feature Serving | Simulated real-time feature lookup | I3 |
# MAGIC
# MAGIC ### 🔗 Full Platform Mapping
# MAGIC | CE Approach | Full Databricks |
# MAGIC |---|---|
# MAGIC | Delta table with primary key | `fe.create_table()` — managed Feature Store |
# MAGIC | Manual PIT join with filter | `FeatureLookup(timestamp_lookup_key=...)` — automatic |
# MAGIC | Loop over sklearn models | `databricks.automl.classify()` — one API call |
# MAGIC | Python dict lookup | Feature Serving REST endpoint backed by Lakebase |
# MAGIC
# MAGIC **Next**: Notebook 04 — Embeddings, RAG, and Vector Search — the foundation of modern GenAI on Databricks.
