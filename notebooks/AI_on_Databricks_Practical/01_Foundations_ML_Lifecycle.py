# Databricks notebook source

# MAGIC %md
# MAGIC # 🧠 Notebook 01 — Foundations of AI/ML & the Classic ML Lifecycle
# MAGIC
# MAGIC **Handbook Sections Covered**: A1 (AI vs ML vs DL vs GenAI), A2 (Why AI on a data platform), A3 (Classic ML lifecycle)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Learning Objectives
# MAGIC By the end of this notebook you will:
# MAGIC 1. Understand the **nesting-doll hierarchy**: AI > ML > Deep Learning > GenAI
# MAGIC 2. See concrete code examples of **each layer** (rule-based AI, classic ML, deep learning, generative AI)
# MAGIC 3. Walk through the **complete classic ML lifecycle** on Databricks: Feature Engineering → Training → Tracking → Evaluation → Comparison
# MAGIC 4. Understand **why** AI work belongs on a governed data platform (the Databricks pitch)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Setup

# COMMAND ----------

# Install additional libraries (most are pre-installed on ML Runtime)
%pip install -q xgboost lightgbm

# COMMAND ----------

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

# Reset any existing MLflow experiment
mlflow.set_experiment("/Users/{}/AI_Handbook_01_Foundations".format(
    spark.sql("SELECT current_user()").collect()[0][0]
))

print("✅ Setup complete!")
print(f"   MLflow version: {mlflow.__version__}")
print(f"   Spark version: {spark.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 1: The Nesting-Doll Hierarchy (Handbook A1)
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────┐
# MAGIC │  ARTIFICIAL INTELLIGENCE (AI)                                   │
# MAGIC │  Any technique making a computer perform human-like tasks       │
# MAGIC │  ┌─────────────────────────────────────────────────────────┐   │
# MAGIC │  │  MACHINE LEARNING (ML)                                   │   │
# MAGIC │  │  System LEARNS patterns from data                        │   │
# MAGIC │  │  ┌─────────────────────────────────────────────────┐   │   │
# MAGIC │  │  │  DEEP LEARNING                                   │   │   │
# MAGIC │  │  │  Neural networks with many layers                │   │   │
# MAGIC │  │  │  ┌─────────────────────────────────────────┐   │   │   │
# MAGIC │  │  │  │  GENERATIVE AI (GenAI/LLMs)             │   │   │   │
# MAGIC │  │  │  │  Generates NEW content (text/code/images)│   │   │   │
# MAGIC │  │  │  └─────────────────────────────────────────┘   │   │   │
# MAGIC │  │  └─────────────────────────────────────────────────┘   │   │
# MAGIC │  └─────────────────────────────────────────────────────────┘   │
# MAGIC └─────────────────────────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Layer 1: Rule-Based AI (no learning — hand-coded rules)
# MAGIC This is AI in the broadest sense: rules written by humans that make the computer behave "intelligently."

# COMMAND ----------

# ============================================================================
# LAYER 1: RULE-BASED AI — hand-coded rules, no learning from data
# ============================================================================
# A simple rule-based system for classifying support tickets.
# The "intelligence" comes entirely from rules a human wrote — NOT from data.

def classify_ticket_rules(ticket_text: str) -> str:
    """
    Rule-based AI: classify a support ticket using hand-written keyword rules.
    This is AI (it performs a task requiring human judgment) but NOT ML (no learning).
    """
    text_lower = ticket_text.lower()
    
    # Hard-coded rules — a human wrote every single one of these
    if any(word in text_lower for word in ['bill', 'charge', 'invoice', 'payment', 'refund', 'price']):
        return 'billing'
    elif any(word in text_lower for word in ['crash', 'error', 'bug', 'broken', 'slow', 'login', 'password']):
        return 'technical'
    elif any(word in text_lower for word in ['ship', 'deliver', 'track', 'package', 'address', 'return']):
        return 'shipping'
    else:
        return 'general'

# Test it
test_tickets = [
    "I was charged twice on my credit card for the same order",
    "The app keeps crashing when I try to upload a photo",
    "Where is my package? It was supposed to arrive yesterday",
    "I love your product and want to recommend it to friends",
    "My password reset email never arrived and I can't login"
]

print("=" * 70)
print("RULE-BASED AI — Classification by hand-coded keyword rules")
print("=" * 70)
for ticket in test_tickets:
    category = classify_ticket_rules(ticket)
    print(f"  [{category:10s}] {ticket[:60]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC **Key insight**: The rule-based system *works* for simple cases, but:
# MAGIC - It can't handle nuance ("I need to return something" → shipping or billing?)
# MAGIC - Every new pattern requires a human to write a new rule
# MAGIC - It doesn't improve with more data — it's frozen at whatever rules we wrote
# MAGIC
# MAGIC This is where **Machine Learning** comes in ⬇️

# COMMAND ----------

# MAGIC %md
# MAGIC ### Layer 2: Machine Learning — the system LEARNS from data

# COMMAND ----------

# ============================================================================
# LAYER 2: MACHINE LEARNING — learning patterns from data
# ============================================================================
# Instead of writing rules, we SHOW the model labeled examples and let it
# learn the patterns on its own. Same task (ticket classification), but now
# the model figures out what words/patterns matter, not a human rule-writer.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Generate synthetic labeled training data (what you'd collect from real tickets)
np.random.seed(42)

billing_templates = [
    "I was overcharged on my last bill",
    "Please refund my payment",
    "Why was I charged twice?",
    "The invoice amount is wrong",
    "I need to update my credit card",
    "Cancel my subscription and stop billing",
    "I see an unauthorized charge",
    "When will I receive my refund?",
    "The pricing on the website doesn't match my bill",
    "My promo code didn't apply to the total",
]

technical_templates = [
    "The app crashes every time I open it",
    "I can't log into my account",
    "Getting a 500 error on the dashboard",
    "The page loads very slowly",
    "My password reset isn't working",
    "The upload feature is broken",
    "Getting a blank screen after login",
    "The mobile app freezes on startup",
    "API returning timeout errors",
    "Two-factor authentication code not sending",
]

shipping_templates = [
    "Where is my package?",
    "My order hasn't been delivered yet",
    "I need to change my shipping address",
    "The tracking number isn't working",
    "My package arrived damaged",
    "Can I return this item?",
    "How long does express shipping take?",
    "Wrong item was delivered to me",
    "I need to track my return shipment",
    "The delivery was left at the wrong door",
]

# Create a synthetic dataset by slightly varying the templates
def create_training_data(templates, label, n_per_template=20):
    variations = []
    prefixes = ["Hi, ", "Hello, ", "Hey ", "Please help: ", "Urgent: ", "Question - ", ""]
    suffixes = [" Thanks.", " Please help.", " This is frustrating.", " Appreciate your help.", ""]
    for template in templates:
        for _ in range(n_per_template):
            prefix = np.random.choice(prefixes)
            suffix = np.random.choice(suffixes)
            variations.append({"text": prefix + template + suffix, "label": label})
    return variations

data = []
data.extend(create_training_data(billing_templates, "billing"))
data.extend(create_training_data(technical_templates, "technical"))
data.extend(create_training_data(shipping_templates, "shipping"))

df_tickets = pd.DataFrame(data)
print(f"Synthetic training data: {len(df_tickets)} labeled tickets")
print(f"Label distribution:\n{df_tickets['label'].value_counts()}")

# COMMAND ----------

# Split into train/test, vectorize, and train a classic ML model
X_train, X_test, y_train, y_test = train_test_split(
    df_tickets['text'], df_tickets['label'], test_size=0.2, random_state=42
)

# TF-IDF: convert text to numeric features (this IS feature engineering — step 1 of A3)
vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Logistic Regression classifier (this IS training — step 2 of A3)
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = lr_model.predict(X_test_tfidf)
print("=" * 70)
print("MACHINE LEARNING — Classification learned from labeled data")
print("=" * 70)
print(classification_report(y_test, y_pred))

# Now test on the SAME tickets the rule-based system tried:
print("\nML model on the same test tickets:")
for ticket in test_tickets:
    features = vectorizer.transform([ticket])
    prediction = lr_model.predict(features)[0]
    confidence = lr_model.predict_proba(features).max()
    print(f"  [{prediction:10s}] (conf: {confidence:.2f}) {ticket[:55]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC **Key insight**: The ML model learned *from data* which words indicate which category — no human wrote rules. It can generalize to tickets it's never seen before, and it gets BETTER with more data, unlike the rule-based approach.
# MAGIC
# MAGIC ### Layer 3: Deep Learning & Layer 4: GenAI
# MAGIC
# MAGIC - **Deep Learning** uses neural networks with many layers. On Community Edition, we can demonstrate this with a simple PyTorch model. In practice, this powers image recognition, speech, and modern LLMs.
# MAGIC - **Generative AI** takes deep learning further: instead of just *classifying*, the model *generates* new content (text, code, images).
# MAGIC
# MAGIC We'll explore these hands-on in Notebooks 04–06. For now, the key takeaway:
# MAGIC > **AI ⊃ ML ⊃ Deep Learning ⊃ GenAI** — each layer nested inside the one before it.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 2: Why Do AI Work on a Data Platform? (Handbook A2)
# MAGIC
# MAGIC > *"AI doesn't have an intelligence problem, it has a CONTEXT problem."*
# MAGIC > — Databricks Summit 2026 Keynote
# MAGIC
# MAGIC The pitch: your AI is only as good as the **data** it's grounded in. If that data lives in a governed lakehouse (Delta Lake + Unity Catalog), then keeping AI on the **same** platform means:
# MAGIC - 🔒 **Same governance** for AI and data access
# MAGIC - 🔄 **Auto-sync** between source data and retrieval indexes
# MAGIC - 📊 **Same tracking** for model calls as for SQL queries
# MAGIC - 💰 **Centralized cost control** instead of scattered API keys
# MAGIC
# MAGIC Let's demonstrate this by creating governed Delta tables that our ML models will consume.

# COMMAND ----------

# ============================================================================
# DEMONSTRATING: AI work on a data platform — governed Delta tables as the
# single source of truth for BOTH analytics AND ML training
# ============================================================================

# Generate a rich synthetic e-commerce dataset
np.random.seed(2026)
n_customers = 2000

customers = pd.DataFrame({
    'customer_id': [f'CUST_{i:05d}' for i in range(n_customers)],
    'signup_date': pd.date_range('2023-01-01', periods=n_customers, freq='4H'),
    'region': np.random.choice(['US-East', 'US-West', 'Europe', 'Asia-Pacific'], n_customers, p=[0.35, 0.25, 0.25, 0.15]),
    'plan': np.random.choice(['free', 'basic', 'premium', 'enterprise'], n_customers, p=[0.4, 0.3, 0.2, 0.1]),
})

# Generate transactions
transactions_list = []
for _, cust in customers.iterrows():
    n_orders = np.random.poisson(lam=8) + 1
    for j in range(n_orders):
        transactions_list.append({
            'transaction_id': f'TXN_{len(transactions_list):07d}',
            'customer_id': cust['customer_id'],
            'order_date': cust['signup_date'] + pd.Timedelta(days=np.random.randint(1, 365)),
            'amount': round(np.random.lognormal(mean=3.5, sigma=0.8), 2),
            'product_category': np.random.choice(['electronics', 'clothing', 'home', 'food', 'sports']),
        })

transactions = pd.DataFrame(transactions_list)

# Save as Delta tables — the governed data foundation
spark_customers = spark.createDataFrame(customers)
spark_transactions = spark.createDataFrame(transactions)

spark_customers.write.format("delta").mode("overwrite").saveAsTable("default.customers")
spark_transactions.write.format("delta").mode("overwrite").saveAsTable("default.transactions")

print(f"✅ Created Delta tables:")
print(f"   default.customers     — {spark_customers.count():,} rows")
print(f"   default.transactions  — {spark_transactions.count():,} rows")
print(f"\n   These tables are the GOVERNED data foundation for all ML work below.")
print(f"   On full Databricks: these would be Unity Catalog tables with lineage/audit.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 3: The Complete Classic ML Lifecycle (Handbook A3)
# MAGIC
# MAGIC Now we walk through all 6 steps of the ML lifecycle from A3, end to end:
# MAGIC
# MAGIC | Step | What | Databricks Tool |
# MAGIC |------|------|-----------------|
# MAGIC | 1 | Feature Engineering | Spark + Delta |
# MAGIC | 2 | Training | scikit-learn / XGBoost |
# MAGIC | 3 | Tracking | MLflow Experiments |
# MAGIC | 4 | Registry | MLflow Model Registry |
# MAGIC | 5 | Serving | (Full Databricks: Model Serving) |
# MAGIC | 6 | Monitoring | (Full Databricks: Lakehouse Monitoring) |

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Feature Engineering
# MAGIC *Derive useful input columns from raw data*

# COMMAND ----------

# ============================================================================
# STEP 1: FEATURE ENGINEERING
# "days since last purchase", "total spend", "order count", etc.
# ============================================================================

# Read from governed Delta tables (same tables an analyst would query)
customers_df = spark.table("default.customers")
transactions_df = spark.table("default.transactions")

# Compute features using Spark (this is the DATA ENGINEERING part of ML)
from pyspark.sql.window import Window

# Reference date for feature computation
reference_date = F.lit("2024-06-01").cast("date")

customer_features = (
    transactions_df
    .groupBy("customer_id")
    .agg(
        F.count("*").alias("total_orders"),
        F.sum("amount").alias("total_spend"),
        F.avg("amount").alias("avg_order_value"),
        F.max("order_date").alias("last_order_date"),
        F.min("order_date").alias("first_order_date"),
        F.countDistinct("product_category").alias("unique_categories"),
        F.stddev("amount").alias("spend_volatility"),
    )
    .withColumn("days_since_last_order", 
                F.datediff(reference_date, F.col("last_order_date")))
    .withColumn("customer_tenure_days",
                F.datediff(reference_date, F.col("first_order_date")))
    .withColumn("orders_per_month",
                F.col("total_orders") / (F.col("customer_tenure_days") / 30.0))
)

# Join with customer attributes
features_df = (
    customer_features
    .join(customers_df, on="customer_id", how="inner")
    .drop("signup_date")
)

# Create a synthetic label: "churned" = 1 if no orders in last 90 days + some noise
features_df = features_df.withColumn(
    "churned",
    F.when(F.col("days_since_last_order") > 90, 
           F.when(F.rand(seed=42) > 0.2, 1).otherwise(0))
     .otherwise(
           F.when(F.rand(seed=42) > 0.85, 1).otherwise(0))
)

# Save as a feature table
features_df.write.format("delta").mode("overwrite").saveAsTable("default.customer_features")

print("✅ Feature table created: default.customer_features")
features_df.select(
    "customer_id", "total_orders", "avg_order_value", "days_since_last_order",
    "orders_per_month", "unique_categories", "plan", "region", "churned"
).show(10, truncate=False)

print(f"\nChurn distribution:")
features_df.groupBy("churned").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Steps 2 & 3: Training + MLflow Tracking
# MAGIC *Train multiple models and track EVERYTHING with MLflow*

# COMMAND ----------

# ============================================================================
# STEPS 2 & 3: TRAINING + TRACKING
# Train 3 different models, log EVERYTHING to MLflow for comparison
# ============================================================================

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Pull data to pandas for scikit-learn
pdf = features_df.toPandas()

# Encode categoricals
le_plan = LabelEncoder()
le_region = LabelEncoder()
pdf['plan_encoded'] = le_plan.fit_transform(pdf['plan'])
pdf['region_encoded'] = le_region.fit_transform(pdf['region'])

feature_cols = [
    'total_orders', 'total_spend', 'avg_order_value', 'days_since_last_order',
    'unique_categories', 'spend_volatility', 'customer_tenure_days',
    'orders_per_month', 'plan_encoded', 'region_encoded'
]

X = pdf[feature_cols].fillna(0)
y = pdf['churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {len(X_train)} | Test set: {len(X_test)}")
print(f"Churn rate (train): {y_train.mean():.2%} | Churn rate (test): {y_test.mean():.2%}")

# COMMAND ----------

# Define models to compare (this is the heart of the ML lifecycle — experiment iteration)
models = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {"algorithm": "logistic_regression", "max_iter": 1000, "regularization": "l2"}
    },
    "Random Forest": {
        "model": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "params": {"algorithm": "random_forest", "n_estimators": 100, "max_depth": 10}
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42),
        "params": {"algorithm": "gradient_boosting", "n_estimators": 150, "max_depth": 5, "learning_rate": 0.1}
    },
}

# Train each model and log to MLflow
run_ids = {}
for model_name, config in models.items():
    with mlflow.start_run(run_name=model_name) as run:
        # Log parameters (step 3 of A3)
        for param_name, param_value in config["params"].items():
            mlflow.log_param(param_name, param_value)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("training_rows", len(X_train))
        
        # Train (step 2 of A3)
        model = config["model"]
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Log metrics (step 3 of A3)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, round(metric_value, 4))
        
        # Log the model artifact (step 3 of A3)
        mlflow.sklearn.log_model(model, "model")
        
        run_ids[model_name] = run.info.run_id
        
        print(f"\n{'=' * 50}")
        print(f"  {model_name}")
        print(f"{'=' * 50}")
        for m, v in metrics.items():
            print(f"  {m:15s}: {v:.4f}")
        print(f"  MLflow Run ID: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🔍 Check the MLflow Experiments UI!
# MAGIC
# MAGIC Click the **Experiments** icon (🧪 flask) in the left sidebar to see:
# MAGIC - All 3 runs logged side-by-side
# MAGIC - Parameters (algorithm, hyperparameters)
# MAGIC - Metrics (accuracy, precision, recall, F1, AUC)
# MAGIC - Model artifacts (the saved `.pkl` model files)
# MAGIC
# MAGIC This is the **Tracking** component of MLflow (handbook A4) — every run is browsable,
# MAGIC comparable, and reproducible.

# COMMAND ----------

# ============================================================================
# Compare all models programmatically
# ============================================================================
print("\n" + "=" * 80)
print("  MODEL COMPARISON SUMMARY")
print("=" * 80)
print(f"  {'Model':<25s} {'Accuracy':>10s} {'F1':>10s} {'AUC':>10s}")
print("  " + "-" * 55)

best_model_name = None
best_auc = 0

for model_name, run_id in run_ids.items():
    run_data = mlflow.get_run(run_id).data
    accuracy = run_data.metrics['accuracy']
    f1 = run_data.metrics['f1_score']
    auc = run_data.metrics['roc_auc']
    marker = ""
    if auc > best_auc:
        best_auc = auc
        best_model_name = model_name
        best_run_id = run_id
        marker = " ⭐ BEST"
    print(f"  {model_name:<25s} {accuracy:>10.4f} {f1:>10.4f} {auc:>10.4f}{marker}")

print(f"\n  🏆 Best model by AUC: {best_model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4: Model Registry
# MAGIC *Register the winning model as a versioned asset*
# MAGIC
# MAGIC > **Note**: On CE, we use the workspace-level MLflow Model Registry. On full Databricks,
# MAGIC > this would be the **Unity Catalog Model Registry** with `catalog.schema.model_name` naming
# MAGIC > and aliases instead of stages (see handbook A4 / I4).

# COMMAND ----------

# ============================================================================
# STEP 4: MODEL REGISTRY
# Register the best model as a versioned, named asset
# ============================================================================

model_name_registry = "churn_prediction_model"

# Register the best model
model_uri = f"runs:/{best_run_id}/model"
registered_model = mlflow.register_model(model_uri, model_name_registry)

print(f"✅ Model registered: {model_name_registry}")
print(f"   Version: {registered_model.version}")
print(f"   Source run: {best_run_id}")
print(f"\n   📌 On full Databricks (Unity Catalog Model Registry):")
print(f"      - Name would be: catalog.schema.{model_name_registry}")
print(f"      - Governed by Unity Catalog (same as tables)")
print(f"      - Uses ALIASES ('champion'/'challenger') instead of deprecated Stages")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Steps 5 & 6: Serving & Monitoring (Concepts)
# MAGIC
# MAGIC > **Serving** and **Monitoring** require features not available on Community Edition.
# MAGIC > Here's what they look like on the full platform:

# COMMAND ----------

# ============================================================================
# STEP 5 (CONCEPT): SERVING — deploying a model behind a REST endpoint
# ============================================================================
# On full Databricks, you'd deploy to Model Serving:
#
#   from databricks.sdk import WorkspaceClient
#   w = WorkspaceClient()
#   w.serving_endpoints.create(
#       name="churn-prediction-endpoint",
#       config=EndpointCoreConfigInput(
#           served_models=[ServedModelInput(
#               model_name="catalog.schema.churn_prediction_model",
#               model_version="1",
#               workload_size="Small",
#               scale_to_zero_enabled=True
#           )]
#       )
#   )
#
# Then any app can call it via REST:
#   POST https://<workspace-url>/serving-endpoints/churn-prediction-endpoint/invocations
#   {"dataframe_records": [{"total_orders": 5, "total_spend": 230.50, ...}]}

# On CE, we can simulate serving by loading the model from the registry:
loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name_registry}/{registered_model.version}")

# Simulate a real-time prediction request
sample_customer = X_test.iloc[[0]]
prediction = loaded_model.predict(sample_customer)
print(f"✅ Simulated real-time prediction:")
print(f"   Input features: {dict(sample_customer.iloc[0])}")
print(f"   Prediction: {'WILL CHURN ⚠️' if prediction[0] == 1 else 'Will NOT churn ✅'}")

# COMMAND ----------

# ============================================================================
# STEP 6 (CONCEPT): MONITORING — watching for drift in production
# ============================================================================
# On full Databricks, Lakehouse Monitoring would:
#   - Track feature distributions over time (data drift)
#   - Track prediction distributions (prediction drift)  
#   - Track actual outcomes vs predictions (performance drift)
#   - Alert when metrics drop below a threshold
#
# On CE, we can simulate monitoring by comparing current predictions to a baseline:

# Baseline metrics (from training)
baseline_auc = best_auc

# Simulate "production" data with slight drift
np.random.seed(99)
X_production = X_test.copy()
X_production['total_spend'] = X_production['total_spend'] * np.random.uniform(0.8, 1.3, len(X_production))
X_production['days_since_last_order'] = X_production['days_since_last_order'] + np.random.randint(-10, 30, len(X_production))

y_prod_pred = loaded_model.predict(X_production)
prod_auc = roc_auc_score(y_test, loaded_model.predict(X_production))

drift = abs(prod_auc - baseline_auc) / baseline_auc * 100
print(f"📊 Monitoring Simulation:")
print(f"   Baseline AUC:   {baseline_auc:.4f}")
print(f"   Production AUC: {prod_auc:.4f}")
print(f"   Drift:          {drift:.1f}%")
if drift > 5:
    print(f"   ⚠️  ALERT: Model performance drift exceeds 5% threshold!")
else:
    print(f"   ✅ Performance within acceptable bounds")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # ✅ Key Takeaways
# MAGIC
# MAGIC | Concept | What We Did | Handbook Section |
# MAGIC |---------|------------|------------------|
# MAGIC | Rule-based AI | Hand-coded keyword classifier | A1 |
# MAGIC | Machine Learning | Learned classifier from labeled data | A1, A3 |
# MAGIC | Delta as data foundation | Stored all data in governed Delta tables | A2, B2 |
# MAGIC | Feature Engineering | Derived `avg_order_value`, `days_since_last_order` from raw tables | A3 step 1, A5 |
# MAGIC | Training | Trained 3 models (LR, RF, GBM) | A3 step 2 |
# MAGIC | MLflow Tracking | Logged params, metrics, artifacts for every run | A3 step 3, A4 |
# MAGIC | Model Registry | Registered best model as versioned asset | A3 step 4, A4 |
# MAGIC | Serving (concept) | Simulated loading model for predictions | A3 step 5 |
# MAGIC | Monitoring (concept) | Simulated drift detection | A3 step 6 |
# MAGIC
# MAGIC ### 🔗 Full Platform Mapping
# MAGIC | What we did on CE | Full Databricks equivalent |
# MAGIC |---|---|
# MAGIC | `default.customers` | `catalog.schema.customers` (Unity Catalog) |
# MAGIC | MLflow workspace registry | Unity Catalog Model Registry with aliases |
# MAGIC | Simulated serving | Model Serving endpoint (REST API) |
# MAGIC | Simulated monitoring | Lakehouse Monitoring (automated drift detection) |
# MAGIC
# MAGIC **Next**: Notebook 02 dives deep into MLflow — tracking details, model registry aliases vs stages, and prompt versioning.
