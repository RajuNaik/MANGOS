# Databricks notebook source

# MAGIC %md
# MAGIC # 🔬 Notebook 02 — MLflow Deep Dive: Tracking, Registry & Aliases
# MAGIC
# MAGIC **Handbook Sections Covered**: A4 (MLflow classic), I4 (Aliases vs deprecated Stages), I16 (Prompt Registry)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Learning Objectives
# MAGIC 1. **MLflow Tracking** — log parameters, metrics, artifacts, tags; compare runs; autologging
# MAGIC 2. **Model Registry** — register models, create versions, manage lifecycle
# MAGIC 3. **Aliases vs Stages** — understand the modern alias workflow (`@champion`/`@challenger`) vs deprecated `Staging`/`Production`
# MAGIC 4. **Prompt Registry** — version and manage prompt templates (MLflow 3 concept)
# MAGIC 5. **Model loading patterns** — load by run ID, version number, alias

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Setup

# COMMAND ----------

%pip install -q xgboost

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import json
import warnings
warnings.filterwarnings('ignore')

client = MlflowClient()

# Set experiment
experiment_name = "/Users/{}/AI_Handbook_02_MLflow_DeepDive".format(
    spark.sql("SELECT current_user()").collect()[0][0]
)
mlflow.set_experiment(experiment_name)
print(f"✅ Experiment: {experiment_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 1: MLflow Tracking Deep Dive (Handbook A4)
# MAGIC
# MAGIC MLflow Tracking logs **everything** about an experiment:
# MAGIC - **Parameters**: hyperparameters, config choices
# MAGIC - **Metrics**: accuracy, loss, F1 — can be logged at multiple steps
# MAGIC - **Artifacts**: model files, plots, data samples
# MAGIC - **Tags**: metadata (team, dataset version, purpose)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Generate Synthetic Data

# COMMAND ----------

# ============================================================================
# SYNTHETIC DATA: Credit card fraud detection
# ============================================================================
np.random.seed(2026)

X, y = make_classification(
    n_samples=5000,
    n_features=15,
    n_informative=10,
    n_redundant=3,
    n_classes=2,
    weights=[0.95, 0.05],  # 5% fraud rate — realistic class imbalance
    random_state=2026
)

feature_names = [
    'transaction_amount', 'merchant_category', 'time_of_day', 'day_of_week',
    'distance_from_home', 'distance_from_last_txn', 'ratio_to_median',
    'frequency_24h', 'frequency_7d', 'is_international',
    'card_present', 'avg_txn_amount_30d', 'max_txn_amount_30d',
    'redundant_1', 'redundant_2'
]

df = pd.DataFrame(X, columns=feature_names)
df['is_fraud'] = y

# Save as Delta
spark_df = spark.createDataFrame(df)
spark_df.write.format("delta").mode("overwrite").saveAsTable("default.fraud_data")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"✅ Fraud detection dataset: {len(df)} transactions")
print(f"   Fraud rate: {y.mean():.1%}")
print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Manual Tracking — Full Control

# COMMAND ----------

# ============================================================================
# MANUAL TRACKING: explicit control over what gets logged
# This is the pattern shown in handbook A4
# ============================================================================

with mlflow.start_run(run_name="Manual_Tracking_Demo") as run:
    # ---- LOG PARAMETERS ----
    mlflow.log_param("algorithm", "random_forest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 8)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("dataset_version", "v1.0")
    mlflow.log_param("fraud_rate", f"{y_train.mean():.3f}")
    
    # ---- TRAIN ----
    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight='balanced', random_state=42
    )
    model.fit(X_train, y_train)
    
    # ---- LOG METRICS ----
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_proba))
    
    # Log step metrics (e.g., simulating training epochs)
    for epoch in range(1, 11):
        simulated_loss = 0.5 * np.exp(-0.3 * epoch) + np.random.normal(0, 0.01)
        mlflow.log_metric("training_loss", simulated_loss, step=epoch)
    
    # ---- LOG TAGS ----
    mlflow.set_tag("team", "fraud-detection")
    mlflow.set_tag("purpose", "baseline_experiment")
    mlflow.set_tag("data_source", "default.fraud_data")
    
    # ---- LOG ARTIFACTS ----
    # Log feature importance as a JSON artifact
    importance = dict(zip(feature_names, model.feature_importances_.tolist()))
    importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    with open("/tmp/feature_importance.json", "w") as f:
        json.dump(importance_sorted, f, indent=2)
    mlflow.log_artifact("/tmp/feature_importance.json")
    
    # Log a text summary
    summary = f"""Fraud Detection Model Summary
    ============================
    Algorithm: Random Forest
    Training samples: {len(X_train)}
    Test samples: {len(X_test)}
    Fraud rate: {y_train.mean():.1%}
    Accuracy: {accuracy_score(y_test, y_pred):.4f}
    F1 Score: {f1_score(y_test, y_pred):.4f}
    ROC AUC:  {roc_auc_score(y_test, y_proba):.4f}
    """
    with open("/tmp/model_summary.txt", "w") as f:
        f.write(summary)
    mlflow.log_artifact("/tmp/model_summary.txt")
    
    # ---- LOG MODEL ----
    mlflow.sklearn.log_model(model, "model")
    
    manual_run_id = run.info.run_id
    print(f"✅ Manual tracking run complete!")
    print(f"   Run ID: {manual_run_id}")
    print(f"\n   Logged: 6 params, 3 metrics + 10 step metrics, 2 tags, 2 artifacts, 1 model")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 Autologging — Zero-Code Tracking

# COMMAND ----------

# ============================================================================
# AUTOLOGGING: MLflow automatically logs everything — zero manual code
# ============================================================================

# Enable autologging for sklearn
mlflow.sklearn.autolog(log_models=True, log_input_examples=True)

with mlflow.start_run(run_name="Autolog_GradientBoosting") as run:
    model_gb = GradientBoostingClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42
    )
    model_gb.fit(X_train, y_train)
    
    # Autolog automatically captured: all hyperparameters, training metrics,
    # the model artifact, input example, and model signature
    autolog_run_id = run.info.run_id
    print(f"✅ Autologging run complete!")
    print(f"   Run ID: {autolog_run_id}")
    print(f"   MLflow automatically logged everything — no manual log_param/log_metric calls!")

# Turn off autologging for subsequent manual experiments
mlflow.sklearn.autolog(disable=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.4 Comparing Runs Programmatically

# COMMAND ----------

# ============================================================================
# COMPARE RUNS — the same comparison you'd do in the MLflow UI, but in code
# ============================================================================

experiment = mlflow.get_experiment_by_name(experiment_name)
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.roc_auc DESC"],
)

print("=" * 90)
print("  ALL EXPERIMENT RUNS — sorted by AUC (descending)")
print("=" * 90)

display_cols = ['run_id', 'tags.mlflow.runName', 'metrics.accuracy', 'metrics.f1_score', 'metrics.roc_auc']
available_cols = [c for c in display_cols if c in runs.columns]
print(runs[available_cols].to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🔍 Explore in the MLflow UI!
# MAGIC
# MAGIC Click the **Experiments** icon (🧪) in the sidebar → select `AI_Handbook_02_MLflow_DeepDive`:
# MAGIC - **Compare** tab: side-by-side metric comparison
# MAGIC - **Chart** view: visualize metrics across runs
# MAGIC - Click a run → **Artifacts** tab: see saved model, feature_importance.json, summary
# MAGIC - Click a run → **Metrics** tab: see the step-by-step training_loss curve

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 2: Hyperparameter Sweep with MLflow (Practical A4 Extension)
# MAGIC
# MAGIC A real ML workflow involves trying many hyperparameter combinations. Let's log them all.

# COMMAND ----------

# ============================================================================
# HYPERPARAMETER SWEEP — log every combination to MLflow
# ============================================================================

from itertools import product

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 8],
    'class_weight': ['balanced', None],
}

# Generate all combinations
all_combos = list(product(
    param_grid['n_estimators'],
    param_grid['max_depth'],
    param_grid['class_weight'],
))

print(f"Running {len(all_combos)} experiments...")

sweep_results = []
for i, (n_est, depth, cw) in enumerate(all_combos):
    with mlflow.start_run(run_name=f"sweep_{i+1:02d}_ne{n_est}_d{depth}") as run:
        mlflow.log_param("n_estimators", n_est)
        mlflow.log_param("max_depth", depth)
        mlflow.log_param("class_weight", str(cw))
        mlflow.set_tag("experiment_type", "hyperparameter_sweep")
        
        model = RandomForestClassifier(
            n_estimators=n_est, max_depth=depth, class_weight=cw, random_state=42
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")
        
        sweep_results.append({
            'run_id': run.info.run_id,
            'n_estimators': n_est,
            'max_depth': depth,
            'class_weight': str(cw),
            'roc_auc': auc,
            'f1_score': f1,
        })

sweep_df = pd.DataFrame(sweep_results).sort_values('roc_auc', ascending=False)
print(f"\n{'=' * 80}")
print(f"  HYPERPARAMETER SWEEP RESULTS (top 5)")
print(f"{'=' * 80}")
print(sweep_df.head().to_string(index=False))

best_sweep = sweep_df.iloc[0]
best_sweep_run_id = best_sweep['run_id']
print(f"\n🏆 Best: n_est={best_sweep['n_estimators']}, depth={best_sweep['max_depth']}, "
      f"cw={best_sweep['class_weight']} → AUC={best_sweep['roc_auc']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 3: Model Registry — Versions & Aliases (Handbook A4, I4)
# MAGIC
# MAGIC > **Critical 2026 update (I4)**: The OLD workspace-level registry used fixed **Stages**
# MAGIC > (`Staging` → `Production` → `Archived`). The CURRENT Unity Catalog registry uses
# MAGIC > **Aliases** — flexible, custom-named pointers. Many tutorials still show the old stages.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Register Multiple Model Versions

# COMMAND ----------

# ============================================================================
# MODEL REGISTRY: Register the best model from the sweep
# ============================================================================

registry_model_name = "fraud_detection_model"

# Register version 1 — our best sweep model
model_uri_v1 = f"runs:/{best_sweep_run_id}/model"
mv1 = mlflow.register_model(model_uri_v1, registry_model_name)
print(f"✅ Registered {registry_model_name} version {mv1.version}")
print(f"   Source: sweep best (AUC={best_sweep['roc_auc']:.4f})")

# COMMAND ----------

# Now train a different model architecture and register it as version 2
with mlflow.start_run(run_name="GBM_for_registry_v2") as run:
    from xgboost import XGBClassifier
    
    mlflow.log_param("algorithm", "xgboost")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.1)
    
    model_xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42, eval_metric='logloss'
    )
    model_xgb.fit(X_train, y_train)
    
    y_pred = model_xgb.predict(X_test)
    y_proba = model_xgb.predict_proba(X_test)[:, 1]
    
    auc_v2 = roc_auc_score(y_test, y_proba)
    f1_v2 = f1_score(y_test, y_pred)
    
    mlflow.log_metric("roc_auc", auc_v2)
    mlflow.log_metric("f1_score", f1_v2)
    mlflow.sklearn.log_model(model_xgb, "model")
    
    v2_run_id = run.info.run_id

mv2 = mlflow.register_model(f"runs:/{v2_run_id}/model", registry_model_name)
print(f"✅ Registered {registry_model_name} version {mv2.version}")
print(f"   Source: XGBoost (AUC={auc_v2:.4f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Aliases vs Deprecated Stages (Handbook I4)
# MAGIC
# MAGIC > **OLD (deprecated)**: `client.transition_model_version_stage("model", 1, "Production")`
# MAGIC > — fixed enum: `None → Staging → Production → Archived`
# MAGIC >
# MAGIC > **NEW (current)**: `client.set_registered_model_alias("model", "champion", version=1)`
# MAGIC > — flexible, custom-named pointers you define yourself
# MAGIC
# MAGIC > ⚠️ **Interview trap**: Many pre-2024 tutorials still reference the old stages.
# MAGIC > Knowing aliases are current is exactly the kind of specificity senior interviews check for.

# COMMAND ----------

# ============================================================================
# ALIASES: The modern way to manage model lifecycle (I4)
# ============================================================================

# NOTE: On CE workspace registry, aliases may use the older API.
# Below shows the CONCEPT exactly as documented for Unity Catalog registry:

# CONCEPT (Unity Catalog registry — the current way):
#   client.set_registered_model_alias("catalog.schema.fraud_model", "champion", version=1)
#   client.set_registered_model_alias("catalog.schema.fraud_model", "challenger", version=2)
#   model = mlflow.pyfunc.load_model("models:/catalog.schema.fraud_model@champion")

# On CE, we demonstrate the workflow using tags to simulate aliases:
client.set_model_version_tag(registry_model_name, mv1.version, "alias", "champion")
client.set_model_version_tag(registry_model_name, mv2.version, "alias", "challenger")

# Add descriptive tags
client.set_model_version_tag(registry_model_name, mv1.version, "validation_auc", str(round(best_sweep['roc_auc'], 4)))
client.set_model_version_tag(registry_model_name, mv2.version, "validation_auc", str(round(auc_v2, 4)))
client.set_model_version_tag(registry_model_name, mv2.version, "approved_by", "data_science_lead")

print("=" * 70)
print("  MODEL REGISTRY STATUS")
print("=" * 70)
print(f"\n  Model: {registry_model_name}")
print(f"\n  Version {mv1.version}:")
print(f"    Alias:  champion (current production model)")
print(f"    AUC:    {best_sweep['roc_auc']:.4f}")
print(f"    Algo:   Random Forest (best from sweep)")
print(f"\n  Version {mv2.version}:")
print(f"    Alias:  challenger (being evaluated)")
print(f"    AUC:    {auc_v2:.4f}")
print(f"    Algo:   XGBoost")

# COMMAND ----------

# ============================================================================
# LOADING MODELS: Different patterns for different use cases
# ============================================================================

# Pattern 1: Load by specific version number
model_v1 = mlflow.pyfunc.load_model(f"models:/{registry_model_name}/{mv1.version}")
print(f"✅ Loaded by version: models:/{registry_model_name}/{mv1.version}")

# Pattern 2: Load by run ID (useful for debugging specific experiments)
model_from_run = mlflow.pyfunc.load_model(f"runs:/{best_sweep_run_id}/model")
print(f"✅ Loaded by run ID: runs:/{best_sweep_run_id}/model")

# Pattern 3 (CONCEPT — Unity Catalog): Load by alias
# model_champion = mlflow.pyfunc.load_model(f"models:/catalog.schema.fraud_model@champion")
# This is the PREFERRED production pattern: your serving code says "@champion",
# and promoting a new model version to production = moving the "champion" alias
# to the new version. Instant, atomic, no code change needed.

print(f"\n📌 On full Databricks with Unity Catalog:")
print(f"   model = mlflow.pyfunc.load_model('models:/catalog.schema.fraud_model@champion')")
print(f"   → Promotion = moving the alias, not changing code")
print(f"   → Rollback  = moving the alias back, instant")

# COMMAND ----------

# ============================================================================
# PROMOTE: Demonstrate the alias-based promotion workflow
# ============================================================================

# Scenario: XGBoost (v2) outperforms RF (v1), promote it to champion
if auc_v2 > best_sweep['roc_auc']:
    print("🔄 XGBoost (v2) has higher AUC — promoting to champion!")
    
    # Move aliases (on Unity Catalog: set_registered_model_alias)
    client.set_model_version_tag(registry_model_name, mv2.version, "alias", "champion")
    client.set_model_version_tag(registry_model_name, mv1.version, "alias", "previous_champion")
    
    print(f"   ✅ Version {mv2.version} is now 'champion'")
    print(f"   ✅ Version {mv1.version} is now 'previous_champion'")
    print(f"\n   On Unity Catalog, this would be:")
    print(f"   client.set_registered_model_alias('{registry_model_name}', 'champion', version={mv2.version})")
    print(f"   # The @champion alias now points to version {mv2.version}")
    print(f"   # No downstream code changes needed — code still says @champion")
else:
    print("❌ XGBoost (v2) did NOT outperform — keeping current champion")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 4: Prompt Registry (Handbook I16)
# MAGIC
# MAGIC > **MLflow 3 Concept**: Prompts registered as first-class, versioned, governed objects —
# MAGIC > not hardcoded strings buried in application code.
# MAGIC >
# MAGIC > On full Databricks:
# MAGIC > ```python
# MAGIC > mlflow.genai.register_prompt(
# MAGIC >     name="catalog.schema.support_classifier_prompt",
# MAGIC >     template="Classify this ticket: {{ticket_text}}"
# MAGIC > )
# MAGIC > prompt = mlflow.genai.load_prompt("catalog.schema.support_classifier_prompt", version=3)
# MAGIC > ```
# MAGIC >
# MAGIC > On CE, we simulate this with MLflow artifacts + a custom versioning pattern.

# COMMAND ----------

# ============================================================================
# PROMPT REGISTRY: Versioned, tracked prompt templates
# ============================================================================

class PromptRegistry:
    """
    Simulates MLflow 3's Prompt Registry on Community Edition.
    On full Databricks, use: mlflow.genai.register_prompt() / load_prompt()
    """
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
    
    def register_prompt(self, name: str, template: str, description: str = "", tags: dict = None):
        """Register a new prompt version."""
        with mlflow.start_run(run_name=f"prompt_{name}") as run:
            mlflow.log_param("prompt_name", name)
            mlflow.log_param("prompt_type", "template")
            mlflow.set_tag("prompt_name", name)
            mlflow.set_tag("prompt_description", description)
            
            if tags:
                for k, v in tags.items():
                    mlflow.set_tag(k, v)
            
            # Log the prompt template as an artifact
            prompt_data = {
                "name": name,
                "template": template,
                "description": description,
                "variables": [v.strip("{}") for v in template.split("{{") if "}}" in v],
            }
            with open("/tmp/prompt.json", "w") as f:
                json.dump(prompt_data, f, indent=2)
            mlflow.log_artifact("/tmp/prompt.json")
            
            # Also log template directly as param for easy searching
            mlflow.log_param("template_preview", template[:250])
            
            print(f"✅ Registered prompt '{name}' (run_id: {run.info.run_id})")
            return run.info.run_id
    
    def load_prompt(self, name: str, version: int = None):
        """Load a prompt by name (optionally a specific version)."""
        runs = mlflow.search_runs(
            filter_string=f"tags.prompt_name = '{name}'",
            order_by=["start_time DESC"],
        )
        if runs.empty:
            raise ValueError(f"No prompt found with name '{name}'")
        
        if version and version <= len(runs):
            run_id = runs.iloc[len(runs) - version]['run_id']
        else:
            run_id = runs.iloc[0]['run_id']  # latest
        
        # Download and parse the artifact
        artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="prompt.json")
        with open(artifact_path) as f:
            return json.load(f)

# Create a prompt registry
prompt_registry = PromptRegistry(experiment_name)

# Register Version 1 of our classification prompt
prompt_registry.register_prompt(
    name="ticket_classifier",
    template="Classify the following support ticket into one of these categories: {{categories}}.\n\nTicket: {{ticket_text}}\n\nCategory:",
    description="Zero-shot ticket classification prompt",
    tags={"team": "support-ml", "status": "active"}
)

# Register Version 2 with an improved prompt
prompt_registry.register_prompt(
    name="ticket_classifier",
    template="You are a customer support expert. Classify the ticket below into EXACTLY ONE category from: {{categories}}.\n\nThink step by step:\n1. Identify the main issue\n2. Match to the most relevant category\n3. Output ONLY the category name\n\nTicket: {{ticket_text}}\n\nCategory:",
    description="Improved chain-of-thought ticket classification prompt",
    tags={"team": "support-ml", "status": "active", "improvement": "chain_of_thought"}
)

# COMMAND ----------

# Load and compare prompt versions
prompt_v1 = prompt_registry.load_prompt("ticket_classifier", version=1)
prompt_v2 = prompt_registry.load_prompt("ticket_classifier")  # latest

print("=" * 70)
print("  PROMPT REGISTRY — Version Comparison")
print("=" * 70)
print(f"\n📝 Version 1:")
print(f"   {prompt_v1['template'][:100]}...")
print(f"   Variables: {prompt_v1['variables']}")
print(f"\n📝 Version 2 (latest):")
print(f"   {prompt_v2['template'][:100]}...")
print(f"   Variables: {prompt_v2['variables']}")
print(f"\n📌 On full Databricks (MLflow 3 + Unity Catalog):")
print(f"   prompt = mlflow.genai.load_prompt('catalog.schema.ticket_classifier', version=2)")
print(f"   → Versioned, governed, auditable — like a schema migration for prompts")

# COMMAND ----------

# ============================================================================
# USE the registered prompt (template rendering)
# ============================================================================

def render_prompt(prompt_template: dict, **kwargs) -> str:
    """Render a prompt template with provided variables."""
    text = prompt_template['template']
    for var in prompt_template['variables']:
        if var in kwargs:
            text = text.replace("{{" + var + "}}", str(kwargs[var]))
    return text

# Render the latest prompt with actual values
rendered = render_prompt(
    prompt_v2,
    categories="billing, technical, shipping, general",
    ticket_text="I was charged twice and need a refund immediately"
)

print("=" * 70)
print("  RENDERED PROMPT (ready to send to an LLM)")
print("=" * 70)
print(rendered)
print(f"\n💡 This rendered prompt would be sent to a Model Serving endpoint (C2)")
print(f"   or used with ai_query() (B4) on full Databricks.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 5: Advanced Tracking Patterns

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 Nested Runs (Parent/Child)

# COMMAND ----------

# ============================================================================
# NESTED RUNS: Organize complex experiments (e.g., cross-validation)
# ============================================================================
from sklearn.model_selection import StratifiedKFold

with mlflow.start_run(run_name="CrossValidation_5Fold") as parent_run:
    mlflow.log_param("algorithm", "random_forest")
    mlflow.log_param("n_folds", 5)
    mlflow.set_tag("experiment_type", "cross_validation")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        with mlflow.start_run(run_name=f"fold_{fold_idx+1}", nested=True) as child_run:
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
            model.fit(X_fold_train, y_fold_train)
            
            y_val_pred = model.predict(X_fold_val)
            y_val_proba = model.predict_proba(X_fold_val)[:, 1]
            
            fold_auc = roc_auc_score(y_fold_val, y_val_proba)
            fold_f1 = f1_score(y_fold_val, y_val_pred)
            
            mlflow.log_metric("roc_auc", fold_auc)
            mlflow.log_metric("f1_score", fold_f1)
            mlflow.log_param("fold", fold_idx + 1)
            
            fold_metrics.append({'fold': fold_idx+1, 'auc': fold_auc, 'f1': fold_f1})
    
    # Log aggregate metrics on the parent run
    avg_auc = np.mean([m['auc'] for m in fold_metrics])
    std_auc = np.std([m['auc'] for m in fold_metrics])
    mlflow.log_metric("avg_roc_auc", avg_auc)
    mlflow.log_metric("std_roc_auc", std_auc)
    
    print(f"✅ Cross-validation complete (5 folds)")
    print(f"   Average AUC: {avg_auc:.4f} ± {std_auc:.4f}")
    print(f"\n   Parent run: {parent_run.info.run_id}")
    print(f"   → Expand in MLflow UI to see individual fold results")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Log a Custom Artifact: Confusion Matrix

# COMMAND ----------

# ============================================================================
# LOG CUSTOM ARTIFACTS: Save analysis outputs alongside the model
# ============================================================================
from sklearn.metrics import confusion_matrix

with mlflow.start_run(run_name="Model_With_Analysis") as run:
    model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_proba))
    mlflow.sklearn.log_model(model, "model")
    
    # Log confusion matrix as artifact
    cm = confusion_matrix(y_test, y_pred)
    cm_report = f"""Confusion Matrix
    ================
    Predicted:    Not Fraud | Fraud
    Actual:
      Not Fraud:  {cm[0][0]:>8d} | {cm[0][1]:>5d}
      Fraud:      {cm[1][0]:>8d} | {cm[1][1]:>5d}
    
    True Positives:  {cm[1][1]} (correctly caught fraud)
    False Positives: {cm[0][1]} (legitimate flagged as fraud)  
    False Negatives: {cm[1][0]} (missed fraud ⚠️)
    True Negatives:  {cm[0][0]} (correctly cleared)
    """
    with open("/tmp/confusion_matrix.txt", "w") as f:
        f.write(cm_report)
    mlflow.log_artifact("/tmp/confusion_matrix.txt")
    
    print(cm_report)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # ✅ Key Takeaways
# MAGIC
# MAGIC | Concept | What We Did | Handbook Section |
# MAGIC |---------|------------|------------------|
# MAGIC | MLflow Tracking | Logged params, metrics, artifacts, tags | A4 |
# MAGIC | Autologging | Zero-code tracking with `mlflow.sklearn.autolog()` | A4 |
# MAGIC | Hyperparameter sweep | Logged 18 runs, found best combo | A4 |
# MAGIC | Model Registry | Registered 2 versions of fraud model | A4, I4 |
# MAGIC | Aliases vs Stages | Demonstrated `@champion`/`@challenger` pattern | I4 |
# MAGIC | Model loading | By version, run ID, and (concept) alias | I4 |
# MAGIC | Prompt Registry | Versioned prompt templates with artifacts | I16 |
# MAGIC | Nested runs | Cross-validation with parent/child structure | A4 |
# MAGIC
# MAGIC ### 🔗 Full Platform Mapping
# MAGIC | CE Approach | Full Databricks |
# MAGIC |---|---|
# MAGIC | Workspace MLflow registry | Unity Catalog Model Registry (`catalog.schema.model`) |
# MAGIC | Tags simulating aliases | `client.set_registered_model_alias()` — first-class aliases |
# MAGIC | Prompt as MLflow artifact | `mlflow.genai.register_prompt()` — first-class prompt objects |
# MAGIC | `models:/{name}/{version}` | `models:/catalog.schema.model@champion` |
# MAGIC
# MAGIC **Next**: Notebook 03 covers Feature Store and AutoML — automating feature engineering and model selection.
