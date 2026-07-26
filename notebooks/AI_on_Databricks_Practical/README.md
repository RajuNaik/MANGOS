<div align="center">

# 🧠 AI on Databricks — Practical Hands-On Notebooks

### Learn Every Topic from the AI on Databricks Handbook with Synthetic Data

**Designed for Databricks Community Edition (Free)**

</div>

---

## 📖 Overview

This collection of **10 notebooks** covers every topic from the *AI on Databricks Complete Handbook (2026 Edition)* with **practical, runnable code** using **synthetic data** — all executable on the **free Databricks Community Edition**.

For features not available on CE (Unity Catalog, Vector Search, AI Functions, etc.), we build **functional open-source equivalents** and clearly explain how they map to the full Databricks platform.

---

## 🚀 Quick Start

### 1. Sign up for Databricks Community Edition (Free)
- Go to [community.cloud.databricks.com](https://community.cloud.databricks.com)
- Sign up with your email — no credit card required

### 2. Import the Notebooks
- In Databricks, click **Workspace** → **Users** → your username
- Right-click → **Import**
- Upload each `.py` notebook file (they're in Databricks notebook source format)

### 3. Create a Cluster
- Go to **Compute** → **Create Cluster**
- Name: `AI-Learning` (or anything you like)
- Runtime: **15.4 LTS ML** (or latest ML runtime available)
  - ⚠️ Use the **ML** runtime — it includes MLflow, scikit-learn, XGBoost, PyTorch pre-installed
- Leave defaults (Community Edition gives you 1 driver node, 15 GB memory)
- Click **Create Cluster** and wait for it to start

### 4. Run Notebooks in Order
- Start with `01_Foundations_ML_Lifecycle.py` and work through sequentially
- Each notebook installs its own additional dependencies via `%pip install`
- Each notebook generates its own synthetic data — no external data needed

---

## 📚 Notebook Map

| # | Notebook | Handbook Sections | Key Topics |
|---|----------|-------------------|------------|
| 01 | Foundations & ML Lifecycle | A1–A3 | AI/ML/DL hierarchy, feature engineering, train/evaluate |
| 02 | MLflow Tracking & Registry | A4, I4, I16 | Experiment tracking, model registry, aliases, prompt versioning |
| 03 | Feature Store & AutoML | A5, I1–I3 | Feature tables, point-in-time joins, AutoML baseline |
| 04 | Embeddings, RAG & Vector Search | A6–A8, B1–B3, B5, I5 | Embeddings, chunking, FAISS index, RAG pipeline, hybrid search |
| 05 | AI Functions & Structured Outputs | B4, I6, I9 | Classification/extraction UDFs, JSON-schema outputs |
| 06 | Agents, Tools & MCP | A9, D1–D5 | ReAct loop, tool calling, multi-agent supervisor |
| 07 | MLflow 3 for GenAI Evaluation | E1, E2, I17 | Tracing, LLM judges, retrieval vs generation metrics |
| 08 | Governance & Cost Control | A10, E3–E5, I15 | AI Gateway simulation, spend caps, routing, caching |
| 09 | Batch & Streaming AI Pipelines | C1–C5, I7, I8 | Batch inference, streaming enrichment, idempotent processing |
| 10 | LLM Internals & Responsible AI | I12–I14, I18–I20 | Temperature, context window, hallucination, bias, attention |

---

## 🔑 API Key (Optional)

Some notebooks can use an **external LLM API** (OpenAI, Anthropic, or Google) for richer demonstrations. This is **completely optional** — every notebook has a **fully local, free fallback** using small open-source models.

If you have a key, set it as a Databricks secret or environment variable:
```python
# Option 1: Environment variable (set in cluster config)
# Cluster → Edit → Advanced → Spark Config:
# spark.databricks.passthrough.enabled true
# Environment Variables → OPENAI_API_KEY=sk-...

# Option 2: In-notebook (less secure, for learning only)
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

---

## ⚙️ Technical Notes

- **Runtime**: Databricks ML Runtime 15.4 LTS or later recommended
- **Compute**: Community Edition single-node cluster (15 GB RAM) is sufficient
- **Libraries**: Each notebook installs what it needs via `%pip install`
- **Data**: All synthetic — generated fresh in each notebook, stored as Delta tables
- **Format**: Databricks notebook source (`.py`) — directly importable

---

## 🗺️ Databricks Full Platform Mapping

Each notebook includes a **"Full Platform Mapping"** section explaining how the open-source approach maps to the managed Databricks feature:

| CE Approach | Full Databricks Equivalent |
|---|---|
| FAISS index | Mosaic AI Vector Search (auto-sync from Delta) |
| `transformers` pipeline UDF | `ai_classify()`, `ai_query()` SQL AI Functions |
| Local embedding model | Foundation Model APIs (pay-per-token) |
| Python proxy class | Unity AI Gateway (managed governance) |
| `default` database | Unity Catalog (`catalog.schema.table`) |
| MLflow OSS registry | Unity Catalog Model Registry (aliases, governance) |

---

**Happy Learning! 🚀**
