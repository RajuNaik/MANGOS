# 🚀 Databricks & PySpark Data Engineering — Complete Enterprise Practical Course (2026 Edition)

Welcome to the **Databricks & PySpark Data Engineering Enterprise Practical Course**. This repository is built strictly according to the syllabus in `Databricks_COMPLETE_HANDBOOK_2026.txt`, covering every section (Parts A through K) through fully executable PySpark and Spark SQL code.

The entire curriculum runs on **Databricks Community Edition (CE)** using free, built-in features and open-source capabilities.

---

## 📂 Repository Structure

```text
notebooks/
└── Databricks_DE_Practical/
    ├── README.md                                # Setup guide, CE mappings, and curriculum overview
    ├── 01_Foundations_Ingestion_Schemas.py      # Part A, B1–B6, B11, B12, B19, B20
    ├── 02_Core_Transformations_ComplexTypes.py  # Part B7–B10, B13–B18, B21–B24, B26
    ├── 03_Spark_Internals_Catalyst_Execution.py # Part A5–A8, C6, C9, C10, C12, C14, J1–J6, K1–K5
    ├── 04_Performance_Tuning_Skew_Mitigation.py # Part C1–C5, C7, C8, C11, C13, C15–C18, K6–K10
    ├── 05_Delta_Lake_Core_Deep_Dive.py          # Part D1–D27, K11–K15
    ├── 06_Structured_Streaming_Realtime.py       # Part F5–F15, J13–J18, K16–K20
    ├── 07_Medallion_Architecture_DataMesh.py    # Part G1–G15, J7–J12
    ├── 08_Lakeflow_Pipelines_Orchestration.py   # Part F1–F4, F16–F20, B25, H1–H5
    ├── 09_Unity_Catalog_Governance_Security.py  # Part E1–E14, J19–J25
    └── 10_Interview_Mastery_Scenarios.py        # Part J1–J25, H1–H5, K1–K20
```

---

## 🏛️ Business Domain & Synthetic Dataset Architecture

All notebooks operate on a Fortune 500 **Omnichannel Retail & Supply Chain** business domain:

- **`customers`**: Customer profiles, loyalty tiers, signup dates, and regional metadata.
- **`products`**: Product catalog, categories, pricing, supplier metadata.
- **`orders`**: Transactional headers, payment methods, order timestamps, status.
- **`order_items`**: Line items, quantity, unit price, discounts.
- **`clickstream`**: High-volume web/app user interaction logs (raw JSON/unstructured).
- **`inventory_iot`**: Real-time IoT warehouse sensor readings (streaming data).

---

## 💻 Databricks Community Edition (CE) Feature Mapping

| Enterprise Feature | CE Limitation | Practical CE Solution in Notebooks |
|---|---|---|
| Unity Catalog (`catalog.schema.table`) | Single metastore (`hive_metastore`) | Simulated 3-level namespace via schema conventions (`prod_catalog_sales_schema`) |
| Lakeflow Pipelines (DLT) | DLT UI not available on CE | Code-first DLT expectation framework (`@dlt.table` syntax commented + executable Spark checks) |
| Lakeflow Connect (Managed CDC) | External DB connectors limited | Delta Change Data Feed (CDF) + `MERGE INTO` CDC implementation |
| Secrets Management (`dbutils.secrets`) | Key Vault integration limited | Simulated secret retrieval with fallback widget parameters |
| Multi-node Clusters | Single driver/worker node | `spark.conf.set()` configuration demonstrations & conceptual architecture notes |

---

## 🚀 Getting Started

1. Import the `.py` files from `notebooks/Databricks_DE_Practical/` into your **Databricks Community Edition** workspace.
2. Attach any standard DBR 14.3+ or DBR 15.4 LTS cluster.
3. Run **Notebook 01** first to initialize the synthetic retail database and table schema foundation.
4. Execute notebooks sequentially (01 through 10) for a structured learning experience!
