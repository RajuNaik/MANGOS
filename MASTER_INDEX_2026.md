# MANGOS Interview Prep Library — Master Index (2026)
### MAANG / Top-Tier Tech — Data & AI Engineer Track

This index ties together every handbook in the library: what each one covers,
what order to read them in, how they reference each other, and where to find
a specific topic fast. Treat this file as the front door to the repo.

---

## 1. Full Inventory

### Core Data Engineering
| File | Covers |
|---|---|
| `Databricks_COMPLETE_HANDBOOK_2026.txt` | Databricks platform, medallion architecture, Delta Lake, Lakeflow, Unity Catalog, 2026 Summit updates (Agent Bricks, Lakebase, Unity AI Gateway) |
| `Databricks_CHEAT_SHEET_2026.txt` | Quick-reference companion to the above |
| `Spark_Internals_Deep_Dive_COMPLETE_HANDBOOK_2026.txt` | Catalyst optimizer, Tungsten/codegen, shuffle, joins, AQE, memory management, Delta Lake internals |
| `Streaming_and_Kafka_COMPLETE_HANDBOOK_2026.txt` | Kafka fundamentals, delivery semantics, ordering, schema registry, Structured Streaming, watermarking, Kafka Streams vs. Flink |
| `Data_Modeling_COMPLETE_HANDBOOK_2026.txt` | Normalization/denormalization, star/snowflake/Data Vault/OBT, fact table grain, all SCD types with SQL |
| `Advanced_SQL_Scenarios_COMPLETE_HANDBOOK...txt` | Complex SQL patterns, window functions, gaps-and-islands, advanced query scenarios |
| `Coding_and_DSA_for_Data_Engineers_COMPLETE_HANDBOOK_2026.txt` | LeetCode-style coding rounds with a DE flavor, PySpark-specific coding questions |

### System Design & Cloud
| File | Covers |
|---|---|
| `System_Design_for_Data_Engineers_COMPLETE_HANDBOOK_2026.txt` | RADIO-DE interview framework, capacity estimation, 9 worked design problems, company-specific patterns |
| `Cloud_Fundamentals_Azure_COMPLETE_HANDBOOK_2026.txt` | Entra ID, ADLS Gen2, networking, Databricks-on-Azure architecture |
| `Cloud_Fundamentals_AWS_COMPLETE_HANDBOOK_2026.txt` | IAM, S3, VPC, Databricks-on-AWS architecture, AWS↔Azure↔GCP equivalence |
| `Cloud_Fundamentals_Azure_for_Data_Engineers_COMPLETE_HANDBOOK_2026.txt` | ⚠️ Earlier/duplicate draft of the Azure file above — see note in Section 5 |

### Python
| File | Covers |
|---|---|
| `Python_for_Data_Engineers_COMPLETE_HANDBOOK...txt` | Production Python patterns for DE pipelines |
| `Python_for_Data_Engineers_Fundamentals_Han...txt` | Python fundamentals refresher |
| `Python_for_AI_Engineers_COMPLETE_HANDBOOK...txt` | Python patterns specific to AI/GenAI engineering |

### AI / GenAI / Agentic
| File | Covers |
|---|---|
| `GenAI_LLM_Engineering_COMPLETE_HANDBOOK_2026.txt` | RAG architecture, vector DBs, embeddings, chunking, LLM evaluation, fine-tuning vs. RAG, LLMOps |
| `Agentic_AI_COMPLETE_HANDBOOK_2026.txt` | Agent loop, MCP, memory systems, planning strategies, multi-agent orchestration, agent evaluation, security |
| `MLOps_Fundamentals_COMPLETE_HANDBOOK_2026.txt` | Feature stores, experiment tracking, deployment patterns, drift detection, A/B testing, retraining |
| `AI_on_Databricks_COMPLETE_HANDBOOK...txt` | AI/ML features specific to the Databricks platform |
| `AI_on_Databricks_CHEAT_SHEET_2026.txt` | Quick-reference companion to the above |
| `AI_on_Snowflake_COMPLETE_HANDBOOK...txt` | AI/ML features specific to the Snowflake platform (Cortex, etc.) |

### Snowflake
| File | Covers |
|---|---|
| `Snowflake_COMPLETE_HANDBOOK_2026.txt` | Snowflake platform, warehousing, Snowpipe, Horizon governance |

**Total: 20 files** (19 unique + 1 duplicate to resolve).

---

## 2. Suggested Study Sequence

The library isn't meant to be read front-to-back randomly — several files
assume concepts from others. Suggested order:

**Phase 1 — Foundations**
1. Python fundamentals → Python for Data Engineers (complete)
2. Data Modeling (SCD types, grain, star schema — referenced everywhere else)
3. Advanced SQL Scenarios

**Phase 2 — Platform depth**
4. Databricks Complete Handbook
5. Spark Internals Deep Dive (assumes Databricks basics)
6. Snowflake Complete Handbook
7. Streaming & Kafka (assumes Spark Structured Streaming basics from #5)

**Phase 3 — Cloud & Infrastructure**
8. Cloud Fundamentals — Azure
9. Cloud Fundamentals — AWS

**Phase 4 — System Design (assumes Phases 1–3)**
10. System Design for Data Engineers — the 9 worked problems lean on SCD2
    (Data Modeling), shuffle/joins (Spark Internals), Kafka semantics
    (Streaming), and cloud security patterns (Cloud handbooks)

**Phase 5 — AI/GenAI (assumes Phases 1–4)**
11. Python for AI Engineers
12. GenAI/LLM Engineering (RAG architecture reuses medallion/CDC concepts
    from Databricks + Data Modeling)
13. Agentic AI (explicitly extends GenAI handbook Section 11)
14. MLOps Fundamentals (feature store point-in-time correctness reuses
    SCD2 "as-of" join logic from Data Modeling)
15. AI on Databricks / AI on Snowflake (platform-specific AI features)

**Phase 6 — Interview mechanics**
16. Coding & DSA for Data Engineers — do this throughout, not just at the
    end; the patterns cross-reference nearly every other file

---

## 3. Cross-Reference Map

These files were deliberately written to reference each other. Reading them
together (not in isolation) surfaces the connections interviewers reward:

- **Coding & DSA** ↔ nearly everything — two-pointer merge ↔ Spark sort-merge
  join; sliding window ↔ streaming windowed aggregation; keep-latest dedup
  ↔ SCD2/CDC MERGE; topological sort ↔ DAG task scheduling
- **Data Modeling** ↔ **MLOps** — SCD2 "as-of" join = feature store
  point-in-time correctness
- **Data Modeling** ↔ **System Design** — every worked design problem
  states fact table grain explicitly
- **Spark Internals** ↔ **Streaming & Kafka** — Structured Streaming is
  "a batch job run repeatedly," same Catalyst/Tungsten engine underneath
- **System Design** ↔ **Cloud Fundamentals** — the multi-tenant isolation
  problem, security sections reference VNet/VPC, Private Endpoint/VPC
  Endpoint directly
- **GenAI** ↔ **Agentic AI** — Agentic AI Section 1 explicitly extends
  GenAI Section 11's introductory treatment
- **GenAI** ↔ **Databricks/Data Modeling** — RAG ingestion is framed
  explicitly as a medallion-architecture pipeline with an embedding step
- **MLOps** ↔ **Streaming & Kafka** — deployment pattern decision tree
  reuses the batch/streaming/real-time framework directly

---

## 4. Topic → File Lookup Table

| If you need... | Go to... |
|---|---|
| SCD Type 2 MERGE SQL | Data Modeling, Section 7 |
| Fact table grain / star vs. snowflake | Data Modeling, Sections 5, 8 |
| Shuffle, skew, salting | Spark Internals, Sections 6–7 |
| Join strategy selection (broadcast vs. sort-merge) | Spark Internals, Section 8 |
| AQE explained precisely | Spark Internals, Section 9 |
| Exactly-once semantics formula | Streaming & Kafka, Section 6 |
| Watermarking mechanics | Streaming & Kafka, Section 10 |
| Kafka vs. Kinesis vs. Event Hubs | Streaming & Kafka, Section 13; Cloud AWS/Azure, Section 7 |
| RADIO-DE interview framework | System Design, Section 2 |
| Capacity estimation math | System Design, Section 5 |
| Multi-tenant isolation models | System Design, Problem 9 |
| IAM explicit-deny-wins rule | Cloud AWS, Section 3 |
| Managed Identity vs. IAM Role | Cloud Azure Section 3 / Cloud AWS Section 3 |
| S3 durability vs. availability | Cloud AWS, Section 4 |
| RAG pipeline end-to-end | GenAI, Section 6 |
| Chunking strategy tradeoffs | GenAI, Section 7 |
| RAG vs. fine-tuning decision | GenAI, Section 10 |
| LLM-as-judge evaluation | GenAI, Section 9 |
| Agent loop / ReAct / planning | Agentic AI, Sections 2, 6–7 |
| MCP explained | Agentic AI, Section 4–5 |
| Prompt injection via tool outputs | Agentic AI, Section 10/12 |
| Feature store point-in-time correctness | MLOps, Section 3 |
| Data drift vs. concept drift | MLOps, Section 7 |
| Model rollout strategy (shadow/canary/blue-green) | MLOps, Section 5 |
| PySpark coding patterns (window fns, dedup) | Coding & DSA, Section 11 |
| Topological sort for DAG scheduling | Coding & DSA, Section 8 |

---

## 5. Housekeeping Notes

- **Duplicate file flagged:** `Cloud_Fundamentals_Azure_for_Data_Engineers_COMPLETE_HANDBOOK_2026.txt`
  appears to be an earlier draft of `Cloud_Fundamentals_Azure_COMPLETE_HANDBOOK_2026.txt`.
  Recommend deleting the `_for_Data_Engineers_` version to avoid confusion —
  just say the word and I'll remove it from the outputs folder before you
  upload.
- All files are self-contained `.txt` — no external dependencies between
  them required to read any single one, but the cross-references above are
  where the real interview-signal depth comes from.

---

## 6. Gap Tracker — What's Not Yet Covered

| Topic | Status |
|---|---|
| Orchestration (Airflow / Lakeflow Jobs, DAG design, backfills) | Not started |
| Behavioral / Leadership Principles playbook (STAR answers, Amazon LP mapping) | Not started |
| Mock interview question bank (cross-topic, company/difficulty tagged) | Not started |
| Data Quality & Testing / DataOps (Great Expectations, dbt tests, data contracts) | Mentioned in passing across several files; no dedicated handbook yet |

---

*Last updated: July 26, 2026. Update this file whenever a new handbook is added.*
