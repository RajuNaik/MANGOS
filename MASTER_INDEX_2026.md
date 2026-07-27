# MANGOS Interview Prep Library — Master Index (2026)
### MAANG / Top-Tier Tech — Data & AI Engineer Track

This index ties together every handbook in the library: what each one covers,
what order to read them in, how they reference each other, and where to find
a specific topic fast. Treat this file as the front door to the repo.

---

## 1. Full Inventory

**22 files total** — 21 handbooks/reference files + this index.

### 🔧 Core Data Engineering (8 files)
| File | Covers |
|---|---|
| `Databricks_COMPLETE_HANDBOOK_2026.txt` | Medallion architecture, Delta Lake, Lakeflow, Unity Catalog, 2026 Summit updates (Agent Bricks, Lakebase, Unity AI Gateway) |
| `Databricks_CHEAT_SHEET_2026.txt` | Fast-lookup companion to the above |
| `Spark_Internals_Deep_Dive_COMPLETE_HANDBOOK_2026.txt` | Catalyst optimizer, Tungsten/codegen, shuffle, joins, AQE, memory management, Delta Lake internals |
| `Streaming_and_Kafka_COMPLETE_HANDBOOK_2026.txt` | Kafka fundamentals, delivery semantics, ordering, schema registry, Structured Streaming, watermarking, Kafka Streams vs. Flink |
| `Data_Modeling_COMPLETE_HANDBOOK_2026.txt` | Normalization/denormalization, star/snowflake/Data Vault/OBT, fact table grain, all SCD types with SQL |
| `Advanced_SQL_Scenarios_COMPLETE_HANDBOOK_2026.txt` | Window functions, gaps-and-islands, complex query patterns |
| `Snowflake_COMPLETE_HANDBOOK_2026.txt` | Warehousing, Snowpipe, Horizon governance |
| `Orchestration_COMPLETE_HANDBOOK_2026.txt` | Airflow architecture, DAG semantics, backfills, retries/SLAs, Lakeflow Jobs vs. Airflow vs. Dagster |

### 🏗️ System Design & Cloud (3 files)
| File | Covers |
|---|---|
| `System_Design_for_Data_Engineers_COMPLETE_HANDBOOK_2026.txt` | RADIO-DE interview framework, capacity estimation, 9 worked design problems, company-specific patterns |
| `Cloud_Fundamentals_Azure_COMPLETE_HANDBOOK_2026.txt` | Entra ID, ADLS Gen2, networking, Databricks-on-Azure architecture |
| `Cloud_Fundamentals_AWS_COMPLETE_HANDBOOK_2026.txt` | IAM, S3, VPC, Databricks-on-AWS architecture, AWS↔Azure↔GCP equivalence |

### 🐍 Python (3 files)
| File | Covers |
|---|---|
| `Python_for_Data_Engineers_COMPLETE_HANDBOOK_2026.txt` | Production Python patterns for DE pipelines |
| `Python_for_Data_Engineers_Fundamentals_Handbook_2026.txt` | Python fundamentals refresher |
| `Python_for_AI_Engineers_COMPLETE_HANDBOOK_2026.txt` | Python patterns specific to AI/GenAI engineering |

### 🤖 AI, GenAI & Agentic Systems (5 files)
| File | Covers |
|---|---|
| `GenAI_LLM_Engineering_COMPLETE_HANDBOOK_2026.txt` | RAG architecture, vector DBs, embeddings, chunking, LLM evaluation, fine-tuning vs. RAG, LLMOps |
| `Agentic_AI_COMPLETE_HANDBOOK_2026.txt` | Agent loop, MCP, memory systems, planning strategies, multi-agent orchestration, agent evaluation, security |
| `MLOps_Fundamentals_COMPLETE_HANDBOOK_2026.txt` | Feature stores, experiment tracking, deployment patterns, drift detection, A/B testing, retraining |
| `AI_on_Databricks_COMPLETE_HANDBOOK_2026.txt` + `AI_on_Databricks_CHEAT_SHEET_2026.txt` | AI/ML features specific to the Databricks platform |
| `AI_on_Snowflake_COMPLETE_HANDBOOK_2026.txt` | AI/ML features specific to the Snowflake platform (Cortex, etc.) |

### 🎯 Interview Mechanics (2 files)
| File | Covers |
|---|---|
| `Coding_and_DSA_for_Data_Engineers_COMPLETE_HANDBOOK_2026.txt` | LeetCode-style coding rounds with a DE flavor, PySpark-specific coding questions |
| `Mock_Interview_Question_Bank_COMPLETE_2026.txt` | 190+ tagged practice questions across every topic, company-flavored tracks, a full mock-loop simulation, self-assessment rubric |

### 🧭 Navigation (1 file)
| File | Purpose |
|---|---|
| `MASTER_INDEX_2026.md` | This file — full inventory, study sequence, cross-reference map, topic lookup table |

---

## 2. Suggested Study Sequence

The library isn't meant to be read front-to-back randomly — several files
assume concepts from others. Suggested order:

**Phase 1 — Foundations**
1. Python Fundamentals → Python for Data Engineers (complete)
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

**Phase 4 — Orchestration & System Design**
10. Orchestration (assumes medallion architecture from #4, idempotency
    concepts introduced in #9's design problems)
11. System Design for Data Engineers — the 9 worked problems lean on SCD2
    (Data Modeling), shuffle/joins (Spark Internals), Kafka semantics
    (Streaming), cloud security patterns (Cloud handbooks), and DAG/
    backfill safety (Orchestration)

**Phase 5 — AI/GenAI (assumes Phases 1–4)**
12. Python for AI Engineers
13. GenAI/LLM Engineering (RAG architecture reuses medallion/CDC concepts
    from Databricks + Data Modeling)
14. Agentic AI (explicitly extends GenAI handbook Section 11)
15. MLOps Fundamentals (feature store point-in-time correctness reuses
    SCD2 "as-of" join logic from Data Modeling)
16. AI on Databricks / AI on Snowflake (platform-specific AI features)

**Phase 6 — Interview Mechanics (run throughout, not just at the end)**
17. Coding & DSA for Data Engineers — the patterns cross-reference nearly
    every other file, so revisit this alongside Phases 2–5, not after them
18. Mock Interview Question Bank — start light drilling once Phase 2 is
    done; save full mock-loop simulations for after Phase 5

---

## 3. Cross-Reference Map

These files were deliberately written to reference each other. Reading them
together (not in isolation) surfaces the connections interviewers reward:

- **Coding & DSA** ↔ nearly everything — two-pointer merge ↔ Spark sort-merge
  join; sliding window ↔ streaming windowed aggregation; keep-latest dedup
  ↔ SCD2/CDC MERGE; topological sort ↔ DAG task scheduling (Orchestration)
- **Data Modeling** ↔ **MLOps** — SCD2 "as-of" join = feature store
  point-in-time correctness
- **Data Modeling** ↔ **System Design** — every worked design problem
  states fact table grain explicitly
- **Spark Internals** ↔ **Streaming & Kafka** — Structured Streaming is
  "a batch job run repeatedly," same Catalyst/Tungsten engine underneath
- **System Design** ↔ **Cloud Fundamentals** — the multi-tenant isolation
  problem and security sections reference VNet/VPC, Private Endpoint/VPC
  Endpoint directly
- **System Design** ↔ **Orchestration** — idempotent MERGE-based writes
  (System Design's CDC problem) are the exact prerequisite Orchestration's
  backfill-safety checklist assumes
- **Orchestration** ↔ **Databricks** — Lakeflow Jobs vs. Lakeflow
  Declarative Pipelines layering is explained precisely in Orchestration
  Section 10, extending the Databricks handbook's medallion coverage
- **GenAI** ↔ **Agentic AI** — Agentic AI Section 1 explicitly extends
  GenAI Section 11's introductory treatment
- **GenAI** ↔ **Databricks/Data Modeling** — RAG ingestion is framed
  explicitly as a medallion-architecture pipeline with an embedding step
- **MLOps** ↔ **Streaming & Kafka** — deployment pattern decision tree
  reuses the batch/streaming/real-time framework directly
- **Mock Interview Question Bank** ↔ every file above — every question is
  tagged with a direct section pointer back to its source handbook

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
| DAG semantics / logical date confusion | Orchestration, Section 5 |
| Backfill safety checklist | Orchestration, Section 6 |
| Lakeflow Jobs vs. Declarative Pipelines | Orchestration, Section 10 |
| Airflow vs. Lakeflow vs. Dagster vs. Step Functions | Orchestration, Section 11 |
| IAM explicit-deny-wins rule | Cloud AWS, Section 3 |
| Managed Identity vs. IAM Role | Cloud Azure Section 3 / Cloud AWS Section 3 |
| S3 durability vs. availability | Cloud AWS, Section 4 |
| RAG pipeline end-to-end | GenAI, Section 6 |
| Chunking strategy tradeoffs | GenAI, Section 7 |
| RAG vs. fine-tuning decision | GenAI, Section 10 |
| LLM-as-judge evaluation | GenAI, Section 9 |
| Agent loop / ReAct / planning | Agentic AI, Sections 2, 6–7 |
| MCP explained | Agentic AI, Section 4 |
| Prompt injection via tool outputs | Agentic AI, Section 10 |
| Feature store point-in-time correctness | MLOps, Section 3 |
| Data drift vs. concept drift | MLOps, Section 7 |
| Model rollout strategy (shadow/canary/blue-green) | MLOps, Section 5 |
| PySpark coding patterns (window fns, dedup) | Coding & DSA, Section 11 |
| Topological sort for DAG scheduling | Coding & DSA, Section 8 |
| Practice questions by topic/company/difficulty | Mock Interview Question Bank, Sections 3–11 |
| A timed full-onsite-day mock simulation | Mock Interview Question Bank, Section 12 |

---

## 5. Housekeeping Notes

- **Duplicate resolved:** an earlier draft (`Cloud_Fundamentals_Azure_for_Data_Engineers_COMPLETE_HANDBOOK_2026.txt`)
  has been removed — `Cloud_Fundamentals_Azure_COMPLETE_HANDBOOK_2026.txt`
  is the current, correct version.
- All files are self-contained — no external dependencies required to read
  any single one, but the cross-references in Section 3 are where the real
  interview-signal depth comes from.

---

## 6. Gap Tracker — What's Not Yet Covered

| Topic | Status |
|---|---|
| Orchestration (Airflow / Lakeflow Jobs, DAG design, backfills) | ✅ Done |
| Mock interview question bank (cross-topic, company/difficulty tagged) | ✅ Done |
| Behavioral / Leadership Principles playbook (STAR answers, Amazon LP mapping) | 🔲 Not started |
| Dedicated Data Quality & DataOps handbook (Great Expectations, dbt tests, data contracts, CI/CD for pipelines) | 🔲 Not started — currently only mentioned in passing across several files |

---

*Last updated: July 27, 2026. Update this file whenever a new handbook is added.*
