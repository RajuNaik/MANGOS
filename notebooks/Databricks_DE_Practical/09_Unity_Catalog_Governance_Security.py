# Databricks notebook source

# MAGIC %md
# MAGIC # 🔐 Notebook 09 — Unity Catalog, Data Governance & Security
# MAGIC
# MAGIC **Handbook Sections Covered**: Part E1–E14, J19–J25
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Learning Objectives
# MAGIC 1. **Unity Catalog Architecture** — 3-level namespace (`catalog.schema.table`), Metastore hierarchy, and cross-workspace sharing.
# MAGIC 2. **Managed vs External Tables & Locations** — Object storage ownership, lifecycle management, and storage credentials.
# MAGIC 3. **Volumes (Managed vs External)** — Govern unstructured files (PDFs, images, CSV dumps) under Unity Catalog access controls.
# MAGIC 4. **Fine-Grained Security Controls** — Implement Column-Level Masking Functions and Row-Level Security Filters.
# MAGIC 5. **Access Control Privilege Model** — `GRANT` and `REVOKE` permissions on catalogs, schemas, tables, and views.
# MAGIC 6. **Data Lineage & Audit System Tables** — Automated column-level lineage tracking and querying `system.access.audit`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Setup & Namespace Initialization

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()

# Create 3-level namespace simulation schemas on CE
spark.sql("CREATE DATABASE IF NOT EXISTS prod_catalog_sales_schema")
spark.sql("USE prod_catalog_sales_schema")

print("✅ Setup complete! Current Namespace Database:", spark.catalog.currentDatabase())

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 1: Unity Catalog 3-Level Namespace & Metastore Hierarchy (Handbook E1–E4)
# MAGIC
# MAGIC ```text
# MAGIC                     UNITY CATALOG METASTORE
# MAGIC                                │
# MAGIC         ┌──────────────────────┼──────────────────────┐
# MAGIC         ▼                      ▼                      ▼
# MAGIC   main (Catalog)        dev_catalog (Catalog)   prod_catalog (Catalog)
# MAGIC         │
# MAGIC    ┌────┴────┐
# MAGIC    ▼         ▼
# MAGIC  sales     finance (Schemas / Databases)
# MAGIC    │
# MAGIC  ┌─┴───────────────────────────────┐
# MAGIC  ▼                                 ▼
# MAGIC orders (Table)             invoices (Volume - Files)
# MAGIC ```
# MAGIC
# MAGIC ### 3-Level Namespace Syntax (E2):
# MAGIC Every query explicitly references: **`SELECT * FROM catalog_name.schema_name.table_name`**

# COMMAND ----------

# Production Unity Catalog SQL DDL Statements (Production Reference Syntax)
print("=" * 80)
print("  UNITY CATALOG 3-LEVEL NAMESPACE CREATION (PRODUCTION SYNTAX)")
print("=" * 80)

# SQL Statements demonstrating Catalog, Schema, Table, and Volume creation
uc_ddl_script = """
-- 1. Create Catalog
CREATE CATALOG IF NOT EXISTS enterprise_prod;

-- 2. Create Schema with Storage Location
CREATE SCHEMA IF NOT EXISTS enterprise_prod.retail_sales
COMMENT 'Production Sales & Orders Domain';

-- 3. Use 3-Level Namespace
USE CATALOG enterprise_prod;
USE SCHEMA retail_sales;
"""
print(uc_ddl_script)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 2: Managed vs External Tables & Volumes (Handbook E5, E6, E9)
# MAGIC
# MAGIC | Property | Managed Table / Volume | External Table / Volume |
# MAGIC |---|---|---|
# MAGIC | **Data Storage Location** | Managed inside Unity Catalog default storage path | Custom specified external storage location (`LOCATION 's3://my-bucket/path'`) |
# MAGIC | **`DROP TABLE` Behavior** | **Deletes BOTH metadata AND physical underlying data files!** | **Deletes ONLY metadata.** Physical raw data files remain intact in S3/ADLS! |
# MAGIC | **Ownership** | Unity Catalog owns the data lifecycle | External team/vendor owns the underlying data files |

# COMMAND ----------

# Create Managed & External Tables (Simulated DDL)
spark.sql("""
CREATE TABLE IF NOT EXISTS prod_catalog_sales_schema.managed_customers (
    customer_id STRING,
    email STRING,
    loyalty_tier STRING
) USING DELTA
""")

# Create External Table (pointing to explicit location)
spark.sql("""
CREATE TABLE IF NOT EXISTS prod_catalog_sales_schema.external_orders (
    order_id STRING,
    total_amount DOUBLE
) USING DELTA
LOCATION '/tmp/de_practical_external_orders'
""")

# Populate Managed Table
df_cust = spark.table("de_practical_db.raw_customers").select("customer_id", "email", "loyalty_tier")
df_cust.write.format("delta").mode("overwrite").insertInto("prod_catalog_sales_schema.managed_customers")

print("=" * 80)
print("  MANAGED VS EXTERNAL TABLES IN METASTORE")
print("=" * 80)
spark.sql("DESCRIBE EXTENDED prod_catalog_sales_schema.managed_customers").filter(F.col("col_name").isin(["Type", "Location"])).show(truncate=False)
spark.sql("DESCRIBE EXTENDED prod_catalog_sales_schema.external_orders").filter(F.col("col_name").isin(["Type", "Location"])).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Volumes: Managed vs External Files Governance (Handbook E9)
# MAGIC **Volumes** are Unity Catalog objects that govern non-tabular files (PDFs, images, ML models, large CSV dumps).
# MAGIC
# MAGIC ```sql
# MAGIC -- Create Managed Volume (Files stored inside Catalog storage)
# MAGIC CREATE VOLUME enterprise_prod.retail_sales.raw_pdf_volume;
# MAGIC
# MAGIC -- Access Volume files in PySpark via POSIX path:
# MAGIC -- /Volumes/enterprise_prod/retail_sales/raw_pdf_volume/invoice_2026.pdf
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 3: Fine-Grained Security: Column Masking & Row Filtering (Handbook E10, E11)
# MAGIC
# MAGIC ### 1. Column-Level Masking UDFs (E10)
# MAGIC Dynamically redacts PII data (e.g., email or SSN) based on the user's role executing the query. Analysts see `redacted***@domain.com`, while Admins see full plain-text.
# MAGIC
# MAGIC ### 2. Row-Level Security Filters (E11)
# MAGIC Dynamically filters returned rows based on user group membership (e.g., European analysts only see rows where `region = 'Europe'`).

# COMMAND ----------

# ============================================================================
# E10: Column Masking Function Implementation
# ============================================================================
spark.sql("""
CREATE OR REPLACE FUNCTION prod_catalog_sales_schema.email_mask(email STRING)
RETURN CASE 
    WHEN IS_MEMBER('admin_group') THEN email
    ELSE CONCAT(LEFT(email, 2), '****@company.com')
END;
""")

# Apply Column Mask to Table
spark.sql("""
ALTER TABLE prod_catalog_sales_schema.managed_customers 
ALTER COLUMN email SET MASK prod_catalog_sales_schema.email_mask;
""")

print("✅ Column-Level Masking Function Applied to 'email' Column.")

# ============================================================================
# E11: Row-Level Security Filter Function Implementation
# ============================================================================
spark.sql("""
CREATE OR REPLACE FUNCTION prod_catalog_sales_schema.tier_row_filter(tier STRING)
RETURN CASE 
    WHEN IS_MEMBER('executive_group') THEN TRUE
    WHEN tier IN ('Gold', 'Platinum') THEN TRUE
    ELSE FALSE
END;
""")

# Apply Row Filter to Table
spark.sql("""
ALTER TABLE prod_catalog_sales_schema.managed_customers 
SET ROW FILTER prod_catalog_sales_schema.tier_row_filter ON (loyalty_tier);
""")

print("✅ Row-Level Security Filter Function Applied to 'loyalty_tier' Column.")

# Query Table to See Masking & Filter Effect
print("\n--- Query Output with Active Security Mask & Row Filter ---")
spark.table("prod_catalog_sales_schema.managed_customers").show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 4: Access Control Privilege Model (`GRANT` & `REVOKE`) (Handbook E7, E8)
# MAGIC
# MAGIC Unity Catalog uses an explicit **Inherited Privilege Model**:
# MAGIC - `GRANT USE CATALOG ON CATALOG main TO group_analysts;`
# MAGIC - `GRANT USE SCHEMA ON SCHEMA main.sales TO group_analysts;`
# MAGIC - `GRANT SELECT ON TABLE main.sales.orders TO group_analysts;`

# COMMAND ----------

# SQL Privilege Granting Statements
grant_script = """
-- Grant Catalog & Schema Use
GRANT USE CATALOG ON CATALOG enterprise_prod TO `data_analysts_group`;
GRANT USE SCHEMA ON SCHEMA enterprise_prod.retail_sales TO `data_analysts_group`;

-- Grant Table Select & Modify
GRANT SELECT ON TABLE enterprise_prod.retail_sales.managed_customers TO `data_analysts_group`;
GRANT MODIFY, INSERT ON TABLE enterprise_prod.retail_sales.managed_customers TO `data_engineers_group`;

-- Revoke Permissions
REVOKE DROP ON TABLE enterprise_prod.retail_sales.managed_customers FROM `data_analysts_group`;
"""

print("=" * 80)
print("  UNITY CATALOG PRIVILEGE GRANT & REVOKE SPECIFICATION")
print("=" * 80)
print(grant_script)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 5: Automated Lineage & System Audit Tables (Handbook E12, E13)
# MAGIC
# MAGIC ### 1. Column-Level Data Lineage (E12)
# MAGIC Unity Catalog automatically tracks end-to-end lineage from source files -> Bronze tables -> Silver tables -> Gold Star Schema tables -> BI dashboards without writing any code.
# MAGIC
# MAGIC ### 2. System Tables (`system.access.audit`) (E13)
# MAGIC System Tables are built-in governed Delta tables located in the `system` catalog:
# MAGIC - `system.access.audit`: Logs every query, user login, credential access, and table read/write operation across the entire enterprise.
# MAGIC - `system.billing.usage`: Tracks DBU (Databricks Unit) consumption, cluster costs, and job spend in real-time.

# COMMAND ----------

# Query System Table Audit Pattern (Production Query Example)
print("=" * 80)
print("  SYSTEM AUDIT TABLE QUERY (PRODUCTION REFERENCE)")
print("=" * 80)

audit_query = """
SELECT 
    event_time,
    user_identity.email AS user_email,
    action_name,
    request_params.table_full_name AS queried_table,
    audit_level
FROM system.access.audit
WHERE service_name = 'unityCatalog'
  AND action_name IN ('getTable', 'select', 'createTable')
ORDER BY event_time DESC
LIMIT 10;
"""

print(audit_query)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 💡 Senior Data Engineer Interview Practice (Handbook H3)
# MAGIC
# MAGIC ### Q1: Explain the critical difference between a Managed Table and an External Table in Unity Catalog.
# MAGIC **Answer**:
# MAGIC - **Managed Table**: Unity Catalog fully manages both the metadata AND the underlying physical data storage path (inside the catalog/schema root location). When you run `DROP TABLE managed_table`, Unity Catalog **permanently deletes both the metadata AND all physical Parquet data files from object storage**.
# MAGIC - **External Table**: Unity Catalog manages ONLY the metadata; the physical data files reside in an external cloud storage path (`LOCATION 's3://my-bucket/path'`) governed by an External Location. When you run `DROP TABLE external_table`, Unity Catalog **deletes ONLY the metadata entry**; the raw underlying data files in object storage remain untouched.
# MAGIC
# MAGIC ### Q2: How do Column-Level Masking and Row-Level Security work in Unity Catalog?
# MAGIC **Answer**: They are implemented as fine-grained SQL UDFs applied directly to target tables via `ALTER TABLE`:
# MAGIC - **Column-Level Masking**: Uses a SQL function (`SET MASK mask_fn`) that dynamically intercepts queries on a column (e.g., `ssn` or `email`). It evaluates `IS_MEMBER('admin_group')`; if True, it returns the unmasked value; otherwise, it returns a redacted string (`XXX-XX-1234`).
# MAGIC - **Row-Level Security**: Uses a SQL filter function (`SET ROW FILTER filter_fn ON (region)`) that injects a hidden `WHERE` predicate into the user's execution plan at runtime based on user identity or group membership.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # ✅ Module 09 Summary
# MAGIC
# MAGIC | Concept | Syntax / Implementation | Handbook Reference |
# MAGIC |---|---|---|
# MAGIC | 3-Level Namespace | `catalog.schema.table` | E1–E4 |
# MAGIC | Table Types | Managed (UC owns files & deletes on drop) vs External (`LOCATION`) | E5, E6 |
# MAGIC | Volumes | Managed vs External Volumes for non-tabular files | E9 |
# MAGIC | Security UDFs | Column Masking (`SET MASK`) & Row Filtering (`SET ROW FILTER`) | E10, E11 |
# MAGIC | Privilege Model | `GRANT SELECT ON TABLE catalog.schema.table TO group` | E7, E8 |
# MAGIC | Governance | Automated Lineage & `system.access.audit` System Tables | E12, E13 |
