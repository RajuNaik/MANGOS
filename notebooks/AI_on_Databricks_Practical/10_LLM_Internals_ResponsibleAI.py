# Databricks notebook source

# MAGIC %md
# MAGIC # 🧬 Notebook 10 — LLM Internals & Responsible AI
# MAGIC
# MAGIC **Handbook Sections Covered**: I12 (Inference Parameters), I13 (Context Window), I14 (Hallucination), I18 (Responsible AI), I19 (Transformer Architecture), I20 (DE's Role Synthesis)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Learning Objectives
# MAGIC 1. **Temperature & sampling** — how randomness controls affect output
# MAGIC 2. **Context window** — token limits and strategies for long conversations
# MAGIC 3. **Hallucination** — detection, causes, and the full mitigation toolkit
# MAGIC 4. **Transformer architecture** — attention mechanism and next-token prediction
# MAGIC 5. **Responsible AI** — fairness, bias, PII, explainability
# MAGIC 6. **DE's role synthesis** — mapping every topic to concrete DE responsibilities

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Setup

# COMMAND ----------

%pip install -q sentence-transformers

# COMMAND ----------

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
import json
import re
import warnings
warnings.filterwarnings('ignore')

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

print("✅ Setup complete!")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 1: LLM Inference Parameters (Handbook I12)
# MAGIC
# MAGIC These are the knobs you turn when calling an LLM:
# MAGIC
# MAGIC | Parameter | What it controls | Typical range |
# MAGIC |-----------|-----------------|---------------|
# MAGIC | **Temperature** | Randomness/creativity | 0.0 (deterministic) – 2.0 (very random) |
# MAGIC | **Top-P** | Nucleus sampling threshold | 0.0 – 1.0 |
# MAGIC | **Max Tokens** | Response length cap | 1 – model max |
# MAGIC | **System Prompt** | Persistent behavior/role | N/A |

# COMMAND ----------

# ============================================================================
# TEMPERATURE: How randomness affects output
# ============================================================================

# Simulate temperature's effect on text generation
# At temp=0, always pick the highest probability word
# At temp=1, sample proportionally to probabilities
# At temp>1, make the distribution more uniform (more random)

def simulate_generation(prompt: str, temperature: float, vocabulary: dict, length: int = 15) -> str:
    """
    Simulate how temperature affects next-token selection.
    vocabulary: {context_word: {next_word: probability}}
    """
    np.random.seed(42)  # Same seed to show temperature's ISOLATED effect
    
    words = prompt.split()
    current = words[-1].lower() if words else "the"
    generated = list(words)
    
    for _ in range(length):
        if current in vocabulary:
            options = vocabulary[current]
            words_list = list(options.keys())
            probs = np.array(list(options.values()), dtype=float)
            
            if temperature == 0:
                # Greedy: always pick highest probability
                idx = np.argmax(probs)
            else:
                # Apply temperature: divide logits by temperature, then softmax
                logits = np.log(probs + 1e-10) / max(temperature, 0.01)
                exp_logits = np.exp(logits - np.max(logits))
                adjusted_probs = exp_logits / exp_logits.sum()
                idx = np.random.choice(len(words_list), p=adjusted_probs)
            
            next_word = words_list[idx]
            generated.append(next_word)
            current = next_word
        else:
            current = np.random.choice(list(vocabulary.keys()))
            generated.append(current)
    
    return " ".join(generated)

# Simple vocabulary for demonstration
vocab = {
    "the": {"customer": 0.4, "order": 0.3, "system": 0.15, "dancing": 0.05, "purple": 0.05, "quantum": 0.05},
    "customer": {"needs": 0.3, "wants": 0.25, "requested": 0.2, "account": 0.15, "exploded": 0.05, "teleported": 0.05},
    "order": {"was": 0.35, "has": 0.25, "should": 0.15, "might": 0.1, "spontaneously": 0.1, "cosmically": 0.05},
    "needs": {"a": 0.4, "to": 0.3, "immediate": 0.15, "philosophical": 0.1, "interdimensional": 0.05},
    "was": {"processed": 0.35, "shipped": 0.3, "cancelled": 0.2, "levitated": 0.1, "harmonized": 0.05},
    "wants": {"to": 0.4, "a": 0.3, "the": 0.15, "universal": 0.1, "transcendent": 0.05},
    "a": {"refund": 0.35, "replacement": 0.25, "response": 0.2, "unicorn": 0.1, "paradox": 0.1},
    "to": {"cancel": 0.3, "upgrade": 0.25, "return": 0.2, "transcend": 0.15, "implode": 0.1},
    "processed": {"successfully": 0.5, "today": 0.25, "quickly": 0.15, "mysteriously": 0.1},
    "shipped": {"today": 0.4, "yesterday": 0.3, "via": 0.2, "telepathically": 0.1},
}

print("=" * 70)
print("  TEMPERATURE EFFECTS ON GENERATION (I12)")
print("=" * 70)

for temp in [0.0, 0.3, 0.7, 1.0, 1.5]:
    output = simulate_generation("The", temp, vocab, length=10)
    creativity = "🧊 Deterministic" if temp == 0 else "🔥" * min(int(temp * 3), 5) + " Creative"
    print(f"\n  Temp={temp:.1f} ({creativity}):")
    print(f"    → {output}")

print(f"""
  💡 KEY INSIGHT for Data Engineers (I12):
     • Batch classification/extraction → LOW temperature (0.0-0.3)
       Consistency and reproducibility matter more than creativity
     • Creative tasks (brainstorming) → HIGHER temperature (0.7-1.0)
     • Never use high temperature for factual extraction in a pipeline
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ### System Prompt vs User Prompt

# COMMAND ----------

# ============================================================================
# SYSTEM PROMPT vs USER PROMPT
# ============================================================================

print("=" * 70)
print("  SYSTEM PROMPT vs USER PROMPT (I12)")
print("=" * 70)
print("""
  ┌────────────────────────────────────────────────────────────────┐
  │  SYSTEM PROMPT (persistent behavior — set ONCE per session)   │
  │                                                                │
  │  "You are a customer support agent for AcmeCorp. Be concise, │
  │   professional, and always cite relevant documentation.        │
  │   Never make up information you're not sure about.             │
  │   If unsure, direct the customer to support@acmecorp.com."    │
  │                                                                │
  │  → Sets the ROLE, TONE, CONSTRAINTS for every response        │
  │  → Usually NOT shown to the end user                          │
  │  → Where guardrails like "never reveal X" go                  │
  ├────────────────────────────────────────────────────────────────┤
  │  USER PROMPT (specific input — changes every turn)            │
  │                                                                │
  │  "How do I cancel my subscription?"                           │
  │                                                                │
  │  → The actual question/instruction for THIS interaction       │
  │  → Combined with system prompt + conversation history         │
  │  → Combined with RAG context (if applicable)                  │
  └────────────────────────────────────────────────────────────────┘

  In code:
    client.chat.completions.create(
        model="...",
        messages=[
            {"role": "system", "content": system_prompt},   # persistent
            {"role": "user", "content": user_question},      # this turn
        ]
    )
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 2: Context Window & Token Limits (Handbook I13)
# MAGIC
# MAGIC Every model has a maximum **context window** — the total tokens it can process at once.

# COMMAND ----------

# ============================================================================
# CONTEXT WINDOW: Demonstration and strategies
# ============================================================================

print("=" * 70)
print("  CONTEXT WINDOW & TOKEN LIMITS (I13)")
print("=" * 70)

# Approximate token counting (1 token ≈ 0.75 words, or ~4 characters)
def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.33))

# Show how a context window fills up
system_prompt = "You are a helpful customer support agent for AcmeCorp. Always be professional and cite sources."
conversation_history = [
    "User: How do I cancel my subscription?\nAssistant: Go to Settings > Billing > Cancel.",
    "User: Will I get a refund?\nAssistant: If within 14 days of billing cycle, yes.",
    "User: What happens to my data?\nAssistant: Retained for 90 days, then deleted.",
    "User: Can I reactivate later?\nAssistant: Yes, within the 90-day retention period.",
    "User: What about my team members?\nAssistant: They'll lose access when your subscription ends.",
]

rag_context = "To cancel your subscription, go to Settings > Billing > Cancel Subscription. Your subscription will remain active until the end of your current billing period. Data retained for 90 days."
current_question = "If I reactivate, will my team members' access be restored automatically?"

# Calculate token usage
context_windows = {"GPT-3.5": 4096, "Llama 3 8B": 8192, "GPT-4o": 128000, "Claude 3.5": 200000}

system_tokens = estimate_tokens(system_prompt)
history_tokens = sum(estimate_tokens(msg) for msg in conversation_history)
rag_tokens = estimate_tokens(rag_context)
question_tokens = estimate_tokens(current_question)
reserved_for_output = 500

total_input = system_tokens + history_tokens + rag_tokens + question_tokens

print(f"\n  Token Budget Breakdown:")
print(f"  ┌─────────────────────────────┬─────────┐")
print(f"  │ System prompt               │ {system_tokens:>5d}   │")
print(f"  │ Conversation history (5 turns) │ {history_tokens:>5d}   │")
print(f"  │ RAG context                  │ {rag_tokens:>5d}   │")
print(f"  │ Current question             │ {question_tokens:>5d}   │")
print(f"  │ Reserved for output          │ {reserved_for_output:>5d}   │")
print(f"  ├─────────────────────────────┼─────────┤")
print(f"  │ TOTAL NEEDED                 │ {total_input + reserved_for_output:>5d}   │")
print(f"  └─────────────────────────────┴─────────┘")

print(f"\n  Model Context Windows:")
for model, window in context_windows.items():
    fits = "✅" if (total_input + reserved_for_output) < window else "❌ WON'T FIT"
    utilization = (total_input + reserved_for_output) / window * 100
    print(f"    {model:<15s} {window:>8,d} tokens  {fits}  ({utilization:.1f}% used)")

print(f"""
  📌 STRATEGIES for managing the context window:
  1. RAG (A7): Retrieve ONLY relevant chunks — don't paste entire KB
  2. Conversation summarization: Summarize old turns instead of keeping raw text
  3. Agent Memory (D4): Store long-term context in Lakebase, not in the prompt
  4. Token-aware chunking: Size RAG chunks to leave room for history + output
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 3: Hallucination — Detection & Mitigation (Handbook I14)

# COMMAND ----------

# ============================================================================
# HALLUCINATION: When the model says something confidently wrong
# ============================================================================

print("=" * 70)
print("  HALLUCINATION — The #1 GenAI reliability problem (I14)")
print("=" * 70)

# Demonstrate hallucination detection
context = """AcmeCorp was founded in 2019. The company offers four plans: Free, Basic ($9.99/month), 
Premium ($24.99/month), and Enterprise (custom pricing). The company is headquartered in San Francisco 
and has 200 employees. AcmeCorp is SOC 2 Type II certified and GDPR compliant."""

answers = {
    "grounded": "AcmeCorp was founded in 2019 and is headquartered in San Francisco. They offer four plans including a Free tier.",
    "hallucinated": "AcmeCorp was founded in 2015 and is headquartered in New York. They have 5,000 employees and are listed on the NASDAQ.",
    "partially_hallucinated": "AcmeCorp offers four plans including a Free tier and is SOC 2 certified. They also hold FedRAMP authorization.",
}

print(f"\n  Context: {context[:100]}...")
print()

for label, answer in answers.items():
    # Semantic similarity between answer and context (groundedness check)
    answer_emb = embedding_model.encode([answer])
    context_emb = embedding_model.encode([context])
    similarity = float(cosine_similarity(answer_emb, context_emb)[0][0])
    
    # Check for specific factual claims
    claims_in_context = 0
    claims_total = 0
    facts = {
        "2019": "founded year",
        "San Francisco": "headquarters",
        "200 employees": "employee count",
        "SOC 2": "certification",
        "four plans": "plan count",
    }
    wrong_facts = {
        "2015": "wrong founding year",
        "New York": "wrong headquarters",
        "5,000": "wrong employee count",
        "NASDAQ": "not stated",
        "FedRAMP": "not stated",
    }
    
    for fact, desc in facts.items():
        if fact.lower() in answer.lower():
            claims_in_context += 1
            claims_total += 1
    for wrong, desc in wrong_facts.items():
        if wrong.lower() in answer.lower():
            claims_total += 1
    
    grounded_ratio = claims_in_context / max(claims_total, 1)
    
    status = "✅ GROUNDED" if grounded_ratio > 0.8 else ("⚠️ PARTIAL" if grounded_ratio > 0.4 else "❌ HALLUCINATED")
    
    print(f"  {status} ({label}):")
    print(f"    Answer: \"{answer[:80]}...\"")
    print(f"    Groundedness: {grounded_ratio:.0%} ({claims_in_context}/{claims_total} facts verified)")
    print(f"    Similarity to context: {similarity:.3f}")
    print()

# COMMAND ----------

# ============================================================================
# THE FULL HALLUCINATION MITIGATION TOOLKIT (I14)
# ============================================================================

print("=" * 70)
print("  HALLUCINATION MITIGATION TOOLKIT (I14)")
print("=" * 70)
print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │  PREVENTION (reduce likelihood):                                │
  │                                                                 │
  │  1. RAG (A7) — ground answers in REAL, retrieved documents      │
  │  2. Low temperature (I12) — reduce randomness for factual tasks │
  │  3. Structured outputs (I6) — constrain the SHAPE of output     │
  │  4. Genie Ontology (F2) — ground business terms from catalog    │
  │  5. Good prompts — "answer ONLY from the context provided"      │
  │                                                                 │
  ├─────────────────────────────────────────────────────────────────┤
  │  DETECTION (catch what slips through):                          │
  │                                                                 │
  │  1. LLM judges/groundedness scoring (E2)                        │
  │     → Automated: "is this answer supported by the context?"     │
  │  2. Human review loops (MLflow 3 Review App, E2)                │
  │     → Catches what automated judges miss                        │
  │  3. Production monitoring (E5)                                  │
  │     → Continuous groundedness scoring on live traffic            │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘

  Interview-ready one-liner:
  "You don't eliminate hallucination entirely — you reduce its likelihood
   (RAG, low temperature, structured outputs) and catch what slips through
   (judges, human review) before it reaches a real decision."
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 4: Transformer Architecture Primer (Handbook I19)

# COMMAND ----------

# ============================================================================
# TRANSFORMER ARCHITECTURE: Plain-English primer
# ============================================================================

print("=" * 70)
print("  TRANSFORMER ARCHITECTURE PRIMER (I19)")
print("=" * 70)
print("""
  Two ideas worth understanding:

  1. ATTENTION MECHANISM
  ──────────────────────
  When processing any word, the model looks at EVERY OTHER word and learns
  how much "attention" (relevance) to pay to each one.

  Example: "The bank raised interest rates"
                ^^^^
  To interpret "bank":
    - HIGH attention to "interest" and "rates" → FINANCIAL bank ✅
    - LOW attention to "the" → not helpful
    - If the sentence were "The bank was covered in mud" → RIVER bank

  This is computed in PARALLEL across the whole sequence (not one word
  at a time), which is WHY transformers train fast on GPUs/TPUs.

  2. NEXT-TOKEN PREDICTION
  ─────────────────────────
  An LLM is trained to do ONE task: given all text so far, predict the
  most likely NEXT token.

  "The customer wants to" → [cancel: 0.3, upgrade: 0.25, return: 0.2, ...]
                                ↑ highest probability → selected

  Generating a full response = repeating this one token at a time,
  feeding each generated token back as input for predicting the next.

  This simple training objective, at sufficient SCALE (billions of 
  parameters + massive data), produces surprisingly general-purpose
  capabilities.
""")

# COMMAND ----------

# Demonstrate attention-like behavior with embeddings
print("=" * 70)
print("  ATTENTION IN ACTION: Same word, different meanings")
print("=" * 70)

sentences = [
    "The bank approved my loan application",     # Financial bank
    "The bank of the river was muddy and steep",  # River bank
    "I need to bank on this investment",          # Trust/rely
]

# Show how context changes the embedding of the sentence
embeddings = embedding_model.encode(sentences)

# Compare all pairs
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
        print(f"\n  Similarity: {sim:.3f}")
        print(f"    A: \"{sentences[i]}\"")
        print(f"    B: \"{sentences[j]}\"")

print(f"\n  💡 Even though all sentences contain 'bank', the embeddings capture")
print(f"     the DIFFERENT meanings from context — this is attention at work!")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 5: Responsible AI (Handbook I18)

# COMMAND ----------

# ============================================================================
# RESPONSIBLE AI: Bias detection demonstration
# ============================================================================

print("=" * 70)
print("  RESPONSIBLE AI — Bias Detection (I18)")
print("=" * 70)

# Synthetic hiring data with a bias pattern
np.random.seed(42)

candidates = []
for i in range(200):
    group = np.random.choice(['Group A', 'Group B'])
    
    # Both groups have the same qualifications distribution
    years_exp = np.random.randint(2, 15)
    skill_score = np.random.uniform(60, 100)
    education = np.random.choice(['bachelors', 'masters', 'phd'], p=[0.5, 0.35, 0.15])
    
    # But the model learned a BIASED pattern from historical data
    if group == 'Group A':
        approved = int(skill_score > 70 and years_exp > 3)  # Fair threshold
    else:
        approved = int(skill_score > 80 and years_exp > 5)  # Higher bar for Group B — BIAS!
    
    candidates.append({
        'candidate_id': i,
        'group': group,
        'years_experience': years_exp,
        'skill_score': round(skill_score, 1),
        'education': education,
        'model_approved': approved,
    })

candidates_df = pd.DataFrame(candidates)

# Detect the bias
print(f"\n  Model approval rates by group:")
for group in ['Group A', 'Group B']:
    group_data = candidates_df[candidates_df['group'] == group]
    rate = group_data['model_approved'].mean()
    avg_score = group_data['skill_score'].mean()
    avg_exp = group_data['years_experience'].mean()
    print(f"    {group}: {rate:.1%} approved (avg score: {avg_score:.1f}, avg exp: {avg_exp:.1f} years)")

disparity = abs(
    candidates_df[candidates_df['group'] == 'Group A']['model_approved'].mean() -
    candidates_df[candidates_df['group'] == 'Group B']['model_approved'].mean()
)

print(f"\n  ⚠️ DISPARITY: {disparity:.1%} difference in approval rates")
print(f"     Despite SIMILAR average qualifications!")
print(f"     → This bias came from HISTORICAL DATA the model learned from")
print(f"     → Not an intentional rule — the model learned humans' past biases")

# COMMAND ----------

# ============================================================================
# PII MASKING: Protect sensitive data in AI pipelines
# ============================================================================

print("\n" + "=" * 70)
print("  PII MASKING (I18, B4)")
print("=" * 70)

def mask_pii(text: str) -> str:
    """
    Simulate ai_mask() — redact PII from text.
    On full Databricks: SELECT ai_mask(text, ARRAY('person', 'email', 'phone', 'ssn'))
    """
    masked = text
    # SSN
    masked = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN REDACTED]', masked)
    # Email
    masked = re.sub(r'[\w.-]+@[\w.-]+\.\w+', '[EMAIL REDACTED]', masked)
    # Phone
    masked = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE REDACTED]', masked)
    # Credit card
    masked = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CC REDACTED]', masked)
    # Names (simple pattern — real PII detection uses NER models)
    masked = re.sub(r'\b(Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+\s+[A-Z][a-z]+', '[NAME REDACTED]', masked)
    
    return masked

test_texts = [
    "Please help Mr. John Smith at john.smith@company.com, SSN 123-45-6789",
    "Call me at 555-123-4567, card number 4111 1111 1111 1111",
    "My account email is user@test.com and phone is 8005551234",
]

for text in test_texts:
    masked = mask_pii(text)
    print(f"\n  Original: {text}")
    print(f"  Masked:   {masked}")

print(f"\n📌 On full Databricks:")
print(f"   SELECT ai_mask(ticket_text, ARRAY('person', 'email', 'phone', 'ssn'))")
print(f"   → Built-in PII redaction as a SQL function")
print(f"   + Unity AI Gateway guardrails detect PII in LLM inputs/outputs automatically")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 6: The Senior Data Engineer's Role in AI (Handbook I20)

# COMMAND ----------

# ============================================================================
# I20: A DE's actual role in the AI stack — the synthesis
# ============================================================================

print("=" * 70)
print("  THE SENIOR DATA ENGINEER'S ROLE IN AI (I20)")
print("=" * 70)
print("""
  A senior DE typically owns or heavily influences:

  ┌────┬──────────────────────────────────────────────────────────────────┐
  │ #  │ RESPONSIBILITY                               │ NOTEBOOK COVERED │
  ├────┼──────────────────────────────────────────────────────────────────┤
  │ 1  │ BUILD governed Delta/Volume data that becomes │ NB 01, 03, 04   │
  │    │ an AI system's RAG knowledge base or training │                 │
  │    │ data. Standard medallion architecture work.   │                 │
  ├────┼──────────────────────────────────────────────────────────────────┤
  │ 2  │ EXPOSE safe, well-scoped TOOLS for agents —   │ NB 06           │
  │    │ Unity Catalog Functions wrapping business      │                 │
  │    │ logic/queries.                                │                 │
  ├────┼──────────────────────────────────────────────────────────────────┤
  │ 3  │ MAINTAIN Feature Store pipelines — keeping     │ NB 03           │
  │    │ feature tables fresh, point-in-time correct,  │                 │
  │    │ and backfillable.                             │                 │
  ├────┼──────────────────────────────────────────────────────────────────┤
  │ 4  │ OPERATE AI Functions and batch inference       │ NB 05, 09       │
  │    │ INSIDE Lakeflow pipelines — treating an LLM   │                 │
  │    │ call as a pipeline step with idempotency/retry.│                 │
  ├────┼──────────────────────────────────────────────────────────────────┤
  │ 5  │ MANAGE Vector Search index freshness/sync     │ NB 04           │
  │    │ as part of the pipeline dependency graph.     │                 │
  ├────┼──────────────────────────────────────────────────────────────────┤
  │ 6  │ GOVERN via Unity Catalog + Unity AI Gateway — │ NB 08           │
  │    │ register models/agents/tools, set spend caps  │                 │
  │    │ and access policies.                          │                 │
  ├────┼──────────────────────────────────────────────────────────────────┤
  │ 7  │ MONITOR quality AND cost in production —      │ NB 07, 08       │
  │    │ using system tables and MLflow traces like    │                 │
  │    │ monitoring pipeline health and cluster spend. │                 │
  └────┴──────────────────────────────────────────────────────────────────┘

  THE THROUGHLINE:
  Almost nothing here requires becoming an ML researcher — it's the SAME
  data-engineering skill set (governed pipelines, Unity Catalog, Lakeflow,
  monitoring, cost control) applied to:
    • A new class of CONSUMER (an agent, not a BI dashboard)
    • A new class of DATA (embeddings, prompts, traces)

  This reframing is usually the strongest way to answer:
  "Why should we trust a data engineer with our AI initiative?"
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 7: Interview Quick Hits (Handbook H3)

# COMMAND ----------

# ============================================================================
# INTERVIEW QUICK HITS — Rapid-fire Q&A from the handbook
# ============================================================================

print("=" * 70)
print("  INTERVIEW QUICK HITS (H3)")
print("=" * 70)

qa_pairs = [
    (
        "How do you keep an AI agent from seeing data it shouldn't?",
        "Unity Catalog governs the agent's tool/MCP connections and data access "
        "exactly like it governs a human's table access. Unity AI Gateway adds "
        "a RUNTIME policy layer on top (allow/deny/require-approval + spend caps)."
    ),
    (
        "How do you evaluate an LLM pipeline step like a unit test for ETL?",
        "MLflow 3 tracing + LLM judges/scorers, run the SAME way in development "
        "and in production monitoring, with a Review App turning human feedback "
        "into an evaluation dataset."
    ),
    (
        "What's the difference between Agent Bricks and the Agent Framework?",
        "Agent Bricks = describe the task, get auto-optimized agent (templates, "
        "auto-benchmarks). Agent Framework = full code-first control (LangGraph, "
        "custom Python). Both share MLflow/UC/AI Gateway governance underneath."
    ),
    (
        "What grounds Genie so it doesn't hallucinate business terms?",
        "Genie Ontology — continuously learned from Unity Catalog Metrics, "
        "Business Glossary, Domains, and query-lineage/popularity signals."
    ),
    (
        "When would you fine-tune instead of using RAG?",
        "RAG = 'model doesn't know our facts/data.' Fine-tuning = 'model needs "
        "to reliably adopt a specific STYLE, FORMAT, or narrow SKILL.' Fine-tuning "
        "is the LAST resort — slower and more expensive to iterate on."
    ),
    (
        "Why do agents need NEW governance beyond table permissions?",
        "Agents don't just READ data — they take ACTIONS via tools (send messages, "
        "call APIs, modify records). Governance must cover 'what can it DO' not "
        "only 'what can it see' — this is Unity AI Gateway's contextual policies."
    ),
]

for i, (q, a) in enumerate(qa_pairs, 1):
    print(f"\n  Q{i}: {q}")
    print(f"  A{i}: {a}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # ✅ Key Takeaways
# MAGIC
# MAGIC | Concept | What We Did | Handbook Section |
# MAGIC |---------|------------|------------------|
# MAGIC | Temperature | Showed effect on output randomness | I12 |
# MAGIC | System vs User prompt | Explained roles and usage | I12 |
# MAGIC | Context window | Calculated token budgets, showed strategies | I13 |
# MAGIC | Hallucination | Detected grounded vs hallucinated answers | I14 |
# MAGIC | Mitigation toolkit | Prevention (RAG, temp, structured) + Detection (judges, review) | I14 |
# MAGIC | Transformer primer | Attention mechanism, next-token prediction | I19 |
# MAGIC | Bias detection | Showed disparate approval rates in hiring | I18 |
# MAGIC | PII masking | Built ai_mask equivalent with regex | I18 |
# MAGIC | DE's role | Mapped every topic to concrete DE work | I20 |
# MAGIC | Interview Q&A | 6 rapid-fire Q&A from handbook H3 | H1-H3 |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎉 Congratulations!
# MAGIC
# MAGIC You've completed all 10 notebooks covering **every topic** in the AI on Databricks Handbook:
# MAGIC
# MAGIC | Notebook | Topics |
# MAGIC |----------|--------|
# MAGIC | 01 | A1-A3: Foundations, ML lifecycle |
# MAGIC | 02 | A4, I4, I16: MLflow tracking, registry, aliases, prompts |
# MAGIC | 03 | A5, I1-I3: Feature Store, AutoML, Feature Serving |
# MAGIC | 04 | A6-A8, B1-B3, B5, I5: Embeddings, RAG, Vector Search, hybrid |
# MAGIC | 05 | B4, I6, I9: AI Functions, structured outputs, multimodal |
# MAGIC | 06 | A9, D1-D5: Agents, tools, ReAct, MCP, multi-agent |
# MAGIC | 07 | E1-E2, I17: MLflow 3, tracing, judges, RAG evaluation |
# MAGIC | 08 | A10, E3-E5, I15, F1-F4, G1-G2: Governance, cost, Genie, products |
# MAGIC | 09 | C1-C5, I7-I8: Serving, batch inference, streaming AI |
# MAGIC | 10 | I12-I14, I18-I20, H1-H3: LLM internals, responsible AI, DE role |
