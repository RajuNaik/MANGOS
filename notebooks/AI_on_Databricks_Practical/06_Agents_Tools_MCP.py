# Databricks notebook source

# MAGIC %md
# MAGIC # 🕵️ Notebook 06 — Agents, Tool Calling & MCP
# MAGIC
# MAGIC **Handbook Sections Covered**: A9 (Agent vs Chatbot), D1 (Tool Calling), D2 (MCP), D3 (Mosaic AI Agent Framework), D4 (Agent Bricks), D5 (Multi-agent / Supervisor)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Learning Objectives
# MAGIC 1. **Agents vs Chatbots** — understand the fundamental difference (tools + reasoning loop)
# MAGIC 2. **Tool calling** — define tools with schemas, implement function calling
# MAGIC 3. **ReAct loop** — build a reason-act-observe loop from scratch
# MAGIC 4. **MCP (Model Context Protocol)** — understand the open standard for tool integration
# MAGIC 5. **Multi-agent / Supervisor** — build a 2-agent system with routing
# MAGIC 6. **Agent Bricks vs Agent Framework** — understand when to use each

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
from datetime import datetime, timedelta
import mlflow
import warnings
warnings.filterwarnings('ignore')

mlflow.set_experiment("/Users/{}/AI_Handbook_06_Agents".format(
    spark.sql("SELECT current_user()").collect()[0][0]
))

print("✅ Setup complete!")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 1: Agent vs Chatbot (Handbook A9)
# MAGIC
# MAGIC | | Chatbot | Agent |
# MAGIC |---|---|---|
# MAGIC | Input/Output | Text in → Text out (one shot) | Text in → (may call tools) → Text out |
# MAGIC | Actions | Can only generate text | Can call tools (query DB, send email, look up data) |
# MAGIC | Reasoning | Single pass | **Loop**: reason → act → observe → repeat |
# MAGIC | Example | "Tell me the weather if you know it" | "Let me **look up** the weather for you" |

# COMMAND ----------

# ============================================================================
# CHATBOT vs AGENT: The fundamental difference
# ============================================================================

# A CHATBOT: takes input, produces output — one shot, no tools
def chatbot_response(user_input: str) -> str:
    """A plain chatbot — can only respond with text, cannot take actions."""
    responses = {
        "exchange rate": "I don't have access to live exchange rates. Please check xe.com.",
        "order status": "I can't look up order information. Please check your email for tracking details.",
        "account balance": "I don't have access to account systems. Please log in to check your balance.",
    }
    for keyword, response in responses.items():
        if keyword in user_input.lower():
            return response
    return "I can answer general questions, but I can't look anything up for you."

# An AGENT: has tools and can reason in a loop
print("=" * 70)
print("  CHATBOT vs AGENT")
print("=" * 70)
print(f"\n  User: 'What is the current USD to EUR exchange rate?'")
print(f"\n  🤖 CHATBOT response:")
print(f"     {chatbot_response('What is the current USD to EUR exchange rate?')}")
print(f"\n  🕵️ AGENT would:")
print(f"     1. REASON: 'I need to look up the exchange rate'")
print(f"     2. ACT: Call tool `get_exchange_rate(from='USD', to='EUR')`")
print(f"     3. OBSERVE: Tool returns 0.92")
print(f"     4. REASON: 'I now have the answer'")
print(f"     5. RESPOND: 'The current USD to EUR exchange rate is 0.92'")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 2: Building Tools (Handbook D1)
# MAGIC
# MAGIC Tools are functions the agent can call. On Databricks, these are often
# MAGIC **Unity Catalog Functions** — same governance as tables.
# MAGIC
# MAGIC Each tool has:
# MAGIC 1. A **name** and **description** (the LLM reads this to decide when to use it)
# MAGIC 2. A **parameter schema** (what arguments it takes)
# MAGIC 3. An **implementation** (the actual code that runs)

# COMMAND ----------

# ============================================================================
# SYNTHETIC DATA: Customer database for agent tools
# ============================================================================
np.random.seed(2026)

customers_agent = pd.DataFrame({
    'customer_id': [f'CUST_{i:05d}' for i in range(100)],
    'name': [f"{np.random.choice(['Alice','Bob','Carol','David','Eve','Frank','Grace','Henry'])} "
             f"{np.random.choice(['Smith','Johnson','Williams','Brown','Jones','Garcia','Miller'])}"
             for _ in range(100)],
    'email': [f"user{i}@example.com" for i in range(100)],
    'plan': np.random.choice(['free', 'basic', 'premium', 'enterprise'], 100, p=[0.3, 0.3, 0.25, 0.15]),
    'monthly_spend': np.round(np.random.lognormal(3.5, 0.8, 100), 2),
    'account_status': np.random.choice(['active', 'suspended', 'cancelled'], 100, p=[0.8, 0.1, 0.1]),
    'open_tickets': np.random.poisson(1.5, 100),
    'satisfaction_score': np.round(np.random.uniform(1, 5, 100), 1),
})

orders_agent = []
for _, cust in customers_agent.iterrows():
    n_orders = np.random.poisson(5) + 1
    for j in range(n_orders):
        orders_agent.append({
            'order_id': f'ORD_{len(orders_agent):06d}',
            'customer_id': cust['customer_id'],
            'product': np.random.choice(['Widget Pro', 'Gadget Plus', 'Sensor Ultra', 'Module Basic', 'Kit Advanced']),
            'amount': round(np.random.lognormal(3, 0.7), 2),
            'status': np.random.choice(['delivered', 'shipped', 'processing', 'returned'], p=[0.6, 0.15, 0.15, 0.1]),
            'order_date': (datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 365))).strftime('%Y-%m-%d'),
        })
orders_agent = pd.DataFrame(orders_agent)

# Save to Delta
spark.createDataFrame(customers_agent).write.format("delta").mode("overwrite").saveAsTable("default.agent_customers")
spark.createDataFrame(orders_agent).write.format("delta").mode("overwrite").saveAsTable("default.agent_orders")

print(f"✅ Agent data: {len(customers_agent)} customers, {len(orders_agent)} orders")

# COMMAND ----------

# ============================================================================
# TOOL DEFINITIONS: Each tool has a name, description, parameters, and implementation
# ============================================================================

# Tool registry — the agent reads these descriptions to decide which tool to use
TOOL_DEFINITIONS = [
    {
        "name": "lookup_customer",
        "description": "Look up a customer's account details by customer ID. Returns name, plan, status, spend, and satisfaction score.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "The customer ID, e.g. 'CUST_00001'"}
            },
            "required": ["customer_id"]
        }
    },
    {
        "name": "get_order_history",
        "description": "Get a customer's recent order history. Returns list of orders with product, amount, status, and date.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "The customer ID"},
                "limit": {"type": "integer", "description": "Max number of orders to return (default 5)"}
            },
            "required": ["customer_id"]
        }
    },
    {
        "name": "calculate_metric",
        "description": "Calculate a business metric. Supported metrics: 'total_revenue', 'avg_satisfaction', 'churn_rate', 'active_customers'.",
        "parameters": {
            "type": "object",
            "properties": {
                "metric_name": {"type": "string", "description": "The metric to calculate"}
            },
            "required": ["metric_name"]
        }
    },
    {
        "name": "search_knowledge_base",
        "description": "Search the company knowledge base for information about policies, features, or troubleshooting. Use this when the user asks 'how do I...' or 'what is the policy for...' questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"]
        }
    },
]

# Tool implementations
def tool_lookup_customer(customer_id: str) -> dict:
    row = customers_agent[customers_agent['customer_id'] == customer_id]
    if row.empty:
        return {"error": f"Customer {customer_id} not found"}
    r = row.iloc[0]
    return {
        "customer_id": r['customer_id'], "name": r['name'], "email": r['email'],
        "plan": r['plan'], "monthly_spend": float(r['monthly_spend']),
        "account_status": r['account_status'], "open_tickets": int(r['open_tickets']),
        "satisfaction_score": float(r['satisfaction_score'])
    }

def tool_get_order_history(customer_id: str, limit: int = 5) -> list:
    orders = orders_agent[orders_agent['customer_id'] == customer_id].head(limit)
    if orders.empty:
        return {"error": f"No orders found for {customer_id}"}
    return orders.to_dict('records')

def tool_calculate_metric(metric_name: str) -> dict:
    metrics = {
        "total_revenue": {"value": float(orders_agent['amount'].sum()), "unit": "USD"},
        "avg_satisfaction": {"value": float(customers_agent['satisfaction_score'].mean()), "unit": "out of 5"},
        "churn_rate": {"value": float((customers_agent['account_status'] == 'cancelled').mean()), "unit": "percentage"},
        "active_customers": {"value": int((customers_agent['account_status'] == 'active').sum()), "unit": "count"},
    }
    if metric_name not in metrics:
        return {"error": f"Unknown metric: {metric_name}. Available: {list(metrics.keys())}"}
    return {"metric": metric_name, **metrics[metric_name]}

def tool_search_knowledge_base(query: str) -> dict:
    # Simplified KB search (full version in Notebook 04)
    kb_snippets = {
        "cancel": "To cancel, go to Settings > Billing > Cancel Subscription. Active until end of billing period. Data retained 90 days.",
        "refund": "Refunds available if cancelled within 14 days of billing cycle. Email billing@acmecorp.com.",
        "password": "Reset via email: go to login page > Forgot Password. Link valid for 1 hour.",
        "api": "API uses Bearer token auth. Rate limits: Free=100/hr, Basic=1000/hr, Premium=10000/hr.",
        "upgrade": "Go to Settings > Billing > Change Plan. Prorated billing for mid-cycle upgrades.",
    }
    for keyword, info in kb_snippets.items():
        if keyword in query.lower():
            return {"result": info, "source": "help_center"}
    return {"result": "No relevant information found. Please contact support@acmecorp.com.", "source": "fallback"}

TOOL_IMPLEMENTATIONS = {
    "lookup_customer": tool_lookup_customer,
    "get_order_history": tool_get_order_history,
    "calculate_metric": tool_calculate_metric,
    "search_knowledge_base": tool_search_knowledge_base,
}

print("✅ 4 tools defined and implemented:")
for tool in TOOL_DEFINITIONS:
    print(f"   🔧 {tool['name']}: {tool['description'][:60]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 3: The ReAct Loop — Building an Agent from Scratch (Handbook A9, D1)
# MAGIC
# MAGIC **ReAct** = **Re**ason + **Act**: the agent's core loop:
# MAGIC 1. **REASON** — analyze the request, decide what to do next
# MAGIC 2. **ACT** — call a tool
# MAGIC 3. **OBSERVE** — look at the tool's result
# MAGIC 4. **REPEAT** or produce a **FINAL ANSWER**

# COMMAND ----------

# ============================================================================
# REACT AGENT: Built from scratch — no LangChain, no abstractions
# This shows you exactly what's happening under the hood
# ============================================================================

class SimpleReActAgent:
    """
    A minimal ReAct agent built from scratch.
    On full Databricks, you'd use the Mosaic AI Agent Framework (D3) or Agent Bricks (D4).
    """
    
    def __init__(self, tools: dict, tool_definitions: list, max_iterations: int = 5):
        self.tools = tools
        self.tool_definitions = tool_definitions
        self.max_iterations = max_iterations
        self.trace = []  # MLflow tracing would capture this automatically
    
    def _reason(self, user_query: str, history: list) -> dict:
        """
        Determine what tool to call next (or produce a final answer).
        In a real agent, this is the LLM deciding — here we use pattern matching
        to simulate the LLM's reasoning.
        """
        query_lower = user_query.lower()
        
        # Check if we already have enough context to answer
        if history:
            last_result = history[-1].get('result', {})
            # If we have tool results, check if we need more info
            if len(history) >= 2:
                return {"action": "final_answer", "thought": "I have enough information to answer."}
        
        # Decide which tool to call based on the query
        if 'customer' in query_lower and any(f'CUST_' in query_lower for _ in [1]):
            # Extract customer ID
            match = re.search(r'CUST_\d+', user_query)
            if match:
                cid = match.group(0)
                if not any(h.get('tool') == 'lookup_customer' for h in history):
                    return {
                        "action": "tool_call",
                        "tool": "lookup_customer",
                        "args": {"customer_id": cid},
                        "thought": f"I need to look up the customer details for {cid}"
                    }
                elif 'order' in query_lower and not any(h.get('tool') == 'get_order_history' for h in history):
                    return {
                        "action": "tool_call",
                        "tool": "get_order_history",
                        "args": {"customer_id": cid, "limit": 5},
                        "thought": f"I found the customer. Now I need their order history."
                    }
        
        if any(m in query_lower for m in ['revenue', 'satisfaction', 'churn', 'active']):
            metric = 'total_revenue' if 'revenue' in query_lower else \
                     'avg_satisfaction' if 'satisfaction' in query_lower else \
                     'churn_rate' if 'churn' in query_lower else 'active_customers'
            if not any(h.get('tool') == 'calculate_metric' for h in history):
                return {
                    "action": "tool_call",
                    "tool": "calculate_metric",
                    "args": {"metric_name": metric},
                    "thought": f"I need to calculate the {metric} metric"
                }
        
        if any(w in query_lower for w in ['cancel', 'refund', 'password', 'api', 'upgrade', 'how do']):
            if not any(h.get('tool') == 'search_knowledge_base' for h in history):
                return {
                    "action": "tool_call",
                    "tool": "search_knowledge_base",
                    "args": {"query": user_query},
                    "thought": "I should search the knowledge base for this policy/how-to question"
                }
        
        return {"action": "final_answer", "thought": "I can answer based on available information."}
    
    def _act(self, tool_name: str, args: dict) -> dict:
        """Execute a tool call."""
        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}
        return self.tools[tool_name](**args)
    
    def _format_answer(self, user_query: str, history: list) -> str:
        """Compose a final answer from the gathered information."""
        parts = [f"Based on the information I gathered:\n"]
        for step in history:
            result = step.get('result', {})
            if isinstance(result, dict):
                if 'error' in result:
                    parts.append(f"⚠️ {result['error']}")
                elif 'name' in result:
                    parts.append(f"👤 Customer: {result['name']} ({result['customer_id']})")
                    parts.append(f"   Plan: {result['plan']} | Status: {result['account_status']}")
                    parts.append(f"   Monthly spend: ${result.get('monthly_spend', 0):.2f}")
                    parts.append(f"   Satisfaction: {result.get('satisfaction_score', 'N/A')}/5")
                elif 'metric' in result:
                    parts.append(f"📊 {result['metric']}: {result['value']} ({result['unit']})")
                elif 'result' in result:
                    parts.append(f"📚 {result['result']}")
            elif isinstance(result, list):
                parts.append(f"📦 Recent orders:")
                for order in result[:3]:
                    parts.append(f"   {order['order_id']}: {order['product']} - ${order['amount']:.2f} ({order['status']})")
        return "\n".join(parts)
    
    def run(self, user_query: str) -> dict:
        """Execute the full ReAct loop."""
        self.trace = []
        history = []
        
        print(f"\n{'━' * 70}")
        print(f"  🕵️ AGENT PROCESSING: \"{user_query}\"")
        print(f"{'━' * 70}")
        
        for iteration in range(self.max_iterations):
            # REASON
            decision = self._reason(user_query, history)
            step = {"iteration": iteration + 1, "thought": decision["thought"]}
            
            print(f"\n  Step {iteration + 1}:")
            print(f"    💭 THINK: {decision['thought']}")
            
            if decision["action"] == "final_answer":
                answer = self._format_answer(user_query, history)
                step["action"] = "final_answer"
                step["answer"] = answer
                self.trace.append(step)
                print(f"    ✅ ANSWER:")
                for line in answer.split('\n'):
                    print(f"       {line}")
                return {"answer": answer, "trace": self.trace, "iterations": iteration + 1}
            
            # ACT
            tool_name = decision["tool"]
            args = decision["args"]
            print(f"    🔧 ACT: {tool_name}({json.dumps(args)})")
            
            result = self._act(tool_name, args)
            step["tool"] = tool_name
            step["args"] = args
            step["result"] = result
            
            # OBSERVE
            print(f"    👁️ OBSERVE: {json.dumps(result)[:100]}...")
            
            history.append(step)
            self.trace.append(step)
        
        # Max iterations reached
        answer = self._format_answer(user_query, history)
        return {"answer": answer, "trace": self.trace, "iterations": self.max_iterations}

# Create the agent
agent = SimpleReActAgent(
    tools=TOOL_IMPLEMENTATIONS,
    tool_definitions=TOOL_DEFINITIONS,
)

# COMMAND ----------

# Run the agent on different types of queries
queries = [
    "Tell me about customer CUST_00005 and their recent orders",
    "What is our total revenue?",
    "How do I cancel my subscription and get a refund?",
    "What is the churn rate?",
]

all_results = []
for query in queries:
    result = agent.run(query)
    all_results.append(result)
    print(f"\n  📊 Completed in {result['iterations']} iteration(s), {len(result['trace'])} trace steps")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 4: MCP — Model Context Protocol (Handbook D2)
# MAGIC
# MAGIC MCP is an **open standard** for how agents connect to external tools/systems.
# MAGIC Before MCP, every agent framework built bespoke integrations. MCP defines one
# MAGIC common protocol so tools are **plug-and-play** across any MCP-compatible agent.

# COMMAND ----------

# ============================================================================
# MCP CONCEPT: Standardized tool interface
# ============================================================================

class MCPToolServer:
    """
    Simulates an MCP-compatible tool server.
    On Databricks: MCP servers are managed through Unity Catalog —
    "which external systems can this agent touch" is governed like table access.
    """
    
    def __init__(self, server_name: str, description: str):
        self.server_name = server_name
        self.description = description
        self.tools = {}
    
    def register_tool(self, name: str, description: str, handler, parameters: dict):
        """Register a tool following MCP's tool definition format."""
        self.tools[name] = {
            "name": name,
            "description": description,
            "inputSchema": parameters,
            "handler": handler,
        }
    
    def list_tools(self) -> list:
        """MCP standard: list available tools."""
        return [{"name": t["name"], "description": t["description"], 
                 "inputSchema": t["inputSchema"]} for t in self.tools.values()]
    
    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """MCP standard: execute a tool call."""
        if tool_name not in self.tools:
            return {"error": f"Tool '{tool_name}' not found on server '{self.server_name}'"}
        return self.tools[tool_name]["handler"](**arguments)

# Create an MCP server for our customer tools
crm_server = MCPToolServer("acme-crm", "AcmeCorp CRM system — customer and order data")
crm_server.register_tool(
    "lookup_customer", "Look up customer details",
    tool_lookup_customer, 
    {"type": "object", "properties": {"customer_id": {"type": "string"}}}
)
crm_server.register_tool(
    "get_order_history", "Get customer order history",
    tool_get_order_history,
    {"type": "object", "properties": {"customer_id": {"type": "string"}, "limit": {"type": "integer"}}}
)

print("=" * 70)
print("  MCP TOOL SERVER — Standardized tool interface")
print("=" * 70)
print(f"\n  Server: {crm_server.server_name}")
print(f"  Description: {crm_server.description}")
print(f"\n  Available tools (via MCP list_tools):")
for tool in crm_server.list_tools():
    print(f"    🔧 {tool['name']}: {tool['description']}")

print(f"\n  Calling via MCP call_tool:")
result = crm_server.call_tool("lookup_customer", {"customer_id": "CUST_00010"})
print(f"    Result: {json.dumps(result, indent=2)[:200]}")

print(f"\n📌 On full Databricks:")
print(f"   - MCP servers registered in Unity Catalog (governed like tables)")
print(f"   - Agent Bricks has an MCP Tool Discovery Catalog (D4)")
print(f"   - Credentials managed through Unity Catalog (not embedded in code)")
print(f"   - Unity AI Gateway applies policies to MCP tool calls (E3)")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 5: Multi-Agent / Supervisor Pattern (Handbook D5)
# MAGIC
# MAGIC When tasks get complex, a SINGLE agent becomes unreliable. The **Supervisor pattern**
# MAGIC uses a top-level agent that **routes** requests to specialist sub-agents.

# COMMAND ----------

# ============================================================================
# MULTI-AGENT SYSTEM: Supervisor + Specialists
# ============================================================================

class SpecialistAgent:
    """A specialist sub-agent focused on one domain."""
    
    def __init__(self, name: str, domain: str, tools: dict):
        self.name = name
        self.domain = domain
        self.tools = tools
    
    def handle(self, query: str) -> dict:
        """Process a query within this agent's domain."""
        results = []
        query_lower = query.lower()
        
        for tool_name, tool_fn in self.tools.items():
            # Simple heuristic — a real agent would use LLM reasoning
            if tool_name == "lookup_customer" and "CUST_" in query:
                cid = re.search(r'CUST_\d+', query).group(0)
                results.append(tool_fn(cid))
            elif tool_name == "get_order_history" and ("order" in query_lower) and "CUST_" in query:
                cid = re.search(r'CUST_\d+', query).group(0)
                results.append(tool_fn(cid))
            elif tool_name == "calculate_metric":
                for metric in ['revenue', 'satisfaction', 'churn', 'active']:
                    if metric in query_lower:
                        results.append(tool_fn(metric.replace('revenue', 'total_revenue')
                                              .replace('satisfaction', 'avg_satisfaction')
                                              .replace('churn', 'churn_rate')
                                              .replace('active', 'active_customers')))
            elif tool_name == "search_knowledge_base":
                results.append(tool_fn(query))
        
        return {"agent": self.name, "domain": self.domain, "results": results}


class SupervisorAgent:
    """
    Top-level supervisor that routes requests to the right specialist.
    This mirrors Agent Bricks' Supervisor template (D4).
    """
    
    def __init__(self):
        self.specialists = {}
    
    def register_specialist(self, agent: SpecialistAgent):
        self.specialists[agent.domain] = agent
    
    def route(self, query: str) -> str:
        """Determine which specialist should handle this query."""
        query_lower = query.lower()
        if any(w in query_lower for w in ['customer', 'account', 'order', 'cust_']):
            return "customer_ops"
        elif any(w in query_lower for w in ['revenue', 'satisfaction', 'churn', 'metric', 'active']):
            return "analytics"
        elif any(w in query_lower for w in ['cancel', 'refund', 'password', 'how', 'policy', 'upgrade']):
            return "knowledge"
        return "knowledge"  # default
    
    def handle(self, query: str) -> dict:
        """Route and delegate to the right specialist."""
        domain = self.route(query)
        
        print(f"  🎯 SUPERVISOR: Routing to '{domain}' specialist")
        
        if domain in self.specialists:
            result = self.specialists[domain].handle(query)
            return {"supervisor_routing": domain, "specialist_response": result}
        return {"error": f"No specialist for domain '{domain}'"}

# Build the multi-agent system
customer_agent = SpecialistAgent("CustomerOps", "customer_ops", {
    "lookup_customer": tool_lookup_customer,
    "get_order_history": tool_get_order_history,
})

analytics_agent = SpecialistAgent("Analytics", "analytics", {
    "calculate_metric": tool_calculate_metric,
})

knowledge_agent = SpecialistAgent("Knowledge", "knowledge", {
    "search_knowledge_base": tool_search_knowledge_base,
})

supervisor = SupervisorAgent()
supervisor.register_specialist(customer_agent)
supervisor.register_specialist(analytics_agent)
supervisor.register_specialist(knowledge_agent)

# Test the multi-agent system
print("=" * 70)
print("  MULTI-AGENT SUPERVISOR SYSTEM")
print("=" * 70)

multi_queries = [
    "Look up customer CUST_00042 and their orders",
    "What is our churn rate?",
    "How do I reset my password?",
]

for query in multi_queries:
    print(f"\n  📥 Query: \"{query}\"")
    result = supervisor.handle(query)
    specialist = result.get('specialist_response', {})
    print(f"  📤 Handled by: {specialist.get('agent', 'unknown')}")
    for r in specialist.get('results', []):
        print(f"     Result: {json.dumps(r)[:100]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 6: Agent Bricks vs Mosaic AI Agent Framework (Handbook D3 vs D4)

# COMMAND ----------

# ============================================================================
# COMPARISON: Agent Bricks vs Mosaic AI Agent Framework
# ============================================================================

print("=" * 70)
print("  AGENT BRICKS vs MOSAIC AI AGENT FRAMEWORK")
print("=" * 70)
print("""
  ┌───────────────────────────────────────────────────────────────────┐
  │  AGENT BRICKS (D4)                                                │
  │  "Describe the task, get an auto-optimized agent"                 │
  │                                                                   │
  │  ✅ Low-code, template-based                                     │
  │  ✅ Auto-generates synthetic data & benchmarks                    │
  │  ✅ Auto-creates LLM judges for evaluation                       │
  │  ✅ Auto-optimizes: prompts, model choice, RAG settings           │
  │  ✅ Can fine-tune smaller custom models automatically              │
  │                                                                   │
  │  Templates:                                                       │
  │    - Knowledge Assistant (RAG chatbot)                            │
  │    - Information Extraction (docs → structured data)              │
  │    - Supervisor (multi-agent routing)                             │
  │    - Custom LLM / Custom Agent                                    │
  │                                                                   │
  │  USE WHEN: Standard use case, want fast results, OK with less     │
  │  control over the internals                                       │
  ├───────────────────────────────────────────────────────────────────┤
  │  MOSAIC AI AGENT FRAMEWORK (D3)                                   │
  │  "Full code-first control, bring your own framework"              │
  │                                                                   │
  │  ✅ Use LangChain, LangGraph, or fully custom Python             │
  │  ✅ Full control over reasoning loop, tool selection, prompts     │
  │  ✅ Same MLflow/Unity Catalog governance as Agent Bricks           │
  │  ✅ Define custom tools (UC Functions + MCP)                       │
  │                                                                   │
  │  Lifecycle: Log (MLflow) → Register (UC) → Deploy (Model Serving) │
  │  Same as ANY model — the agent IS a model from the platform's POV │
  │                                                                   │
  │  USE WHEN: Need custom control, Agent Bricks templates don't fit,  │
  │  building novel agent architectures                                │
  └───────────────────────────────────────────────────────────────────┘

  BOTH paths share:
  - MLflow for tracking/tracing
  - Unity Catalog for governance
  - Model Serving for deployment
  - Unity AI Gateway for runtime governance
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # ✅ Key Takeaways
# MAGIC
# MAGIC | Concept | What We Did | Handbook Section |
# MAGIC |---------|------------|------------------|
# MAGIC | Agent vs Chatbot | Showed one-shot vs tool-calling + reasoning loop | A9 |
# MAGIC | Tool Calling | Defined 4 tools with schemas and implementations | D1 |
# MAGIC | ReAct Loop | Built reason-act-observe loop from scratch | A9, D1 |
# MAGIC | MCP | Created an MCP-compatible tool server | D2 |
# MAGIC | Multi-agent Supervisor | Built supervisor routing to 3 specialists | D5 |
# MAGIC | Agent Bricks vs Framework | Compared high-level vs code-first approaches | D3, D4 |
# MAGIC
# MAGIC ### 🔗 Full Platform Mapping
# MAGIC | CE Approach | Full Databricks |
# MAGIC |---|---|
# MAGIC | Python class agent | Mosaic AI Agent Framework (D3) with LangGraph/custom |
# MAGIC | Simulated tool registry | Unity Catalog Functions (governed, auditable) |
# MAGIC | Python MCP server class | MCP servers managed through Unity Catalog |
# MAGIC | Pattern-matching routing | LLM-powered reasoning with real Foundation Models |
# MAGIC | Manual multi-agent code | Agent Bricks Supervisor template (D4) |
# MAGIC
# MAGIC **Next**: Notebook 07 — MLflow 3 for GenAI: Tracing, LLM Judges & Evaluation.
