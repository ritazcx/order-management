# AI Ticket Automation System — n8n Workflow Architecture

This document explains the **n8n workflow architecture** for the enterprise-grade AI Ticket Automation System.
It is intended for **non-technical stakeholders**, reviewers, and interviewers.

---

## 1. System Purpose

The n8n workflow orchestrates how incoming tickets are:
- Classified by ML models
- Routed by business logic
- Enhanced by LLMs when necessary
- Automatically responded or escalated
- Logged for auditing and KPI tracking

n8n acts as the **central decision engine**.

---

## 2. High-Level Workflow Diagram

flowchart TD
    A[Incoming Ticket<br/>(Email / Form / API)] --> B[n8n Webhook<br/>Receive Ticket]

    B --> C[Call ML Prediction API<br/>(FastAPI)]

    C --> D{Decision Engine<br/>Severity / Rules}

    D -->|Low / Medium| E[Template / Rule-based Reply]

    D -->|High / Complex| F[LLM Response Generation]

    F --> G[Optional Human Review]

    E --> H[Send Response<br/>(Email / Slack / System)]
    G --> H

    H --> I[Log Ticket & Metrics<br/>(DB / File / Dashboard)]

```text
┌────────────────────────────┐
│ Incoming Ticket            │
│ (Email / Form / API)       │
└─────────────┬──────────────┘
              ↓
┌────────────────────────────┐
│ Webhook (n8n)              │
│ Receive ticket payload     │
└─────────────┬──────────────┘
              ↓
┌────────────────────────────┐
│ Call ML Service (FastAPI)  │
│ - Category prediction     │
│ - Severity prediction     │
└─────────────┬──────────────┘
              ↓
┌────────────────────────────┐
│ Decision Logic (Switch)    │
│ Route by category          │
└─────────────┬──────────────┘
              ↓
┌────────────────────────────┐
│ Severity Check (IF)        │
│ High / Medium / Low        │
└─────────────┬──────────────┘
      ┌───────┴────────┐
      ↓                ↓
┌──────────────┐   ┌────────────────┐
│ Direct Reply │   │ Call LLM        │
│ (Template)   │   │ (GPT / Claude)  │
└──────┬───────┘   └───────┬────────┘
       ↓                   ↓
┌──────────────────────────────────┐
│ Optional Human Review             │
│ (Slack / Email Approval)          │
└─────────────┬────────────────────┘
              ↓
┌────────────────────────────┐
│ Send Response               │
│ Email / Ticket System       │
└─────────────┬──────────────┘
              ↓
┌────────────────────────────┐
│ Log & Metrics               │
│ DB / Sheet / Monitoring     │
└────────────────────────────┘
```

---

## 3. Node-by-Node Explanation

### 3.1 Webhook — Receive Ticket
**Purpose:** Entry point of the system  
**Input:**
```json
{
  "ticket_id": "12345",
  "title": "VPN cannot connect",
  "description": "VPN fails after OS update",
  "user": "employee@company.com"
}
```

---

### 3.2 HTTP Request — Call ML Model
Calls your FastAPI service.

**Endpoint:**  
`POST /predict`

**Returns:**
```json
{
  "category": "Network Issue",
  "severity": "High",
  "confidence": 0.82,
  "needs_llm": true
}
```

---

### 3.3 Switch — Category Routing
Routes tickets based on category:
- Network Issue
- Access / Permission
- Hardware
- Software Bug
- Email / Office

Each category can have:
- Different SLA
- Different templates
- Different escalation paths

---

### 3.4 IF — Severity Decision
Business rules example:
- **Low:** auto-reply
- **Medium:** LLM-generated reply
- **High:** LLM + alert + escalation

---

### 3.5 LLM Node — Response Generation
Used only when needed.

**Prompt Inputs:**
- Ticket description
- Category
- Severity
- Company policy snippets

**Outputs:**
- Explanation
- Troubleshooting steps
- Professional response draft

---

### 3.6 Human Review (Optional)
Used for:
- High-risk tickets
- Early-stage rollout
- Compliance reasons

Approval channels:
- Slack
- Email
- Internal dashboard

---

### 3.7 Send Response
Automatically sends:
- Email reply
- Ticket system update
- Slack notification

---

### 3.8 Logging & Metrics
Stores:
- Prediction results
- LLM usage
- Response time
- SLA compliance

Used for:
- KPI dashboards
- Model evaluation
- Audits

---

## 4. Why This Architecture Is Enterprise-Grade

- ML is deterministic and fast
- LLM usage is controlled and cost-efficient
- n8n provides visibility and control
- Easy to extend and debug
- Matches real enterprise workflows

---

## 5. Next Implementation Steps

1. Build minimal n8n workflow:
   - Webhook → ML → IF → Response
2. Add LLM node only for Medium/High severity
3. Add logging
4. Iterate with real tickets

---

## 6. How to Present This in Interviews

> “I designed and implemented an enterprise AI ticket automation system using a hybrid ML + LLM architecture, orchestrated via n8n for reliability, cost control, and observability.”

This is a **strong production-ready story**.
