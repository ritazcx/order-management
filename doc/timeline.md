# AI Ticket Automation — Project Timeline Gantt Chart

## Timeline Overview
- **Day 1**: Data creation
- **Day 2**: Category model
- **Day 3**: Severity model (baseline + hybrid)
- **Day 4**: FastAPI backend
- **Day 5**: Deployment + Docker
- **Day 6**: n8n workflow + LLM
- **Day 7**: QA, docs, launch

---

## Gantt Chart

| Task | Day 1 | Day 2 | Day 3 | Day 4 | Day 5 | Day 6 | Day 7 |
|------|-------|-------|-------|-------|-------|-------|-------|
| **Data: Create 300–400 synthetic tickets** | ████ | ░ | ░ | | | | |
| **Data: Review category/severity distribution** | ███ | ░ | | | | | |
| **Model: Train Category Classifier** | | █████ | ░ | | | | |
| **Model: Evaluate + improve** | | ███ | ░ | | | | |
| **Model: Train Severity Baseline** | | | ████ | ░ | | | |
| **Model: Add Hybrid Rules for Severity** | | | ███ | ░ | | | |
| **Backend: Build FastAPI inference service** | | | | █████ | ░ | | |
| **Backend: Add hybrid severity logic endpoint** | | | | ███ | ░ | | |
| **Backend: Dockerize service** | | | | | ████ | ░ | |
| **Backend: Deploy to cloud (Render/Railway)** | | | | | ███ | ░ | |
| **n8n: Create Webhook + API call to FastAPI** | | | | | | ████ | |
| **n8n: Add branching logic by category** | | | | | | ███ | |
| **n8n: LLM generation (troubleshooting + replies)** | | | | | | ███ | |
| **n8n: Optional human approval (Slack/Telegram)** | | | | | | █ | |
| **QA: End-to-end testing** | | | | | | | ███ |
| **Docs: README + architecture diagram** | | | | | | | ███ |
| **Launch: Write announcement (LinkedIn/X/小红书)** | | | | | | | ██ |

---

## Checklist (Mark as Done)

### Day 1 — Data
- [X] Create 300–400 synthetic tickets  
- [X] Validate categories  
- [X] Validate severity rules  

### Day 2 — Category Model
- [X] Train classifier  
- [X] Evaluate accuracy  
- [X] Save model + vectorizer  

### Day 3 — Severity Model
- [X] Train baseline severity model  
- [X] Add severity_ruls.md, complete train_severity model (with SentenceBERT)  
- [X] Export hybrid severity classifier  

### Day 4 — Backend
- [X] Create FastAPI project  
- [X] Add `/predict` endpoint  
- [X] Integrate hybrid logic  
- [X] Test locally  

### Day 5 — Deployment
- [ ] Create Dockerfile  
- [ ] Build + run locally  
- [ ] Deploy to cloud  
- [ ] Test public endpoint  

### Day 6 — n8n Workflow
- [ ] Create Webhook  
- [ ] Call FastAPI  
- [ ] Branch by category  
- [ ] Add LLM troubleshooting  
- [ ] Add auto-reply + human approval  

### Day 7 — Polishing + Launch
- [ ] End-to-end tests  
- [ ] Documentation  
- [ ] Demo video (optional)  
- [ ] LinkedIn/X/小红书 announcement  

