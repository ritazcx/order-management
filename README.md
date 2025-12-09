# order-management
An enterprise-grade AI ticket automation system that combines ML classification, LLM reasoning, and n8n workflow orchestration to automatically triage, resolve, and respond to IT support tickets end-to-end.
## System Architecture (Mermaid Diagram)

```mermaid
flowchart TD
    A[Incoming Ticket] --> B[ML Classifier]
    B --> C{n8n Orchestrator}
    C --> D[LLM Engine]
    C --> F[Category Sub-Flows]
    D --> E[Human Review]
    E --> G[Auto Reply]
    F --> G
    G --> H[DB Logging]
