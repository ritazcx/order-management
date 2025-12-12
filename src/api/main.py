from fastapi import FastAPI
from pydantic import BaseModel
from src.inference.predictor import Predictor
from src.api.schema import TicketRequest, TicketResponse
import uvicorn

app = FastAPI(
    title="Order Management ML API",
    version="1.0.0",
    description="ML models for category & severity classification"
)

# Load predictor (loads the newest model from registry.json)
predictor = Predictor()

# ----------- API Endpoints -----------

@app.get("/healthz")
def health_check():
    return {"status": "ok"}

@app.get("/version")
def get_version():
    return predictor.get_model_version()

@app.post("/predict", response_model=TicketResponse)
def predict_ticket(req: TicketRequest):

    # input validation
    if not req.text or len(req.text.strip()) == 0:
        return {"error": "Ticket text cannot be empty."}
    pred = predictor.predict(req.text)

    return TicketResponse(
        category=pred["category"],
        severity=pred["severity"],
        confidence=pred["confidence"]
    )

# ----------- Local Run -----------
if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
