from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from src.inference.predictor import Predictor
from src.api.schema import TicketRequest, TicketResponse
import uvicorn

app = FastAPI(
    title="Order Management ML API",
    version="1.0.0",
    description="ML models for category & severity classification"
)

# ----------- event -----------
@app.on_event("startup")
def startup_event():
    print("Starting up the Order Management ML API...") 
    try:
        app.state.predictor = Predictor()
        print("Predictor loaded successfully.")
    except Exception as e:
        app.state.predictor = None
        print(f"Error loading predictor: {e}")

@app.on_event("shutdown")
def shutdown_event():
    print("Shutting down the Order Management ML API...") 
    app.state.predictor = None

# ----------- API Endpoints -----------

@app.get("/healthz")
def health_check():
    return {"status": "ok"}

@app.get("/version")
def get_version(request: Request):
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not loaded")
    return predictor.get_model_version()

@app.post("/predict", response_model=TicketResponse)
def predict_ticket(req: TicketRequest, request: Request):

    # input validation
    if not req.text or len(req.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not loaded")

    pred = predictor.predict(req.text)

    return TicketResponse(
        category=pred["category"],
        severity=pred["severity"],
        confidence=pred["confidence"]
    )

# ----------- Local Run -----------
if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
