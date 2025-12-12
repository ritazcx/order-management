from pydantic import BaseModel

class TicketRequest(BaseModel):
    text: str

class TicketResponse(BaseModel):
    category: str
    severity: str
    confidence: float
    
class ModelVersionResponse(BaseModel):  
    category_model: str
    severity_model: str
    version: str    