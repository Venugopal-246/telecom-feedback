from typing import Optional, List
from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
 text: str


class AnalyzeResponse(BaseModel):
 sentiment: str
confidence: float
intensity: float
keywords: List[str]
urgency: bool
emotion: Optional[str] = None


class FeedbackCreate(BaseModel):
 customer_id: Optional[str] = None
 age_group: Optional[str] = None
 name: Optional[str] = None
 gender: Optional[str] = None
 location: Optional[str] = None
 tenure_months: Optional[int] = None
 service_type: Optional[str] = None
 text: str

class FeedbackResponse(AnalyzeResponse):
 
 id: int


class ReportResponse(BaseModel):
 totals: dict
 by_service: dict
 top_pain_points: list
 top_positives: list
 urgent_count: int
 recommendations: list