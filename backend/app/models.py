from typing import Optional
from sqlmodel import Field, SQLModel
from datetime import datetime


class Feedback(SQLModel, table=True):
	id: Optional[int] = Field(default=None, primary_key=True)
	customer_id: Optional[str] = None
	name: Optional[str] = None
	age_group: Optional[str] = None
	gender: Optional[str] = None
	location: Optional[str] = None
	tenure_months: Optional[int] = None
	service_type: Optional[str] = None # Internet, Billing, Support, Streaming, etc.
	text: str

	# Analysis outputs
	sentiment: str
	confidence: float
	intensity: float # [-1..+1]
	keywords: Optional[str] # comma-separated
	urgency: Optional[bool]
	emotion: Optional[str]

	created_at: datetime = Field(default_factory=datetime.utcnow)