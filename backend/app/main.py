from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select
from typing import List
from collections import Counter

from .db import engine, init_db
from .models import Feedback
from .schemas import AnalyzeRequest, AnalyzeResponse, FeedbackCreate, FeedbackResponse, ReportResponse
from .nlp import analyze_sentiment, extract_keywords, detect_urgency, detect_emotion


# ---------------------------------------------------
# FastAPI app
# ---------------------------------------------------
app = FastAPI(title="VOICERS Sentiment API", version="1.0.0")

# Allow CORS for frontend (React, Live Server, etc.)
origins = [
    "http://127.0.0.1:5500",  # VS Code Live Server
    "http://localhost:5500"
   
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------
# Startup event
# ---------------------------------------------------
@app.on_event("startup")
def on_startup():
    init_db()


# ---------------------------------------------------
# Database session dependency
# ---------------------------------------------------
def get_session():
    with Session(engine) as session:
        yield session


# ---------------------------------------------------
# Analyze API
# ---------------------------------------------------
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    label, conf, intensity = analyze_sentiment(req.text)
    kws = extract_keywords(req.text)
    urgency = detect_urgency(req.text)
    emotion = detect_emotion(req.text)

    return AnalyzeResponse(
        sentiment=label,
        confidence=round(conf, 4),
        intensity=round(float(intensity), 4),
        keywords=kws,
        urgency=urgency,
        emotion=emotion or None,
    )


# ---------------------------------------------------
# Feedback API
# ---------------------------------------------------
@app.post("/feedback", response_model=FeedbackResponse)
def create_feedback(feedback: FeedbackCreate, session: Session = Depends(get_session)):
    label, conf, intensity = analyze_sentiment(feedback.text)
    kws = extract_keywords(feedback.text)
    kws_str = ",".join(kws) if kws else None
    urgency = detect_urgency(feedback.text)
    emotion = detect_emotion(feedback.text)

    fb = Feedback(
        customer_id=feedback.customer_id,
        name=feedback.name,
        age_group=feedback.age_group,
        gender=feedback.gender,
        location=feedback.location,
        tenure_months=feedback.tenure_months,
        service_type=feedback.service_type,
        text=feedback.text,
        sentiment=label,
        confidence=conf,
        intensity=intensity,
        keywords=kws_str,
        urgency=urgency,
        emotion=emotion,
    )

    session.add(fb)
    session.commit()
    session.refresh(fb)

    return FeedbackResponse.from_orm(fb)


# ---------------------------------------------------
# Reports API
# ---------------------------------------------------
@app.get("/report", response_model=ReportResponse)
def get_reports(session: Session = Depends(get_session)):
    feedbacks = session.exec(select(Feedback)).all()
    if not feedbacks:
        return ReportResponse(
            totals={
                 "total_feedback": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            },
            by_service={},
            top_pain_points=[],
            top_positives=[],
            urgent_count=0,
            recommendations=[]
        )

    # ------------------------------
    # Totals (overall sentiment count)
    # ------------------------------
    totals = {
        "total_feedback": len(feedbacks),
        "positive": sum(1 for f in feedbacks if f.sentiment.lower() == "positive"),
        "negative": sum(1 for f in feedbacks if f.sentiment.lower() == "negative"),
        "neutral": sum(1 for f in feedbacks if f.sentiment.lower() == "neutral"),
    }

    # ------------------------------
    # By service type
    # ------------------------------
    by_service = {}
    for f in feedbacks:
        svc = f.service_type or "Unknown"
        if svc not in by_service:
            by_service[svc] = {"positive": 0, "negative": 0, "neutral": 0}
        by_service[svc][f.sentiment.lower()] += 1

    # ------------------------------
    # Top pain points (negative keywords)
    # ------------------------------
    neg_keywords = []
    for f in feedbacks:
        if f.sentiment.lower() == "negative" and f.keywords:
            neg_keywords.extend(f.keywords or [])
    top_pain_points = [kw for kw, _ in Counter(neg_keywords).most_common(5)]

    # ------------------------------
    # Top positives (positive keywords)
    # ------------------------------
    pos_keywords = []
    for f in feedbacks:
        if f.sentiment.lower() == "positive" and f.keywords:
            pos_keywords.extend(f.keywords)
    top_positives = [kw for kw, _ in Counter(pos_keywords).most_common(5)]

    # ------------------------------
    # Urgent feedback count
    # ------------------------------
    urgent_count = sum(1 for f in feedbacks if f.urgency is True)

    # ------------------------------
    # Recommendations
    # ------------------------------
    recommendations = []
    if top_pain_points:
        recommendations.append(f"Work on improving {', '.join(top_pain_points)}.")
    if urgent_count > 0:
        recommendations.append(f"Address {urgent_count} urgent customer complaints quickly.")
    if top_positives:
        recommendations.append(f"Promote strengths such as {', '.join(top_positives)}.")

    return ReportResponse(
        totals=totals,
        by_service=by_service,
        top_pain_points=top_pain_points,
        top_positives=top_positives,
        urgent_count=urgent_count,
        recommendations=recommendations,
    )
