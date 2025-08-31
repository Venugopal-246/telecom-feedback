import re
from typing import List, Tuple


import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import yake
import json


# --- Sentiment (3-class) ---
# cardiffnlp supports Negative/Neutral/Positive
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_pipe = pipeline("text-classification", model=SENTIMENT_MODEL, tokenizer=SENTIMENT_MODEL, return_all_scores=True, truncation=True)


# --- Emotion (optional) ---
try:
	EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
	emotion_pipe = pipeline("text-classification", model=EMOTION_MODEL, return_all_scores=False, truncation=True)
except Exception:
	emotion_pipe = None


# --- Keyword Extractor (YAKE) ---
yk = yake.KeywordExtractor(lan="en", n=1, top=6)


URGENCY_PATTERNS = re.compile(r"(urgent|immediately|asap|right away|not working since|down since|2 days|two days|emergency|can\'t access|unable to)", re.I)


LABEL_MAP = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}




def analyze_sentiment(text: str) -> Tuple[str, float, float]:
	"""Return (label, confidence, intensity)
	intensity mapped to [-1..+1] using class probs.
	"""
	scores = sentiment_pipe(text)[0] # list of dicts with score per class
	# scores example: [{'label': 'LABEL_0', 'score': 0.1}, ...]
	probs = {LABEL_MAP.get(s['label'], s['label']): s['score'] for s in scores}
	label = max(probs, key=probs.get)
	confidence = float(probs[label])
	# intensity = P(Positive) - P(Negative)
	intensity = float(probs.get("Positive", 0.0) - probs.get("Negative", 0.0))
	return label, confidence, intensity




def extract_keywords(text: str) -> List[str]:
	try:
		kws = [k for k, _ in yk.extract_keywords(text)]
		return kws[:5]
	except Exception:
		return []




def detect_urgency(text: str) -> bool:
 return bool(URGENCY_PATTERNS.search(text))


def detect_emotion(text: str) -> str:
    """
    Basic emotion detection from text.
    Returns one of: joy, sadness, anger, fear, neutral
    """
    text = text.lower()

    if any(word in text for word in ["happy", "joy", "excited", "glad", "great", "love"]):
        return "joy"
    elif any(word in text for word in ["sad", "unhappy", "depressed", "cry", "bad"]):
        return "sadness"
    elif any(word in text for word in ["angry", "mad", "furious", "rage", "hate"]):
        return "anger"
    elif any(word in text for word in ["fear", "scared", "afraid", "nervous", "worried"]):
        return "fear"
    else:
        return "neutral"


    return ""