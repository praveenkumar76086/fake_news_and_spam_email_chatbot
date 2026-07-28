import streamlit as st
import pandas as pd
import json
import re
import os
import joblib
from datetime import datetime


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TruthGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# BASIC UI
# ============================================================

st.title("🛡️ TruthGuard AI")
st.subheader("Fake News & Spam Email Detection Chatbot")

st.write(
    "TruthGuard AI analyzes news articles and emails using Gemini AI, "
    "trained ensemble ML fallback models, optional long-term memory, and domain-only follow-up Q&A "
    "to provide verdicts, suspicion scores, reasons, and safety advice."
)


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "sample_text" not in st.session_state:
    st.session_state.sample_text = ""

if "content_type" not in st.session_state:
    st.session_state.content_type = "News Article"

if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []


# ============================================================
# LOAD GEMINI SETTINGS - MULTIPLE API KEY SUPPORT
# ============================================================

try:
    MODEL_NAME = st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash-lite")

    GEMINI_API_KEYS = st.secrets.get("GEMINI_API_KEYS", [])

    single_key = st.secrets.get("GEMINI_API_KEY", None)

    if isinstance(GEMINI_API_KEYS, str):
        GEMINI_API_KEYS = [GEMINI_API_KEYS]
    else:
        GEMINI_API_KEYS = list(GEMINI_API_KEYS)

    if single_key and single_key not in GEMINI_API_KEYS:
        GEMINI_API_KEYS.insert(0, single_key)

    GEMINI_API_KEYS = [
        key for key in GEMINI_API_KEYS
        if key and str(key).strip()
    ]

except Exception:
    MODEL_NAME = "gemini-2.5-flash-lite"
    GEMINI_API_KEYS = []


# ============================================================
# LOAD HINDSIGHT MEMORY SETTINGS
# ============================================================

try:
    HINDSIGHT_API_KEY = st.secrets.get("HINDSIGHT_API_KEY", None)
    HINDSIGHT_BASE_URL = st.secrets.get("HINDSIGHT_BASE_URL", None)
    HINDSIGHT_BANK_ID = st.secrets.get("HINDSIGHT_BANK_ID", "truthguard-ai-final")
except Exception:
    HINDSIGHT_API_KEY = None
    HINDSIGHT_BASE_URL = None
    HINDSIGHT_BANK_ID = "truthguard-ai-final"


def hindsight_available():
    return bool(HINDSIGHT_API_KEY and HINDSIGHT_BASE_URL and HINDSIGHT_BANK_ID)


# ============================================================
# LOAD LOCAL ML MODELS
# ============================================================

@st.cache_resource
def load_ml_models():
    models = {
        "news": None,
        "email": None
    }

    if os.path.exists("fake_news_model.pkl"):
        try:
            models["news"] = joblib.load("fake_news_model.pkl")
        except Exception as e:
            st.warning(f"Could not load fake_news_model.pkl: {e}")

    if os.path.exists("spam_model.pkl"):
        try:
            models["email"] = joblib.load("spam_model.pkl")
        except Exception as e:
            st.warning(f"Could not load spam_model.pkl: {e}")

    return models


ml_models = load_ml_models()


# ============================================================
# JSON EXTRACTOR
# ============================================================

def extract_json(text):
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group())
        except Exception:
            return None

    return None


# ============================================================
# WARNING WORD DETECTOR
# ============================================================

def find_warning_words(text):
    text_lower = text.lower()

    suspicious_patterns = [
        "urgent",
        "click here",
        "free money",
        "lottery",
        "winner",
        "limited time",
        "verify your account",
        "password",
        "bank account",
        "miracle cure",
        "shocking",
        "breaking",
        "guaranteed",
        "act now",
        "claim reward",
        "congratulations",
        "risk-free",
        "share this message",
        "before it gets deleted",
        "otp",
        "prize",
        "reward",
        "account verification",
        "verify",
        "claim",
        "deleted",
        "secret remedy",
        "government is hiding",
        "immediately",
        "bank details",
        "personal information",
        "limited offer",
        "you have won",
        "cure all diseases",
        "hidden truth",
        "forward this message",
        "verify your identity",
        "account suspended",
        "unusual login",
        "confirm your details",
        "credit card",
        "act fast",
        "donate now",
        "500% returns",
        "financial freedom"
    ]

    matched = []

    for pattern in suspicious_patterns:
        if pattern in text_lower:
            matched.append(pattern)

    return list(dict.fromkeys(matched))


# ============================================================
# INPUT TYPE DETECTOR
# ============================================================

def detect_possible_input_type(text):
    text_lower = text.lower()

    email_indicators = [
        "dear",
        "regards",
        "subject",
        "click here",
        "verify your account",
        "password",
        "otp",
        "bank account",
        "claim your reward",
        "email",
        "sender",
        "inbox",
        "department",
        "please carry",
        "room",
        "congratulations",
        "account suspended",
        "verify your identity",
        "dear student",
        "dear team",
        "invoice",
        "application received",
        "meeting reminder"
    ]

    news_indicators = [
        "breaking",
        "news",
        "officials",
        "government",
        "report",
        "announced",
        "article",
        "source",
        "published",
        "doctors",
        "reserve bank",
        "policy",
        "committee",
        "inflation",
        "growth projections",
        "official statement",
        "rbi",
        "quarterly earnings",
        "revenue",
        "financial results"
    ]

    email_score = sum(1 for word in email_indicators if word in text_lower)
    news_score = sum(1 for word in news_indicators if word in text_lower)

    if email_score > news_score:
        return "Email Message"
    elif news_score > email_score:
        return "News Article"
    else:
        return "Unclear"


# ============================================================
# DOMAIN QUESTION CHECKER
# ============================================================

def is_domain_related_question(question):
    """
    Allows only questions related to:
    - fake news
    - real news verification
    - spam/phishing
    - latest analysis explanation
    - project/model behavior

    It also allows follow-up questions about specific terms present
    in the latest analyzed text, such as S-400, RBI, Microsoft, etc.
    """
    question_lower = question.lower().strip()

    blocked_keywords = [
        "love letter",
        "write a letter",
        "movie",
        "song",
        "lyrics",
        "game",
        "joke",
        "story",
        "math homework",
        "programming assignment",
        "recipe",
        "travel plan",
        "dating",
        "poem",
        "shayari"
    ]

    if any(blocked in question_lower for blocked in blocked_keywords):
        return False

    domain_keywords = [
        "fake news",
        "real news",
        "news",
        "spam",
        "phishing",
        "email",
        "safe email",
        "suspicion",
        "score",
        "risk",
        "verdict",
        "reason",
        "explanation",
        "indicator",
        "suspicious",
        "credible",
        "credibility",
        "source",
        "verify",
        "verification",
        "misinformation",
        "disinformation",
        "manipulation",
        "tone",
        "intent",
        "evidence",
        "phishing link",
        "otp",
        "password",
        "bank account",
        "click",
        "reward",
        "lottery",
        "model",
        "gemini",
        "local model",
        "ml fallback",
        "ensemble",
        "hindsight",
        "memory",
        "classified",
        "detected",
        "unsafe",
        "safe",
        "uncertain",
        "review recommended",
        "why was this",
        "why did the system",
        "which words",
        "which word",
        "how can i verify",
        "what does risk",
        "what does score"
    ]

    if any(keyword in question_lower for keyword in domain_keywords):
        return True

    # Allow follow-up questions about specific terms from latest analyzed text
    last_analysis = st.session_state.get("last_analysis", None)

    if last_analysis:
        analyzed_text = last_analysis.get("user_text", "").lower()

        # Extract important words/terms from latest analyzed text
        context_terms = re.findall(r"[a-zA-Z0-9\-]{4,}", analyzed_text)

        stop_words = {
            "this", "that", "with", "from", "have", "will", "they", "their",
            "about", "which", "where", "when", "what", "there", "these",
            "those", "been", "also", "into", "such", "only", "more",
            "than", "because", "after", "before", "using", "used"
        }

        context_terms = [
            term for term in context_terms
            if term not in stop_words
        ]

        # Also normalize question by removing punctuation except hyphen
        question_terms = re.findall(r"[a-zA-Z0-9\-]{3,}", question_lower)

        for term in question_terms:
            if term in context_terms:
                return True

        # Extra support for terms like "s400" vs "s-400"
        normalized_context = analyzed_text.replace("-", "")
        normalized_question = question_lower.replace("-", "")

        important_context_terms = re.findall(r"[a-zA-Z0-9]{4,}", normalized_context)

        for term in re.findall(r"[a-zA-Z0-9]{3,}", normalized_question):
            if term in important_context_terms:
                return True

    return False

# ============================================================
# SAMPLE INPUTS
# ============================================================

def get_sample_text(sample_type):
    samples = {
        "Spam Email": """Congratulations! You have won ₹5,00,000 in our lucky draw. Click here immediately to claim your reward. You must verify your bank account within 2 hours or the prize will expire.""",

        "Non-Spam Email": """Dear Student,

This is to inform you that your AI Essentials class test will be conducted tomorrow at 10:00 AM in Room 204. Please carry your ID card and reach the classroom 10 minutes before the test begins.

Regards,
Department of Computer Science""",

        "Fake News": """Breaking shocking news! Doctors have discovered one secret home remedy that cures all diseases in 24 hours. Government officials are hiding this truth from the public. Share this message before it gets deleted.""",

        "Real News": """The Reserve Bank of India announced its monetary policy decision after the scheduled meeting of the Monetary Policy Committee. According to the official statement published on the RBI website, the committee reviewed inflation trends, growth projections, liquidity conditions, and global economic developments before announcing the policy decision. The report also mentioned that future policy actions will depend on incoming economic data and inflation outlook."""
    }

    return samples.get(sample_type, "")


# ============================================================
# HINDSIGHT MEMORY FUNCTIONS
# ============================================================

def make_safe_memory_summary(user_text, content_type, result, analysis_mode):
    """
    Store only summarized detection patterns.
    Do not store full private email/news text.
    """
    verdict = result.get("verdict", "Unknown")
    score = result.get("suspicion_score", 0)
    risk = result.get("risk_level", "Unknown")
    reasons = result.get("key_reasons", [])

    warning_words = find_warning_words(user_text)

    summary = f"""
TruthGuard AI analyzed a {content_type}.
Verdict: {verdict}.
Suspicion score: {score}/100.
Risk level: {risk}.
Analysis mode: {analysis_mode}.
Important pattern indicators: {", ".join(warning_words[:8]) if warning_words else "No major suspicious keyword pattern found"}.
Main reasons: {"; ".join(reasons[:3]) if reasons else "No reason available"}.
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.
"""

    return summary.strip()


def get_hindsight_client():
    try:
        from hindsight_client import Hindsight

        try:
            client = Hindsight(
                base_url=HINDSIGHT_BASE_URL,
                api_key=HINDSIGHT_API_KEY
            )
        except TypeError:
            client = Hindsight(
                base_url=HINDSIGHT_BASE_URL
            )

        return client, None

    except Exception as e:
        return None, str(e)


def close_hindsight_client(client):
    try:
        if hasattr(client, "close"):
            result = client.close()

            if hasattr(result, "__await__"):
                pass

    except Exception:
        pass


def recall_hindsight_memory(user_text, content_type):
    if not hindsight_available():
        return "Hindsight memory is not configured."

    client, error = get_hindsight_client()

    if error:
        return f"Hindsight client creation failed: {error}"

    try:
        query = f"""
Find previous TruthGuard AI memories similar to this {content_type}.
Focus on suspicious patterns, verdicts, risk levels, phishing indicators, spam signs, and fake news indicators.
Do not return private full text.
Current text pattern preview: {user_text[:500]}
"""

        memories = client.recall(
            bank_id=HINDSIGHT_BANK_ID,
            query=query
        )

        close_hindsight_client(client)

        if not memories:
            return "No similar previous memory found."

        cleaned_memories = []
        raw_memory_text = str(memories)

        extracted_texts = re.findall(
            r"text='(.*?)'(?:,|\))",
            raw_memory_text,
            flags=re.DOTALL
        )

        if not extracted_texts:
            extracted_texts = re.findall(
                r'text="(.*?)"(?:,|\))',
                raw_memory_text,
                flags=re.DOTALL
            )

        if not extracted_texts:
            if hasattr(memories, "results"):
                memory_items = memories.results
            else:
                memory_items = memories if isinstance(memories, list) else []

            for memory in memory_items:
                if hasattr(memory, "text"):
                    extracted_texts.append(memory.text)
                elif hasattr(memory, "content"):
                    extracted_texts.append(memory.content)

        for memory_text in extracted_texts[:5]:
            memory_text = memory_text.replace("\\n", " ")
            memory_text = memory_text.replace("\\'", "'")
            memory_text = memory_text.replace('\\"', '"')
            memory_text = re.sub(r"\s+", " ", memory_text).strip()

            unwanted_phrases = [
                "team rule",
                "api route handlers",
                "business logic moved to services",
                "route handlers should be thin"
            ]

            if any(phrase in memory_text.lower() for phrase in unwanted_phrases):
                continue

            if len(memory_text) > 450:
                memory_text = memory_text[:450] + "..."

            if memory_text:
                cleaned_memories.append(memory_text)

        if not cleaned_memories:
            return "No directly relevant previous TruthGuard memory found."

        final_output = "The system found similar previous analysis patterns:\n\n"

        for index, memory_text in enumerate(cleaned_memories[:3], start=1):
            final_output += f"- **Case {index}:** {memory_text}\n\n"

        final_output += (
            "\nThese memories are used only for comparison. "
            "The final decision is still based on the current input analysis."
        )

        return final_output.strip()

    except Exception as e:
        close_hindsight_client(client)
        return f"Hindsight recall failed: {e}"


def retain_hindsight_memory(user_text, content_type, result, analysis_mode):
    if not hindsight_available():
        return "Hindsight memory is not configured."

    client, error = get_hindsight_client()

    if error:
        return f"Hindsight client creation failed: {error}"

    try:
        safe_summary = make_safe_memory_summary(
            user_text=user_text,
            content_type=content_type,
            result=result,
            analysis_mode=analysis_mode
        )

        try:
            client.retain(
                bank_id=HINDSIGHT_BANK_ID,
                content=safe_summary
            )
        except TypeError:
            client.retain(
                bank_id=HINDSIGHT_BANK_ID,
                text=safe_summary
            )

        close_hindsight_client(client)

        return "Memory saved successfully."

    except Exception as e:
        close_hindsight_client(client)
        return f"Hindsight retain failed: {e}"


# ============================================================
# GEMINI ANALYSIS FUNCTION WITH MULTI-KEY FAILOVER
# ============================================================

def analyze_with_gemini(user_text, content_type, temperature, top_p):
    if not GEMINI_API_KEYS:
        return {
            "error": True,
            "message": "No Gemini API key found. Please add GEMINI_API_KEYS in secrets.toml."
        }

    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        return {
            "error": True,
            "message": f"Gemini library import failed: {e}"
        }

    system_instruction = """
You are TruthGuard AI, a professional domain-specific AI assistant for detecting fake news, spam emails, and phishing messages.

Your role is not only to classify the text, but also to explain the reasoning in a clear, structured, professional, and responsible way.

You must analyze:
1. Content authenticity
2. Language tone
3. Suspicious words or phrases
4. Credibility signals
5. Manipulation techniques
6. User safety risks
7. Possible intent behind the text
8. Evidence quality
9. Risk level

Rules:
1. If the input is a news article, classify it as Likely Real News, Likely Fake News, or Unclear.
2. If the input is an email, classify it as Safe Email, Likely Spam, Possible Phishing, or Unclear.
3. Do not claim 100% certainty.
4. Use professional but easy-to-understand language.
5. Return ONLY valid JSON.
6. Do not use markdown.
7. Do not wrap the response inside ```json.
8. The suspicion score must be between 0 and 100.
9. A low score means low suspicion. A high score means high suspicion.
10. Give strong explanations, not generic lines.

Required JSON format:
{
  "verdict": "Likely Fake News / Likely Real News / Likely Spam / Possible Phishing / Safe Email / Unclear",
  "suspicion_score": 0,
  "risk_level": "Low / Medium / High",
  "key_reasons": [
    "clear reason 1",
    "clear reason 2",
    "clear reason 3"
  ],
  "suspicious_indicators": [
    "suspicious word, phrase, claim, or pattern 1",
    "suspicious word, phrase, claim, or pattern 2"
  ],
  "credibility_indicators": [
    "credible or safe signal 1",
    "credible or safe signal 2"
  ],
  "manipulation_patterns": [
    "urgency / fear / reward bait / emotional pressure / conspiracy framing / unsupported claim"
  ],
  "language_tone_analysis": "Explain whether the tone is professional, sensational, urgent, emotional, manipulative, or neutral.",
  "intent_analysis": "Explain the likely purpose of the message or article.",
  "evidence_quality": "Explain whether the text gives verifiable facts, official source, dates, author, links, or supporting evidence.",
  "source_reliability": "Explain whether the source appears reliable, weak, unknown, or needs verification.",
  "professional_explanation": "Give a detailed professional explanation of why this verdict and score were assigned.",
  "safety_advice": [
    "practical safety advice 1",
    "practical safety advice 2",
    "practical safety advice 3"
  ]
}
"""

    prompt = f"""
Content Type: {content_type}

Analyze this text:

{user_text}
"""

    error_messages = []

    for index, api_key in enumerate(GEMINI_API_KEYS, start=1):
        try:
            client = genai.Client(api_key=api_key)

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=1500,
                    response_mime_type="application/json"
                )
            )

            return {
                "error": False,
                "text": response.text,
                "key_used": index,
                "key_label": f"Gemini Key {index}"
            }

        except Exception as e:
            error_messages.append(f"Gemini Key {index} failed: {str(e)}")
            continue

    return {
        "error": True,
        "message": "All Gemini API keys failed.\n\n" + "\n\n".join(error_messages)
    }


# ============================================================
# DOMAIN-ONLY FOLLOW-UP Q&A FUNCTION
# ============================================================

def answer_domain_question_with_gemini(question):
    """
    Answers only domain-related questions using Gemini API failover.
    Uses latest analysis context if available.
    """
    if not GEMINI_API_KEYS:
        return {
            "error": True,
            "message": "No Gemini API key available. Follow-up Q&A requires Gemini API."
        }

    if not is_domain_related_question(question):
        return {
            "error": False,
            "answer": (
                "I can only answer questions related to fake news detection, spam/phishing emails, "
                "suspicion score, risk level, analysis explanation, verification, and this project."
            ),
            "key_label": "Domain Guard"
        }

    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        return {
            "error": True,
            "message": f"Gemini library import failed: {e}"
        }

    last_analysis = st.session_state.get("last_analysis", None)

    if last_analysis:
        context_text = f"""
Latest Analysis Context:
Content Type: {last_analysis.get("content_type")}
Analysis Mode: {last_analysis.get("analysis_mode")}
Original Text Preview: {last_analysis.get("user_text", "")[:1200]}

Analysis Result:
{json.dumps(last_analysis.get("result", {}), indent=2)}
"""
    else:
        context_text = """
No previous analysis is available. Answer only as a general domain assistant for fake news, spam, phishing, and verification guidance.
"""

    system_instruction = """
You are TruthGuard AI Follow-up Assistant.

You must answer ONLY questions related to:
1. Fake news detection
2. Real news verification
3. Spam email detection
4. Phishing detection
5. Suspicion score
6. Risk level
7. Verdict explanation
8. Suspicious words or indicators
9. Credibility indicators
10. Safety advice
11. Gemini/API analysis
12. Local ML fallback and ensemble model
13. Hindsight memory used in this project

Rules:
1. If the question is outside this domain, politely refuse.
2. Do not answer unrelated questions.
3. If the user asks about the latest result, use the provided analysis context.
4. Explain clearly and professionally.
5. Keep the answer practical and easy to understand.
6. Do not invent facts that are not present in the context.
7. If the answer requires checking live websites, tell the user to verify from trusted sources.
8. Do not claim 100% certainty.
"""

    prompt = f"""
{context_text}

User Question:
{question}

Give a helpful domain-specific answer.
"""

    error_messages = []

    for index, api_key in enumerate(GEMINI_API_KEYS, start=1):
        try:
            client = genai.Client(api_key=api_key)

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                    top_p=0.8,
                    max_output_tokens=900
                )
            )

            return {
                "error": False,
                "answer": response.text,
                "key_label": f"Gemini Key {index}"
            }

        except Exception as e:
            error_messages.append(f"Gemini Key {index} failed: {str(e)}")
            continue

    return {
        "error": True,
        "message": "All Gemini API keys failed for follow-up Q&A.\n\n" + "\n\n".join(error_messages)
    }


# ============================================================
# ENSEMBLE ML HELPER FUNCTIONS
# ============================================================

def get_positive_class_probability_from_model(model, user_text, positive_class=1):
    """
    Returns probability of positive/suspicious class from a single model.
    Works with sklearn pipelines that support predict_proba().
    """
    try:
        probabilities = model.predict_proba([user_text])[0]
        classes = list(model.classes_)

        if positive_class in classes:
            positive_index = classes.index(positive_class)
            return float(probabilities[positive_index])

        if str(positive_class) in classes:
            positive_index = classes.index(str(positive_class))
            return float(probabilities[positive_index])

        return float(max(probabilities))

    except Exception:
        return None


def get_ensemble_probability(model_object, user_text, positive_class=1):
    """
    Supports:
    1. New ensemble format:
       {
           "model_format": "ensemble_v1",
           "models": {...},
           "threshold": ...
       }

    2. Old direct sklearn Pipeline format.
    """
    if isinstance(model_object, dict) and model_object.get("model_format") == "ensemble_v1":
        models = model_object.get("models", {})
        probs = []

        for model_name, model in models.items():
            prob = get_positive_class_probability_from_model(
                model=model,
                user_text=user_text,
                positive_class=positive_class
            )

            if prob is not None:
                probs.append(prob)

        if len(probs) == 0:
            return None, model_object.get("threshold", 0.5), "Ensemble model probability failed"

        final_prob = sum(probs) / len(probs)
        threshold = model_object.get("threshold", 0.5)
        model_type = model_object.get(
            "model_type",
            "TF-IDF Ensemble: Logistic Regression + Linear SVM + Naive Bayes"
        )

        return final_prob, threshold, model_type

    prob = get_positive_class_probability_from_model(
        model=model_object,
        user_text=user_text,
        positive_class=positive_class
    )

    return prob, 0.5, "TF-IDF Logistic Regression"


def adjust_ml_suspicion_score(raw_score, user_text, content_type):
    """
    Calibration layer for local ML fallback.
    Reduces false positives for normal business/academic emails and trusted news.
    Avoids unrealistic 0/100 confidence.
    """
    text_lower = user_text.lower()
    adjusted_score = raw_score

    trusted_news_indicators = [
        "official statement",
        "published on",
        "reserve bank of india",
        "rbi website",
        "monetary policy committee",
        "according to",
        "scheduled meeting",
        "policy decision",
        "inflation outlook",
        "growth projections",
        "official website",
        "microsoft",
        "quarterly earnings",
        "revenue",
        "cloud services",
        "financial performance",
        "company reported",
        "earnings report",
        "publicly traded company",
        "official filing",
        "sec filing",
        "press release",
        "reported revenue",
        "increase in revenue",
        "business update",
        "financial results"
    ]

    safe_email_indicators = [
        "dear team",
        "dear student",
        "hello",
        "regards",
        "sincerely",
        "thank you",
        "department",
        "computer science",
        "university",
        "class test",
        "room",
        "please carry your id card",
        "conducted tomorrow",
        "10:00 am",
        "meeting reminder",
        "meeting agenda",
        "scheduled meeting",
        "project discussion",
        "invoice attached",
        "please find attached",
        "attached invoice",
        "payment receipt",
        "application received",
        "thank you for applying",
        "job application",
        "interview schedule",
        "event invitation",
        "university event",
        "workshop",
        "seminar",
        "registered successfully"
    ]

    fake_news_indicators = [
        "shocking",
        "breaking shocking",
        "secret remedy",
        "cure all diseases",
        "government is hiding",
        "share this message",
        "before it gets deleted",
        "miracle cure",
        "hidden truth",
        "rumors spread",
        "without evidence",
        "viral claim",
        "no official confirmation",
        "secret plan",
        "doctors hate this",
        "they don't want you to know"
    ]

    strong_spam_indicators = [
        "click here",
        "verify your account",
        "bank account",
        "credit card",
        "password",
        "otp",
        "claim your reward",
        "you have won",
        "winner",
        "lottery",
        "limited time",
        "act fast",
        "prize will expire",
        "within 2 hours",
        "account suspended",
        "urgent account suspension",
        "confirm your identity",
        "donate now",
        "fake charity",
        "investment",
        "500% returns",
        "financial freedom"
    ]

    medium_spam_indicators = [
        "password reset",
        "reset your password",
        "invoice",
        "attached",
        "account",
        "application",
        "payment",
        "limited offer"
    ]

    if content_type == "News Article":
        trusted_hits = sum(1 for phrase in trusted_news_indicators if phrase in text_lower)
        fake_hits = sum(1 for phrase in fake_news_indicators if phrase in text_lower)

        adjusted_score -= trusted_hits * 8
        adjusted_score += fake_hits * 8

    elif content_type == "Email Message":
        safe_hits = sum(1 for phrase in safe_email_indicators if phrase in text_lower)
        strong_spam_hits = sum(1 for phrase in strong_spam_indicators if phrase in text_lower)
        medium_spam_hits = sum(1 for phrase in medium_spam_indicators if phrase in text_lower)

        adjusted_score -= safe_hits * 10
        adjusted_score += strong_spam_hits * 8
        adjusted_score += medium_spam_hits * 2

    adjusted_score = max(0, min(100, int(adjusted_score)))

    if adjusted_score <= 3:
        adjusted_score = 5
    elif adjusted_score >= 97:
        adjusted_score = 95

    return adjusted_score


# ============================================================
# LOCAL ADVANCED EXPLANATION
# ============================================================

def generate_local_advanced_explanation(
    user_text,
    content_type,
    suspicion_score,
    verdict,
    risk_level,
    model_type
):
    """
    Generates detailed explanation for local ML fallback.
    """
    text_lower = user_text.lower()
    suspicious_words = find_warning_words(user_text)

    trusted_news_terms = [
        "official statement",
        "published on",
        "reserve bank of india",
        "rbi website",
        "monetary policy committee",
        "according to",
        "scheduled meeting",
        "policy decision",
        "press release",
        "financial results",
        "quarterly earnings",
        "reported revenue",
        "official website"
    ]

    safe_email_terms = [
        "dear student",
        "dear team",
        "hello",
        "regards",
        "sincerely",
        "thank you",
        "department",
        "class test",
        "room",
        "meeting reminder",
        "invoice attached",
        "please find attached",
        "application received",
        "university event",
        "interview schedule"
    ]

    fake_news_patterns = [
        "shocking",
        "secret remedy",
        "cure all diseases",
        "government is hiding",
        "share this message",
        "before it gets deleted",
        "miracle cure",
        "hidden truth",
        "without evidence",
        "viral claim"
    ]

    phishing_patterns = [
        "click here",
        "verify your account",
        "bank account",
        "credit card",
        "password",
        "otp",
        "claim your reward",
        "you have won",
        "lottery",
        "winner",
        "account suspended",
        "confirm your identity",
        "act fast"
    ]

    if content_type == "News Article":
        credibility_hits = [
            term for term in trusted_news_terms
            if term in text_lower
        ]

        manipulation_hits = [
            term for term in fake_news_patterns
            if term in text_lower
        ]

        if manipulation_hits:
            tone = (
                "The text contains sensational or emotionally persuasive wording. "
                "Such language is commonly found in misleading or low-credibility news content."
            )
        elif credibility_hits:
            tone = (
                "The tone appears formal and information-based. The presence of official or verifiable terms "
                "reduces suspicion."
            )
        else:
            tone = (
                "The tone is not strongly suspicious, but the local model could not find enough trusted-source indicators."
            )

        if suspicion_score >= 70:
            intent = (
                "The article may be attempting to influence the reader using suspicious or unsupported claims."
            )
        elif suspicion_score >= 40:
            intent = (
                "The article requires verification because the local model found mixed signals."
            )
        else:
            intent = (
                "The article appears mostly informational and does not show strong fake-news patterns."
            )

        evidence_quality = (
            "The local model checks text patterns only. It cannot independently verify facts from live sources. "
            "Therefore, official source, author, date, and supporting evidence should still be checked manually."
        )

        professional_explanation = (
            f"The local fallback used {model_type} to analyze TF-IDF word patterns in the news text. "
            f"The final suspicion score is {suspicion_score}/100. "
            f"The verdict is {verdict}. "
            "The score was influenced by suspicious expressions, credibility indicators, and the model's learned patterns. "
            "Because local ML is less context-aware than Gemini, medium scores are treated as uncertain and should be verified manually."
        )

        return {
            "suspicious_indicators": list(dict.fromkeys(suspicious_words + manipulation_hits)),
            "credibility_indicators": credibility_hits,
            "manipulation_patterns": manipulation_hits,
            "language_tone_analysis": tone,
            "intent_analysis": intent,
            "evidence_quality": evidence_quality,
            "source_reliability": (
                "Source reliability cannot be fully confirmed by the local model. "
                "Trusted official references should be checked."
            ),
            "professional_explanation": professional_explanation
        }

    elif content_type == "Email Message":
        safe_hits = [
            term for term in safe_email_terms
            if term in text_lower
        ]

        phishing_hits = [
            term for term in phishing_patterns
            if term in text_lower
        ]

        if phishing_hits:
            tone = (
                "The email contains urgency, reward bait, account verification, or sensitive information patterns. "
                "These are common indicators of spam or phishing."
            )
        elif safe_hits:
            tone = (
                "The email tone appears formal, professional, or academic. This reduces suspicion."
            )
        else:
            tone = (
                "The email contains mixed signals. It does not have enough safe indicators for full confidence."
            )

        if suspicion_score >= 80:
            intent = (
                "The email may be attempting to make the user click a link, reveal personal details, or respond quickly."
            )
        elif suspicion_score >= 50:
            intent = (
                "The email may be legitimate, but it contains patterns that require manual review."
            )
        else:
            intent = (
                "The email appears mostly safe and informational."
            )

        evidence_quality = (
            "The local model cannot verify sender identity, domain reputation, or real links. "
            "It only evaluates text patterns. Sender address and links should be checked manually."
        )

        professional_explanation = (
            f"The local fallback used {model_type} to analyze TF-IDF word patterns in the email. "
            f"The final suspicion score is {suspicion_score}/100. "
            f"The verdict is {verdict}. "
            "The score was influenced by suspicious keywords, safe-email indicators, and learned spam patterns. "
            "Password reset, invoice, account, and payment-related emails may be marked as review recommended because they can be legitimate or phishing depending on sender and links."
        )

        return {
            "suspicious_indicators": list(dict.fromkeys(suspicious_words + phishing_hits)),
            "credibility_indicators": safe_hits,
            "manipulation_patterns": phishing_hits,
            "language_tone_analysis": tone,
            "intent_analysis": intent,
            "evidence_quality": evidence_quality,
            "source_reliability": (
                "Sender reliability cannot be fully verified by local ML. "
                "The user should check sender email address, domain, and links."
            ),
            "professional_explanation": professional_explanation
        }

    return {
        "suspicious_indicators": suspicious_words,
        "credibility_indicators": [],
        "manipulation_patterns": [],
        "language_tone_analysis": "No tone analysis available.",
        "intent_analysis": "No intent analysis available.",
        "evidence_quality": "No evidence quality analysis available.",
        "source_reliability": "No source reliability analysis available.",
        "professional_explanation": "No professional explanation available."
    }


# ============================================================
# LOCAL ML FALLBACK ANALYSIS
# ============================================================

def analyze_with_ml_model(user_text, content_type):
    warning_words = find_warning_words(user_text)

    # ========================================================
    # NEWS ARTICLE FALLBACK
    # ========================================================
    if content_type == "News Article":
        model_object = ml_models["news"]

        if model_object is None:
            return {
                "verdict": "Fake News ML model not found",
                "suspicion_score": 0,
                "risk_level": "Unknown",
                "key_reasons": [
                    "Gemini API failed or was turned off.",
                    "fake_news_model.pkl is missing from the project folder.",
                    "The system could not perform local fake news analysis."
                ],
                "safety_advice": [
                    "Add fake_news_model.pkl to the project folder.",
                    "Restart the Streamlit app."
                ],
                "explanation": "The local fake news model is not available."
            }

        fake_probability, threshold, model_type = get_ensemble_probability(
            model_object=model_object,
            user_text=user_text,
            positive_class=1
        )

        if fake_probability is None:
            fake_probability = 0.50

        raw_score = int(fake_probability * 100)

        suspicion_score = adjust_ml_suspicion_score(
            raw_score=raw_score,
            user_text=user_text,
            content_type=content_type
        )

        if suspicion_score >= 70:
            verdict = "Likely Fake News"
            risk_level = "High"
        elif suspicion_score >= 40:
            verdict = "Uncertain / Needs Verification"
            risk_level = "Medium"
        else:
            verdict = "Likely Real News"
            risk_level = "Low"

        key_reasons = [
            "Gemini API was unavailable or turned off, so the trained local ML fallback was used.",
            f"The fallback uses {model_type}.",
            "The final suspicion score is calculated using soft-voting probability from local models when ensemble format is available.",
            "A calibration layer reduces false positives for trusted-source indicators and increases score for suspicious fake-news patterns."
        ]

        if warning_words:
            key_reasons.append(
                "Suspicious terms found: " + ", ".join(warning_words[:8])
            )

        advanced_explanation = generate_local_advanced_explanation(
            user_text=user_text,
            content_type=content_type,
            suspicion_score=suspicion_score,
            verdict=verdict,
            risk_level=risk_level,
            model_type=model_type
        )

        return {
            "verdict": verdict,
            "suspicion_score": suspicion_score,
            "risk_level": risk_level,
            "key_reasons": key_reasons,
            "safety_advice": [
                "Verify the article from trusted news sources.",
                "Check the author, date, source, and supporting evidence.",
                "Medium-risk fallback results should be treated as uncertain and manually verified."
            ],
            "explanation": (
                "The local fallback analyzed the article using TF-IDF text features. "
                "If an ensemble model is available, Logistic Regression, Linear SVM, and Naive Bayes "
                "are combined through soft voting. The score is then calibrated using trusted-source "
                "and suspicious-pattern indicators to reduce false positives."
            ),
            **advanced_explanation
        }

    # ========================================================
    # EMAIL FALLBACK
    # ========================================================
    elif content_type == "Email Message":
        model_object = ml_models["email"]

        if model_object is None:
            return {
                "verdict": "Spam Email ML model not found",
                "suspicion_score": 0,
                "risk_level": "Unknown",
                "key_reasons": [
                    "Gemini API failed or was turned off.",
                    "spam_model.pkl is missing from the project folder.",
                    "The system could not perform local spam email analysis."
                ],
                "safety_advice": [
                    "Add spam_model.pkl to the project folder.",
                    "Restart the Streamlit app."
                ],
                "explanation": "The local spam email model is not available."
            }

        spam_probability, threshold, model_type = get_ensemble_probability(
            model_object=model_object,
            user_text=user_text,
            positive_class=1
        )

        if spam_probability is None:
            spam_probability = 0.50

        raw_score = int(spam_probability * 100)

        suspicion_score = adjust_ml_suspicion_score(
            raw_score=raw_score,
            user_text=user_text,
            content_type=content_type
        )

        if suspicion_score >= 80:
            verdict = "Likely Spam / Possible Phishing"
            risk_level = "High"
        elif suspicion_score >= 50:
            verdict = "Uncertain / Review Recommended"
            risk_level = "Medium"
        else:
            verdict = "Safe Email"
            risk_level = "Low"

        key_reasons = [
            "Gemini API was unavailable or turned off, so the trained local ML fallback was used.",
            f"The fallback uses {model_type}.",
            "The final suspicion score is calculated using soft-voting probability from local models when ensemble format is available.",
            "A calibration layer reduces false positives for safe-email indicators and increases score for suspicious spam/phishing patterns."
        ]

        if warning_words:
            key_reasons.append(
                "Suspicious terms found: " + ", ".join(warning_words[:8])
            )

        advanced_explanation = generate_local_advanced_explanation(
            user_text=user_text,
            content_type=content_type,
            suspicion_score=suspicion_score,
            verdict=verdict,
            risk_level=risk_level,
            model_type=model_type
        )

        return {
            "verdict": verdict,
            "suspicion_score": suspicion_score,
            "risk_level": risk_level,
            "key_reasons": key_reasons,
            "safety_advice": [
                "Do not click suspicious links.",
                "Do not share passwords, OTP, or bank details.",
                "Medium-risk fallback results should be reviewed manually before action."
            ],
            "explanation": (
                "The local fallback analyzed the email using TF-IDF text features. "
                "If an ensemble model is available, Logistic Regression, Linear SVM, and Naive Bayes "
                "are combined through soft voting. The score is then calibrated using safe-email "
                "and suspicious-pattern indicators to reduce false positives."
            ),
            **advanced_explanation
        }

    return {
        "verdict": "Unknown content type",
        "suspicion_score": 0,
        "risk_level": "Unknown",
        "key_reasons": ["Invalid content type selected."],
        "safety_advice": ["Select News Article or Email Message."],
        "explanation": "The system could not analyze the input."
    }

def apply_security_policy_adjustment(result, user_text, content_type):
    """
    Applies final safety policy correction after Gemini or ML analysis.
    Useful for sensitive email actions like password reset, account verification,
    OTP, bank details, and clickable links.
    """
    text_lower = user_text.lower()

    verdict = result.get("verdict", "Unknown")
    score = int(result.get("suspicion_score", 0))
    risk = result.get("risk_level", "Unknown")

    key_reasons = result.get("key_reasons", [])
    safety_advice = result.get("safety_advice", [])
    suspicious_indicators = result.get("suspicious_indicators", [])

    has_url = bool(re.search(r"http[s]?://|www\.|\.com|\.in|\.org|\.net", text_lower))

    password_reset_terms = [
        "password reset",
        "reset your password",
        "reset your account password",
        "create a new password"
    ]

    account_verification_terms = [
        "verify your account",
        "confirm your identity",
        "account suspended",
        "unusual login",
        "verify your identity"
    ]

    sensitive_terms = [
        "otp",
        "bank account",
        "credit card",
        "bank details",
        "personal information",
        "login"
    ]

    has_password_reset = any(term in text_lower for term in password_reset_terms)
    has_account_verification = any(term in text_lower for term in account_verification_terms)
    has_sensitive_term = any(term in text_lower for term in sensitive_terms)

    if content_type == "Email Message":
        # Password reset with link should not be treated as completely safe.
        if has_password_reset and has_url and score < 45:
            score = 45
            verdict = "Uncertain / Review Recommended"
            risk = "Medium"

            key_reasons.append(
                "Password reset email contains a link. Even if it looks legitimate, it should be verified before clicking."
            )

            suspicious_indicators.append("password reset link")

            safety_advice.append(
                "Open the official website manually instead of clicking the password reset link directly."
            )

        # Account verification / sensitive action should be at least medium risk.
        if (has_account_verification or has_sensitive_term) and has_url and score < 55:
            score = 55
            verdict = "Uncertain / Review Recommended"
            risk = "Medium"

            key_reasons.append(
                "The email contains account verification or sensitive information patterns with a link."
            )

            suspicious_indicators.append("sensitive account action with link")

            safety_advice.append(
                "Do not provide OTP, password, bank details, or personal information through email links."
            )

    result["verdict"] = verdict
    result["suspicion_score"] = score
    result["risk_level"] = risk
    result["key_reasons"] = list(dict.fromkeys(key_reasons))
    result["safety_advice"] = list(dict.fromkeys(safety_advice))
    result["suspicious_indicators"] = list(dict.fromkeys(suspicious_indicators))

    return result

# ============================================================
# DISPLAY RESULT FUNCTION
# ============================================================

def display_result(result, analysis_mode, user_text, content_type):
    final_score = int(result.get("suspicion_score", 0))
    final_score = max(0, min(final_score, 100))

    final_risk = result.get("risk_level", "Unknown")
    verdict = result.get("verdict", "Unknown")

    # ========================================================
    # TOP RESULT METRICS
    # ========================================================
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Verdict", verdict)

    with col2:
        st.metric("Suspicion Score", f"{final_score}/100")

    with col3:
        st.metric("Risk Level", final_risk)

    with col4:
        st.metric("Analysis Mode", analysis_mode)

    st.progress(final_score / 100)

    # ========================================================
    # RESULT SUMMARY BOX
    # ========================================================
    if final_risk == "High":
        st.error(
            f"High-risk content detected. The system classified this input as **{verdict}** "
            f"with a suspicion score of **{final_score}/100**."
        )
    elif final_risk == "Medium":
        st.warning(
            f"Moderate-risk or uncertain content detected. The system classified this input as **{verdict}** "
            f"with a suspicion score of **{final_score}/100**."
        )
    elif final_risk == "Low":
        st.success(
            f"Low-risk content detected. The system classified this input as **{verdict}** "
            f"with a suspicion score of **{final_score}/100**."
        )
    else:
        st.info(
            f"The system classified this input as **{verdict}** "
            f"with a suspicion score of **{final_score}/100**."
        )

    if analysis_mode == "Local ML Fallback":
        st.caption(
            "Note: Local ML fallback is based on TF-IDF word patterns and may be less context-aware than Gemini. "
            "Medium-risk fallback results are treated as uncertain and should be verified manually."
        )

    # ========================================================
    # KEY REASONS
    # ========================================================
    st.markdown("### Key Reasons")

    reasons = result.get("key_reasons", [])

    if reasons:
        for reason in reasons:
            st.write(f"- {reason}")
    else:
        st.write("- No specific reason provided.")

    # ========================================================
    # SAFETY ADVICE
    # ========================================================
    st.markdown("### Safety Advice")

    advice_list = result.get("safety_advice", [])

    if advice_list:
        for advice in advice_list:
            st.write(f"- {advice}")
    else:
        st.write("- No safety advice provided.")

    # ========================================================
    # ADVANCED ANALYSIS BREAKDOWN
    # ========================================================
    suspicious_indicators = result.get("suspicious_indicators", [])
    credibility_indicators = result.get("credibility_indicators", [])
    manipulation_patterns = result.get("manipulation_patterns", [])

    language_tone_analysis = result.get("language_tone_analysis", "")
    intent_analysis = result.get("intent_analysis", "")
    evidence_quality = result.get("evidence_quality", "")
    source_reliability = result.get("source_reliability", "")

    has_advanced_details = any([
        suspicious_indicators,
        credibility_indicators,
        manipulation_patterns,
        language_tone_analysis,
        intent_analysis,
        evidence_quality,
        source_reliability
    ])

    if has_advanced_details:
        st.markdown("### Advanced Analysis Breakdown")

        tab1, tab2, tab3 = st.tabs([
            "Indicators",
            "Intent & Tone",
            "Credibility"
        ])

        with tab1:
            st.markdown("#### Suspicious Indicators")

            if suspicious_indicators:
                for item in suspicious_indicators:
                    st.write(f"- {item}")
            else:
                st.write("- No major suspicious indicators detected.")

            st.markdown("#### Manipulation Patterns")

            if manipulation_patterns:
                for item in manipulation_patterns:
                    st.write(f"- {item}")
            else:
                st.write("- No clear manipulation pattern detected.")

        with tab2:
            st.markdown("#### Language and Tone Analysis")
            if language_tone_analysis:
                st.write(language_tone_analysis)
            else:
                st.write("No tone analysis available.")

            st.markdown("#### Intent Analysis")
            if intent_analysis:
                st.write(intent_analysis)
            else:
                st.write("No intent analysis available.")

        with tab3:
            st.markdown("#### Credibility Indicators")

            if credibility_indicators:
                for item in credibility_indicators:
                    st.write(f"- {item}")
            else:
                st.write("- No strong credibility indicator detected.")

            st.markdown("#### Evidence Quality")
            if evidence_quality:
                st.write(evidence_quality)
            else:
                st.write("No evidence quality analysis available.")

            st.markdown("#### Source Reliability")
            if source_reliability:
                st.write(source_reliability)
            else:
                st.write("No source reliability analysis available.")

    # ========================================================
    # PROFESSIONAL EXPLANATION
    # ========================================================
    st.markdown("### Professional Explanation")

    if final_score >= 70:
        confidence_note = (
            "The system detected strong suspicious signals. "
            "This content should be treated as high risk until it is verified from trusted sources."
        )
    elif final_score >= 40:
        confidence_note = (
            "The system detected moderate suspicious signals. "
            "This result should be interpreted as uncertain and verified before taking action."
        )
    else:
        confidence_note = (
            "The system detected limited suspicious signals. "
            "However, important news claims or emails should still be verified from trusted sources."
        )

    if analysis_mode.startswith("Gemini API"):
        method_note = (
            "This result was generated using Gemini API. Gemini performed contextual language analysis "
            "to understand the meaning, tone, intent, credibility signals, suspicious patterns, and possible user safety risks."
        )
    elif analysis_mode == "Local ML Fallback":
        method_note = (
            "This result was generated using the trained local machine learning fallback. "
            "If the ensemble model is available, the fallback combines Logistic Regression, Linear SVM, "
            "and Naive Bayes using soft voting. The score may also be calibrated using suspicious and safe indicators."
        )
    else:
        method_note = "This result was generated using the available analysis pipeline."

    explanation_text = (
        result.get("professional_explanation")
        or result.get("explanation")
        or "No explanation provided."
    )

    st.info(
        f"""
**Decision Summary:**  
The system classified this input as **{verdict}** with a **{final_risk}** risk level and a suspicion score of **{final_score}/100**.

**Analysis Method:**  
{method_note}

**Reasoning:**  
{explanation_text}

**Confidence Interpretation:**  
{confidence_note}

**Recommended Action:**  
For news articles, verify the claim from trusted news sources, official websites, or multiple reliable references.  
For emails, avoid clicking unknown links and never share passwords, OTPs, bank details, or personal information.
"""
    )

    # ========================================================
    # MEMORY INSIGHT
    # ========================================================
    if hindsight_available():
        memory_insight = recall_hindsight_memory(
            user_text=user_text,
            content_type=content_type
        )

        st.markdown("### Memory Insight")

        with st.expander("View similar previous analysis patterns", expanded=False):
            st.markdown(memory_insight)

    # ========================================================
    # SAVE TO HISTORY
    # ========================================================
    st.session_state.history.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Content Type": content_type,
        "Text Preview": user_text[:60] + "..." if len(user_text) > 60 else user_text,
        "Verdict": verdict,
        "Suspicion Score": final_score,
        "Risk Level": final_risk,
        "Analysis Mode": analysis_mode
    })

    # Save latest analysis for domain-only Q&A
    st.session_state.last_analysis = {
        "content_type": content_type,
        "user_text": user_text,
        "result": result,
        "analysis_mode": analysis_mode,
        "time": datetime.now().strftime("%H:%M:%S")
    }

    # ========================================================
    # SAVE MEMORY AFTER RESULT
    # ========================================================
    if hindsight_available():
        memory_status = retain_hindsight_memory(
            user_text=user_text,
            content_type=content_type,
            result=result,
            analysis_mode=analysis_mode
        )

        st.caption(f"Hindsight Memory: {memory_status}")


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.title("⚙️ Model Settings")

content_type = st.sidebar.selectbox(
    "Select content type",
    ["News Article", "Email Message"],
    index=["News Article", "Email Message"].index(st.session_state.content_type),
    key="content_type_selector"
)

st.session_state.content_type = content_type

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.1
)

top_p = st.sidebar.slider(
    "Top-p",
    min_value=0.1,
    max_value=1.0,
    value=0.8,
    step=0.1
)

use_gemini = st.sidebar.toggle(
    "Use Gemini API",
    value=True,
    help="Turn off during testing to save Gemini quota. If off, the app uses local ML fallback directly."
)

st.sidebar.info(
    "Low temperature gives stable and factual answers. "
    "Higher temperature gives more creative but less predictable answers."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")

if GEMINI_API_KEYS:
    st.sidebar.success(f"{len(GEMINI_API_KEYS)} Gemini API key(s) loaded")
else:
    st.sidebar.error("Gemini API key missing")

if ml_models["news"] is not None:
    if isinstance(ml_models["news"], dict) and ml_models["news"].get("model_format") == "ensemble_v1":
        st.sidebar.success("Fake News Ensemble model loaded")
    else:
        st.sidebar.success("Fake News ML model loaded")
else:
    st.sidebar.warning("Fake News ML model missing")

if ml_models["email"] is not None:
    if isinstance(ml_models["email"], dict) and ml_models["email"].get("model_format") == "ensemble_v1":
        st.sidebar.success("Spam Email Ensemble model loaded")
    else:
        st.sidebar.success("Spam Email ML model loaded")
else:
    st.sidebar.warning("Spam Email ML model missing")

if hindsight_available():
    st.sidebar.success("Hindsight memory configured")
else:
    st.sidebar.warning("Hindsight memory not configured")


# ============================================================
# USER INPUT
# ============================================================

st.markdown("---")
st.markdown("## Quick Test Samples")
st.caption(
    "Use these examples to quickly test spam, safe email, fake news, and real news detection."
)

sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)

with sample_col1:
    if st.button("Sample Spam Email"):
        st.session_state.sample_text = get_sample_text("Spam Email")
        st.session_state.content_type = "Email Message"
        st.rerun()

with sample_col2:
    if st.button("Sample Non-Spam Email"):
        st.session_state.sample_text = get_sample_text("Non-Spam Email")
        st.session_state.content_type = "Email Message"
        st.rerun()

with sample_col3:
    if st.button("Sample Fake News"):
        st.session_state.sample_text = get_sample_text("Fake News")
        st.session_state.content_type = "News Article"
        st.rerun()

with sample_col4:
    if st.button("Sample Real News"):
        st.session_state.sample_text = get_sample_text("Real News")
        st.session_state.content_type = "News Article"
        st.rerun()

st.caption("Tip: Sample buttons automatically select the correct content type in the sidebar.")

st.markdown("## Input Text")

user_text = st.text_area(
    "Paste your news article or email here:",
    value=st.session_state.sample_text,
    height=260,
    placeholder="Paste a news article or email message here..."
)

analyze_button = st.button("Analyze Text", type="primary")


# ============================================================
# ANALYSIS LOGIC
# ============================================================

if analyze_button:
    if not user_text.strip():
        st.warning("Please paste some text first.")

    else:
        detected_type = detect_possible_input_type(user_text)

        if detected_type != "Unclear" and detected_type != content_type:
            st.error(
                f"Content type mismatch detected. The text looks more like **{detected_type}**, "
                f"but selected type is **{content_type}**."
            )

            st.info(
                "Please select the correct content type from the sidebar before analysis. "
                "This prevents the news model from analyzing emails or the spam model from analyzing news articles."
            )

            st.stop()

        with st.spinner("Analyzing text..."):

            # ------------------------------------------------
            # STEP 1: TRY GEMINI FIRST IF ENABLED
            # ------------------------------------------------
            if use_gemini:
                gemini_result = analyze_with_gemini(
                    user_text=user_text,
                    content_type=content_type,
                    temperature=temperature,
                    top_p=top_p
                )
            else:
                gemini_result = {
                    "error": True,
                    "message": "Gemini API is turned off. Using local ML fallback to save quota."
                }

            # ------------------------------------------------
            # CASE 1: GEMINI FAILED OR OFF → USE LOCAL ML FALLBACK
            # ------------------------------------------------
            if gemini_result["error"]:
                st.error("Gemini analysis failed or is turned off.")
                st.write(gemini_result["message"])

                st.info(
                    "The system is using the trained local ML fallback. "
                    "If the ensemble model files are available, it combines Logistic Regression, "
                    "Linear SVM, and Naive Bayes using soft voting."
                )

                result = analyze_with_ml_model(
                    user_text=user_text,
                    content_type=content_type
                )

                result = apply_security_policy_adjustment(
                    result=result,
                    user_text=user_text,
                    content_type=content_type
                )

                display_result(
                    result=result,
                    analysis_mode="Local ML Fallback",
                    user_text=user_text,
                    content_type=content_type
                )

            # ------------------------------------------------
            # CASE 2: GEMINI WORKED
            # ------------------------------------------------
            else:
                result = extract_json(gemini_result["text"])

                if result is None:
                    st.warning("Gemini response could not be parsed as valid JSON.")

                    with st.expander("View Raw Gemini Response"):
                        st.write(gemini_result["text"])

                    st.info(
                        "Because Gemini response was not in the expected JSON format, "
                        "the system is using the trained local ML fallback."
                    )

                    result = analyze_with_ml_model(
                        user_text=user_text,
                        content_type=content_type
                    )
                    
                    result = apply_security_policy_adjustment(
                        result=result,
                        user_text=user_text,
                        content_type=content_type
                    )

                    display_result(
                        result=result,
                        analysis_mode="Local ML Fallback",
                        user_text=user_text,
                        content_type=content_type
                    )

                else:
                    warning_words = find_warning_words(user_text)

                    if warning_words:
                        existing_reasons = result.get("key_reasons", [])
                        existing_reasons.append(
                            "Suspicious terms found: " + ", ".join(warning_words[:8])
                        )
                        result["key_reasons"] = existing_reasons

                    result = apply_security_policy_adjustment(
                        result=result,
                        user_text=user_text,
                        content_type=content_type
                    )

                    display_result(
                        result=result,
                        analysis_mode=f"Gemini API ({gemini_result.get('key_label', 'Key Used')})",
                        user_text=user_text,
                        content_type=content_type
                    )


# ============================================================
# HISTORY SECTION
# ============================================================

if len(st.session_state.history) > 0:
    st.markdown("---")
    st.markdown("## Analysis History")

    if st.button("Clear Analysis History"):
        st.session_state.history = []
        st.rerun()

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(history_df, width="stretch")

    csv = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Analysis History as CSV",
        data=csv,
        file_name="analysis_history.csv",
        mime="text/csv"
    )


# ============================================================
# DOMAIN-ONLY FOLLOW-UP Q&A SECTION
# ============================================================

st.markdown("---")
st.markdown("## Ask TruthGuard AI")
st.caption(
    "Ask domain-related questions about fake news, spam/phishing, suspicion score, risk level, "
    "or the latest analysis explanation."
)

if st.session_state.last_analysis is not None:
    latest = st.session_state.last_analysis
    st.info(
        f"Latest analysis available: **{latest.get('content_type')}** | "
        f"Mode: **{latest.get('analysis_mode')}** | "
        f"Time: **{latest.get('time')}**"
    )
else:
    st.warning(
        "No analysis result is available yet. You can still ask general questions about fake news, spam, phishing, or verification."
    )

with st.expander("Example questions you can ask", expanded=False):
    st.write("- Why was this email marked as phishing?")
    st.write("- Which words increased the suspicion score?")
    st.write("- How can I verify this news article?")
    st.write("- Why is password reset email marked uncertain?")
    st.write("- What does risk level mean?")
    st.write("- Why can Gemini and local ML give different scores?")

qa_question = st.text_area(
    "Ask a domain-related follow-up question:",
    height=100,
    placeholder="Example: Why did the system classify this as possible phishing?"
)

qa_col1, qa_col2 = st.columns([1, 4])

with qa_col1:
    ask_button = st.button("Ask Question")

with qa_col2:
    clear_qa = st.button("Clear Q&A History")

if clear_qa:
    st.session_state.qa_history = []
    st.rerun()

if ask_button:
    if not qa_question.strip():
        st.warning("Please enter a question first.")
    else:
        if not is_domain_related_question(qa_question):
            answer = (
                "I can only answer questions related to fake news detection, spam/phishing emails, "
                "suspicion score, risk level, analysis explanation, verification, and this project."
            )

            st.session_state.qa_history.append({
                "Question": qa_question,
                "Answer": answer,
                "Mode": "Domain Guard"
            })

            st.warning(answer)

        elif not use_gemini:
            answer = (
                "Follow-up Q&A requires Gemini API because local ML fallback can classify text "
                "but cannot answer open-ended explanation questions. Please turn on Gemini API to ask follow-up questions."
            )

            st.session_state.qa_history.append({
                "Question": qa_question,
                "Answer": answer,
                "Mode": "Gemini OFF"
            })

            st.info(answer)

        else:
            with st.spinner("Generating domain-specific answer..."):
                qa_result = answer_domain_question_with_gemini(qa_question)

                if qa_result["error"]:
                    answer = (
                        "Follow-up Q&A could not be completed because Gemini API is unavailable. "
                        "You can still use the local ML fallback for classification."
                    )

                    st.error(qa_result["message"])
                else:
                    answer = qa_result["answer"]

                st.session_state.qa_history.append({
                    "Question": qa_question,
                    "Answer": answer,
                    "Mode": qa_result.get("key_label", "Gemini API")
                })

if st.session_state.qa_history:
    st.markdown("### Q&A History")

    for idx, item in enumerate(reversed(st.session_state.qa_history), start=1):
        with st.expander(f"Q{idx}: {item['Question']}", expanded=(idx == 1)):
            st.markdown(f"**Answer Mode:** {item['Mode']}")
            st.write(item["Answer"])


# ============================================================
# SYSTEM WORKFLOW SECTION
# ============================================================

st.markdown("---")
st.markdown("## How TruthGuard AI Works")

flow_col1, flow_col2, flow_col3 = st.columns(3)

with flow_col1:
    st.markdown("### 1. GenAI Analysis + Q&A")
    st.write(
        "Gemini 2.5 Flash-Lite analyzes the text and generates verdict, score, reasons, and safety advice. "
        "It also supports domain-only follow-up questions when API is available."
    )

with flow_col2:
    st.markdown("### 2. Ensemble ML Fallback")
    st.write(
        "If Gemini fails or is turned off, the local fallback can combine "
        "Logistic Regression, Linear SVM, and Naive Bayes using soft voting."
    )

with flow_col3:
    st.markdown("### 3. Memory Layer")
    st.write(
        "Hindsight recalls similar previous analysis patterns and stores only "
        "safe summarized memories."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "TruthGuard AI uses Gemini API for generative analysis, trained local ensemble ML models as fallback, "
    "optional Hindsight memory for recalling previous analysis patterns, and domain-only Q&A for follow-up explanations. "
    "Results are assistive and should be verified from trusted sources."
)