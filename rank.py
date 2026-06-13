#!/usr/bin/env python3
"""
rank.py — Production-Ready Talent Search Ranking Pipeline
==========================================================
Parses candidates from candidates.jsonl.gz, filters adversarial traps,
applies an advanced multi-signal behavioral matrix with text similarity,
and leverages mathematically safe early-stopping constraints.

Usage:
    python rank.py --candidates ./candidates.jsonl.gz --out ./team_cache_Q.csv
"""

# ── Block all HuggingFace network access before any imports ──
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./model_cache"

import argparse
import gzip
import json
import math
import re
import time
import sys
from datetime import datetime, date
from typing import Any, Optional
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS & RUNTIME MATRICES
# ──────────────────────────────────────────────────────────────────────────────
BASELINE_DATE = date.today()
BASELINE_DATETIME = datetime.today()
DEFAULT_INPUT_FILE = "candidates.jsonl"
DEFAULT_OUTPUT_FILE = "team_cache_Q.csv"
MODEL_CACHE_PATH = "./model_cache/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 256
TOP_K = 100
WALL_CLOCK_LIMIT_SECONDS = 300  # 5 minutes

JD_TEXT = (
    "Senior AI Engineer for founding team at Series A AI-native talent intelligence platform. "
    "Must have production experience with embeddings-based retrieval systems deployed to real users, "
    "vector databases like Pinecone Weaviate Qdrant Milvus FAISS Elasticsearch, "
    "and hybrid search infrastructure. Expertise in ranking models, LLMs, fine-tuning, "
    "RAG pipelines, and recommendation engines. Must have shipped at least one end-to-end "
    "ranking search or recommendation system to real users at meaningful scale. "
    "Hands-on experience designing evaluation frameworks for ranking systems including "
    "NDCG MRR MAP offline-to-online correlation and A/B test interpretation. "
    "Scrappy product-engineering attitude willing to ship a working ranker quickly. "
    "Strong Python, production deployment, scalable AI infrastructure, deep learning, "
    "NLP, transformer architectures, model optimization, benchmark-driven development. "
    "Located in Pune or Noida India. 5 to 9 years experience at product companies."
)

CORE_SKILL_PILLARS = {
    "embeddings", "retrieval", "ranking", "llms",
    "fine-tuning", "rag", "vector",
}

NON_TECHNICAL_PATTERN = re.compile(
    r"\b(marketing|sales|recruiter|hr|writer|product\s*manager|pm|designer"
    r"|graphic|accountant|operations)\b",
    re.IGNORECASE,
)
TECHNICAL_PATTERN = re.compile(
    r"\b(engineer|developer|scientist|ml|ai|data|backend|software|architect)\b",
    re.IGNORECASE,
)
CONSULTING_PATTERN = re.compile(
    r"\b(it\s+services|it\s+consulting|outsourcing|systems\s+integration)\b",
    re.IGNORECASE,
)

PRODUCTION_VERBS = {
    "ship", "deploy", "scale", "infrastructure",
    "benchmark", "optimize", "production",
}
FRAMEWORK_BUZZWORDS = {
    "langchain", "llamaindex", "prompt engineering", "wrapper",
}

TARGET_HUBS = {"pune", "noida"}
TIER1_CITIES = {
    "bangalore", "bengaluru", "hyderabad",
    "mumbai", "delhi", "gurgaon", "chennai",
}
ASSESSMENT_KEYS = {"ml", "ai", "python", "feature", "search"}

PRODUCT_COMPANIES = {
    "flipkart", "razorpay", "swiggy", "zomato", "ola", "meesho",
    "paytm", "phonepe", "dream11", "cred", "zerodha", "freshworks",
    "zoho", "postman", "browserstack", "unacademy", "nykaa",
    "policybazaar", "inmobi", "glance", "pharmeasy", "byju's",
    "haptik", "yellow.ai", "observe.ai", "rephrase.ai", "sarvam ai",
    "krutrim", "niramai", "verloop.io", "saarthi.ai",
    "google", "microsoft", "amazon", "meta", "apple", "netflix",
}

CAREER_DEPTH_KEYWORDS = {
    "ranking system", "search infrastructure", "recommendation engine",
    "retrieval system", "a/b test", "ndcg", "embedding", "vector search",
    "fine-tun", "reranking", "re-ranking", "candidate ranking",
    "information retrieval", "semantic search", "hybrid search",
}

WALL_START = time.monotonic()

# ──────────────────────────────────────────────────────────────────────────────
# UTILITY HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def _safe_get(obj: Any, *keys: str, default: Any = None) -> Any:
    current = obj
    for k in keys:
        if isinstance(current, dict):
            current = current.get(k, default)
        else:
            return default
    return current

def _parse_date(raw: Optional[str]) -> Optional[date]:
    if not raw or not isinstance(raw, str):
        return None
    raw = raw.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%SZ", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    try:
        return datetime.strptime(raw[:7], "%Y-%m").date()
    except (ValueError, IndexError):
        return None

def _month_diff(d1: date, d2: date) -> float:
    return abs((d1.year - d2.year) * 12 + (d1.month - d2.month))

def _check_wall_clock():
    elapsed = time.monotonic() - WALL_START
    if elapsed > WALL_CLOCK_LIMIT_SECONDS - 10:
        raise RuntimeError(f"Wall-clock budget nearly exhausted ({elapsed:.1f}s). Aborting.")

def _open_input(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")

def _buzzword_match(phrase: str, text: str) -> bool:
    """Bug 4 Fix: Differentiates phrase substrings from bound single-words."""
    if " " in phrase:
        return phrase in text
    return bool(re.search(r'\b' + re.escape(phrase) + r'\b', text))

def generate_reasoning(
    rank: int, yoe: float, top_skill: str, company: str,
    rrr_pct: float, notice_days: int, title: str = "AI Engineer",
    location: str = "India", github_score: float = -1,
    n_core_skills: int = 0, has_ranking_exp: bool = False,
    is_product_co: bool = False
) -> str:
    """Generates specific, JD-connected, varied reasoning for each ranked candidate."""
    seed = (rank * 17 + int(yoe * 7) + int(rrr_pct) + n_core_skills * 3) % 8
    loc_lower = location.lower()
    is_pune_noida = "pune" in loc_lower or "noida" in loc_lower
    in_india = any(c in loc_lower for c in [
        "india", "pune", "noida", "bangalore", "bengaluru", "hyderabad", "mumbai",
        "delhi", "gurgaon", "chennai", "kolkata", "chandigarh", "indore", "bhubaneswar",
        "vizag", "visakhapatnam", "kochi", "trivandrum", "coimbatore", "ahmedabad",
        "jaipur", "lucknow", "patna", "bhopal", "nagpur", "surat", "thiruvananthapuram",
        "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa",
        "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala",
        "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland",
        "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura",
        "uttar pradesh", "uttarakhand", "west bengal", "puducherry", "lakshadweep",
        "ladakh", "jammu", "kashmir"
    ])

    # ── Part 1: Opening sentence — rank-appropriate, includes specific facts ──
    if rank <= 10:
        openers = [
            f"{title} with {yoe:.1f} years of experience and {n_core_skills}/7 core ML skill pillars matched — strong alignment with the Senior AI Engineer founding-team role.",
            f"Highly ranked {title} ({yoe:.1f} YOE) currently at {company}, matching {n_core_skills} of 7 core competencies the JD requires for building retrieval and ranking infrastructure.",
            f"{title} at {company} with {yoe:.1f} years and demonstrated depth in {top_skill} — fits the JD's emphasis on production-grade ML systems and evaluation-driven development.",
            f"Top candidate: {title} with {yoe:.1f} YOE at {company}. Matches {n_core_skills}/7 core pillars including {top_skill}; aligns with the JD's need for hands-on builders over researchers.",
            f"{yoe:.1f}-year {title} at {company} with specialization in {top_skill}. Strong match for the JD's requirement of shipping end-to-end ranking and retrieval systems.",
            f"{title} currently at {company}, {yoe:.1f} YOE. Covers {n_core_skills}/7 core skill areas; particularly strong in {top_skill} which directly maps to the JD's hybrid search infrastructure needs.",
            f"Exceptional {title} ({yoe:.1f} YOE) bringing {n_core_skills} core ML competencies from {company}. Profile closely matches the JD's emphasis on embeddings, retrieval, and production deployment.",
            f"{title} with {yoe:.1f} years at {company}. {n_core_skills}/7 core pillars matched including {top_skill} — well-suited for the founding-team AI engineer position.",
        ]
    elif rank <= 50:
        openers = [
            f"{title} with {yoe:.1f} YOE at {company}, covering {n_core_skills}/7 core ML competencies. Solid background relevant to the retrieval and ranking requirements in the JD.",
            f"{yoe:.1f}-year {title} at {company} with expertise in {top_skill}. Matches several JD requirements around production ML systems and scalable AI infrastructure.",
            f"{title} ({yoe:.1f} YOE) currently at {company}. {n_core_skills} core skills matched; shows practical experience in areas the JD prioritizes: embeddings, ranking, and production deployment.",
            f"Qualified {title} with {yoe:.1f} years and {n_core_skills}/7 core skill matches from {company}. Relevant experience in {top_skill} maps to the JD's search and ranking focus.",
            f"{title} at {company} ({yoe:.1f} YOE) with {n_core_skills} core skill matches including {top_skill}. Profile shows the production-engineering orientation the JD describes.",
            f"{yoe:.1f}-year {title} from {company}. Covers {n_core_skills}/7 core pillars; {top_skill} expertise aligns with the JD's need for hands-on ML system builders.",
            f"{title} with {yoe:.1f} YOE, bringing {n_core_skills} core competencies from {company}. Experience in {top_skill} relevant to the JD's retrieval infrastructure requirements.",
            f"Competent {title} ({yoe:.1f} YOE) at {company} with {n_core_skills}/7 core ML skills matched. Background in {top_skill} connects to the JD's ranking and evaluation focus.",
        ]
    else:
        openers = [
            f"{title} with {yoe:.1f} YOE at {company}. Only {n_core_skills}/7 core ML pillars matched — below the depth the JD targets for the founding AI engineering team.",
            f"{yoe:.1f}-year {title} from {company} with limited core skill coverage ({n_core_skills}/7). Partial overlap with JD requirements but gaps in key retrieval and ranking areas.",
            f"{title} at {company}, {yoe:.1f} YOE. {n_core_skills}/7 core matches — included as a borderline candidate; strongest signal is {top_skill} but overall JD alignment is weaker.",
            f"{title} ({yoe:.1f} YOE) currently at {company}. Covers {n_core_skills} of 7 core pillars; ranking this low due to limited depth in the JD's primary focus areas.",
            f"{yoe:.1f}-year {title} from {company}. {n_core_skills}/7 core skills — some relevant background in {top_skill} but missing depth in retrieval systems and evaluation frameworks the JD requires.",
            f"{title} at {company} with {yoe:.1f} YOE. Below-threshold core skill match ({n_core_skills}/7); included as a filler based on {top_skill} relevance.",
            f"Lower-ranked {title} ({yoe:.1f} YOE, {company}). {n_core_skills}/7 core pillars; profile is adjacent to JD requirements but lacks the production ranking system experience the role demands.",
            f"{title} with {yoe:.1f} years at {company}. Limited core skill overlap ({n_core_skills}/7) and weaker alignment with the JD's emphasis on shipped retrieval and search infrastructure.",
        ]
    opener = openers[seed]

    # ── Part 2: Behavioral & concern sentence — honest, specific, JD-connected ──
    concerns = []
    strengths = []

    # Notice period (JD says "sub-30-day notice preferred, can buy out up to 30")
    if notice_days <= 30:
        strengths.append(f"{notice_days}-day notice period (within JD's preferred window)")
    elif notice_days <= 60:
        concerns.append(f"{notice_days}-day notice period (above the JD's preferred 30-day window but manageable)")
    else:
        concerns.append(f"{notice_days}-day notice period is a significant hiring delay — JD prefers sub-30 days")

    # Location (JD says Pune/Noida preferred, Tier-1 with relocation OK)
    if is_pune_noida:
        strengths.append("based in Pune/Noida (JD's target location)")
    elif in_india:
        concerns.append(f"based in {location}, not in JD's preferred Pune/Noida hubs")
    else:
        concerns.append(f"based in {location} (outside India) — JD states no visa sponsorship")

    # Engagement
    if rrr_pct >= 70:
        strengths.append(f"{rrr_pct:.0f}% recruiter response rate (highly responsive)")
    elif rrr_pct >= 40:
        pass  # neutral, don't mention
    else:
        concerns.append(f"only {rrr_pct:.0f}% recruiter response rate — low availability signal")

    # GitHub
    if isinstance(github_score, (int, float)):
        gs = float(github_score)
        if gs == -1:
            concerns.append("no GitHub linked (JD values open-source contributions)")
        elif gs >= 75:
            strengths.append(f"strong GitHub activity (score: {gs:.0f}/100)")
        elif gs < 25 and gs >= 0:
            concerns.append(f"low GitHub activity (score: {gs:.0f}/100)")

    # Ranking/search experience
    if has_ranking_exp:
        strengths.append("career history shows direct ranking/retrieval/search system experience")
    elif rank <= 30:
        concerns.append("no explicit ranking or search system experience found in career descriptions")

    # Product company
    if is_product_co:
        strengths.append(f"currently at {company} (product company, aligns with JD's preference)")

    # YOE range (JD says 5-9 ideal)
    if yoe < 4.0:
        concerns.append(f"{yoe:.1f} YOE is below the JD's 5-9 year target range")
    elif yoe > 10.0:
        concerns.append(f"{yoe:.1f} YOE is above the JD's 5-9 year preferred band")

    # Build the second sentence from strengths and concerns
    parts = []
    if strengths:
        parts.append(", ".join(strengths[:2]))
    if concerns:
        conc_str = "; ".join(concerns[:2])
        if strengths:
            parts.append(f"however {conc_str}")
        else:
            parts.append(f"Concerns: {conc_str}")

    if parts:
        detail_sentence = ". ".join(p.capitalize() if i == 0 and not p[0].isupper() else p for i, p in enumerate(parts)) + "."
    else:
        detail_sentence = f"Engagement signals are neutral; {notice_days}-day notice period."

    return f"{opener} {detail_sentence}"

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1: GATED FILTERING (STREAMING ENTRIES)
# ──────────────────────────────────────────────────────────────────────────────
def is_honeypot(record: dict) -> bool:
    """Deterministic Honeypot Arithmetic Checker."""
    profile = record.get("profile", {})
    if not isinstance(profile, dict):
        return True  
    yoe = profile.get("years_of_experience")
    if yoe is None or not isinstance(yoe, (int, float)):
        return True
    total_career_months = float(yoe) * 12.0

    # Skills Inflation Trap Check
    skills = record.get("skills", [])
    if isinstance(skills, list):
        expert_months = 0.0
        for skill in skills:
            if not isinstance(skill, dict):
                continue
            if str(skill.get("proficiency", "")).lower() == "expert":
                dur = skill.get("duration_months")
                if dur is not None and isinstance(dur, (int, float)):
                    if dur == 0:
                        return True  # Zero-duration expert trap
                    expert_months += float(dur)
                else:
                    return True
        if total_career_months > 0 and expert_months > (total_career_months * 1.5):
            return True
        if total_career_months == 0 and expert_months > 0:
            return True

    # Temporal Career Paradox Check
    career_history = record.get("career_history", [])
    if isinstance(career_history, list):
        for job in career_history:
            if not isinstance(job, dict):
                continue
            start_raw = job.get("start_date")
            start_dt = _parse_date(start_raw)
            if start_dt is None:
                continue  
            if start_dt > BASELINE_DATE:
                return True  
            end_raw = job.get("end_date")
            end_dt = _parse_date(end_raw) if end_raw else BASELINE_DATE
            
            calc_months = _month_diff(start_dt, end_dt)
            reported_months = job.get("duration_months")
            if reported_months is not None and isinstance(reported_months, (int, float)):
                if abs(calc_months - float(reported_months)) > 3:
                    return True
    return False

def has_valid_title(record: dict) -> bool:
    """Bug 7 Fix: Implements core-skill escape-hatch fallback for non-standard technical titles."""
    title = _safe_get(record, "profile", "current_title", default="")
    if not isinstance(title, str) or not title.strip():
        return False
    title_clean = title.strip()
    if NON_TECHNICAL_PATTERN.search(title_clean):
        return False
    if TECHNICAL_PATTERN.search(title_clean):
        return True
        
    # Escape hatch filter boundary
    skills = record.get("skills", [])
    if isinstance(skills, list):
        pillar_hits = sum(
            1 for s in skills if isinstance(s, dict)
            and any(p in str(s.get("name", "")).lower() for p in CORE_SKILL_PILLARS)
        )
        if pillar_hits >= 2:
            return True
            
    return False

def passes_industry_filter(record: dict) -> bool:
    """Bug 6 Fix: Evaluates deep career history to trace real mid-market product pivots."""
    profile = record.get("profile", {})
    if not isinstance(profile, dict):
        return False
    industry = str(profile.get("current_industry", "")).strip()
    
    if not CONSULTING_PATTERN.search(industry):
        return True  # Clears immediately if current company isn't consulting
        
    # Deep trace career array mapping
    for job in record.get("career_history", []):
        if isinstance(job, dict):
            job_industry = str(job.get("industry", ""))
            if job_industry and not CONSULTING_PATTERN.search(job_industry):
                return True  # Valid product background verified
                
    return False

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2: STRUCTURAL COMPONENT CALCULATIONS
# ──────────────────────────────────────────────────────────────────────────────
def compute_experience_curve(yoe: float) -> float:
    if 5.0 <= yoe <= 9.0:
        return 1.0
    return max(0.1, 1.0 - 0.15 * abs(7.0 - yoe))

def compute_skill_intersection(record: dict) -> tuple[float, str]:
    """Bug 2 Fix: Extracts the highest-endorsed matching skill variable for explanations."""
    skills = record.get("skills", [])
    matched = set()
    matched_skills = []  # List of tuples: (endorsements, display_name)
    
    if isinstance(skills, list):
        for skill in skills:
            if not isinstance(skill, dict):
                continue
            skill_name = str(skill.get("name", "")).lower().strip()
            for pillar in CORE_SKILL_PILLARS:
                if pillar in skill_name:
                    matched.add(pillar)
                    matched_skills.append((
                        int(skill.get("endorsements", 0)),
                        str(skill.get("name", skill_name))
                    ))
                    break  # Avoid duplicate counting loops
                    
    s_skills = len(matched) / len(CORE_SKILL_PILLARS)
    best_skill_name = max(matched_skills, key=lambda x: x[0])[1] if matched_skills else "Systems Architecture"
    return s_skills, best_skill_name

def compute_execution_context(record: dict) -> float:
    text_parts = []
    profile = record.get("profile", {})
    summary = profile.get("summary", "")
    if isinstance(summary, str):
        text_parts.append(summary.lower())
    
    career_history = record.get("career_history", [])
    if isinstance(career_history, list):
        for job in career_history:
            if isinstance(job, dict):
                desc = job.get("description", "")
                if isinstance(desc, str):
                    text_parts.append(desc.lower())
                    
    joined = " ".join(text_parts)
    words = set(re.findall(r'\b\w+\b', joined))
    
    prod_count = sum(1 for v in PRODUCTION_VERBS if v in words)
    framework_count = sum(1 for b in FRAMEWORK_BUZZWORDS if _buzzword_match(b, joined))
    
    if framework_count > 0 and prod_count == 0:
        return 0.4
    return 1.0 + (min(prod_count, 12) * 0.05)

def compute_education_bonus(record: dict) -> float:
    """Missing Signal Integration: Maps additive credential tier bonuses."""
    for edu in record.get("education", []):
        if isinstance(edu, dict):
            tier = str(edu.get("tier", "")).lower().strip()
            if tier == "tier_1":
                return 0.05
            if tier == "tier_2":
                return 0.02
    return 0.0

def build_candidate_text(record: dict) -> str:
    """Bug 1 Fix: Sorts career timelines chronologically using the is_current flag."""
    profile = record.get("profile", {})
    parts = []
    title = profile.get("current_title", "")
    if isinstance(title, str) and title.strip():
        parts.append(title.strip())
    summary = profile.get("summary", "")
    if isinstance(summary, str) and summary.strip():
        parts.append(summary.strip())
        
    skills = record.get("skills", [])
    if isinstance(skills, list):
        skill_names = [s.get("name", "").strip() for s in skills if isinstance(s, dict) and s.get("name")]
        if skill_names:
            parts.append("Skills: " + ", ".join(skill_names))
            
    career = record.get("career_history", [])
    if isinstance(career, list):
        career_sorted = sorted(career, key=lambda j: (0 if isinstance(j, dict) and j.get("is_current") else 1))
        for job in career_sorted[:3]:
            if isinstance(job, dict):
                desc = job.get("description", "")
                if isinstance(desc, str) and desc.strip():
                    parts.append(desc.strip())
                    
    return (". ".join(parts))[:2000]

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 3: MULTI-SIGNAL BEHAVIORAL CALIBRATION MATRIX
# ──────────────────────────────────────────────────────────────────────────────
def compute_m_behavior(record: dict) -> float:
    """Calculates composite scalar scaling including 3 missing platform records."""
    signals = record.get("redrob_signals", {})
    if not isinstance(signals, dict): signals = {}
    profile = record.get("profile", {})
    if not isinstance(profile, dict): profile = {}
    
    modifier = 1.0
    
    # 1. Temporal Inactivity Decay
    last_active_raw = signals.get("last_active_date")
    last_active_dt = _parse_date(str(last_active_raw)) if last_active_raw else None
    if last_active_dt is not None:
        delta_months = _month_diff(last_active_dt, BASELINE_DATE)
        modifier *= math.exp(-0.1 * delta_months)
    else:
        modifier *= math.exp(-0.1 * 24)
        
    # 2. Response Rate Index
    rrr = signals.get("recruiter_response_rate")
    if isinstance(rrr, (int, float)):
        modifier *= (0.2 + 0.8 * max(0.0, min(1.0, float(rrr))))
    else:
        modifier *= 0.2
        
    # Bug 3 Fix: Re-integrates the open_to_work availability booster
    if bool(signals.get("open_to_work_flag", False)):
        modifier *= 1.08
        
    # 3. Notice Period Alignment
    notice = signals.get("notice_period_days")
    if isinstance(notice, (int, float)):
        notice_val = float(notice)
        if notice_val <= 30: modifier *= 1.15
        elif notice_val > 60: modifier *= 0.80
        
    # 4. Geographic & Hybrid Target Weighting
    location = str(profile.get("location", "")).lower().strip()
    country = str(profile.get("country", "")).lower().strip()
    willing = bool(signals.get("willing_to_relocate", False))
    
    geo_applied = False
    for hub in TARGET_HUBS:
        if hub in location:
            modifier *= 1.25
            geo_applied = True
            break
    if not geo_applied:
        for city in TIER1_CITIES:
            if city in location:
                if willing: modifier *= 1.10
                geo_applied = True
                break
    if not geo_applied:
        if country == "india" and not willing:
            modifier *= 0.50
        elif country != "india":
            modifier *= 0.05
            
    # 5. GitHub Source Velocity
    github = signals.get("github_activity_score")
    if isinstance(github, (int, float)):
        github_val = float(github)
        if github_val == -1: modifier *= 0.90
        elif github_val < 25: modifier *= 0.75
        elif github_val > 75: modifier *= 1.15
    else:
        modifier *= 0.90
        
    # 6. Verified Assessment Validation
    assessments = signals.get("skill_assessment_scores", {})
    if isinstance(assessments, dict) and assessments:
        relevant_scores = []
        for key, val in assessments.items():
            key_lower = str(key).lower()
            for core_key in ASSESSMENT_KEYS:
                if core_key in key_lower and isinstance(val, (int, float)):
                    relevant_scores.append(float(val))
                    break
        if relevant_scores:
            mean_score = sum(relevant_scores) / len(relevant_scores)
            modifier *= (0.8 + (mean_score / 100.0) * 0.4)
            
    # Missing Signal Integration: Metric 7 (offer_acceptance_rate)
    offer_rate = signals.get("offer_acceptance_rate")
    if isinstance(offer_rate, (int, float)) and float(offer_rate) >= 0:
        modifier *= (0.90 + 0.15 * min(1.0, float(offer_rate)))
        
    # Missing Signal Integration: Metric 8 (interview_completion_rate)
    interview_rate = signals.get("interview_completion_rate")
    if isinstance(interview_rate, (int, float)) and float(interview_rate) < 0.6:
        modifier *= 0.88  # Reduces score for low interview attendance
        
    # Missing Signal Integration: Metric 9 (profile_completeness_score)
    completeness = signals.get("profile_completeness_score")
    if isinstance(completeness, (int, float)):
        modifier *= (0.88 + 0.14 * min(1.0, float(completeness) / 100.0))

    # Signal 10: Recruiter demand proxy (saved_by_recruiters_30d)
    saved = signals.get("saved_by_recruiters_30d")
    if isinstance(saved, (int, float)):
        saved_val = float(saved)
        if saved_val >= 10: modifier *= 1.08
        elif saved_val >= 5: modifier *= 1.03
        elif saved_val == 0: modifier *= 0.95

    # Signal 11: Response speed (avg_response_time_hours)
    resp_time = signals.get("avg_response_time_hours")
    if isinstance(resp_time, (int, float)):
        resp_val = float(resp_time)
        if resp_val <= 24: modifier *= 1.06
        elif resp_val > 168: modifier *= 0.92

    # Signal 12: Profile trust (verified_email + verified_phone)
    v_email = bool(signals.get("verified_email", False))
    v_phone = bool(signals.get("verified_phone", False))
    if v_email and v_phone: modifier *= 1.04
    elif not v_email and not v_phone: modifier *= 0.92

    # Signal 13: Work mode alignment (preferred_work_mode)
    work_mode = str(signals.get("preferred_work_mode", "")).lower()
    if work_mode in ("hybrid", "flexible", "onsite"):
        modifier *= 1.03
    elif work_mode == "remote":
        modifier *= 0.94

    # Signal 14: Market visibility (search_appearance_30d)
    search_app = signals.get("search_appearance_30d")
    if isinstance(search_app, (int, float)):
        sa_val = float(search_app)
        if sa_val >= 200: modifier *= 1.05
        elif sa_val < 20: modifier *= 0.95

    # Signal 15: Active job-seeking (applications_submitted_30d)
    apps = signals.get("applications_submitted_30d")
    if isinstance(apps, (int, float)):
        apps_val = float(apps)
        if apps_val >= 3: modifier *= 1.04

    return modifier

# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE PIPELINE EXECUTION LOOP
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Optimized Talent Search Ranking Pipeline — team cache_Q")
    parser.add_argument("--candidates", default=DEFAULT_INPUT_FILE, help="Path to gzipped JSONL input")
    parser.add_argument("--out", default=DEFAULT_OUTPUT_FILE, help="Output CSV path")
    return parser.parse_args()

def main():
    args = parse_args()
    input_file = args.candidates
    output_file = args.out
    
    print("=" * 70)
    print("  RANK.PY — Comprehensive Verified Production Pipeline  [team: cache_Q]")
    print("=" * 70)
    t0 = time.monotonic()
    
    # ── STAGE 1: Streaming Gated Filtering ──
    print("\n[STAGE 1] Running streaming filter gates...")
    surviving_records = []
    total_read, filtered_honeypot, filtered_title, filtered_industry = 0, 0, 0, 0
    
    with _open_input(input_file) as f:
        for line_num, line in enumerate(f, start=1):
            if line_num % 25000 == 0:
                _check_wall_clock()
                print(f"  ... processed {line_num:>7,} source profiles")
            line = line.strip()
            if not line: continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError: continue
            
            total_read += 1
            if is_honeypot(record): filtered_honeypot += 1; continue
            if not has_valid_title(record): filtered_title += 1; continue
            if not passes_industry_filter(record): filtered_industry += 1; continue
            
            surviving_records.append(record)
            
    print(f"  Surviving records pool size: {len(surviving_records):,}")
    if len(surviving_records) == 0:
        raise RuntimeError("FATAL: Zero candidates survived filtering.")

    # ── STAGE 2: Bounds Pre-Computation & Structural Sorting ──
    print("\n[STAGE 2] Running bounds mapping calculations...")
    candidates_bound_pool = []
    
    for idx, rec in enumerate(surviving_records):
        s_skills, top_skill = compute_skill_intersection(rec)
        r_exec = compute_execution_context(rec)
        profile = rec.get("profile", {})
        yoe = float(profile.get("years_of_experience", 0))
        e_curve = compute_experience_curve(yoe)
        edu_bonus = compute_education_bonus(rec)
        
        # Recalibrated base weighting matrix including education
        s_base = (0.33 * s_skills) + (0.19 * e_curve) + (0.13 * r_exec) + edu_bonus
        s_tech_max = s_base + 0.30
        
        m_behavior = compute_m_behavior(rec)
        score_max = s_tech_max * m_behavior
        candidate_id = str(rec.get("candidate_id", f"UNKNOWN_{idx}"))
        
        candidates_bound_pool.append({
            "record": rec,
            "s_base": s_base,
            "m_behavior": m_behavior,
            "score_max": score_max,
            "candidate_id": candidate_id,
            "top_skill": top_skill,
            "yoe": yoe
        })
        
    candidates_bound_pool.sort(key=lambda x: (-x["score_max"], x["candidate_id"]))
    
    # ── STAGE 3: Lazy Inference Loop with Early Stopping ──
    print("\n[STAGE 3] Running inference re-ranking operations...")
    t_model_start = time.monotonic()
    model = SentenceTransformer(MODEL_CACHE_PATH, device="cpu")
    jd_embedding = model.encode([JD_TEXT], normalize_embeddings=True, show_progress_bar=False)
    
    fully_evaluated = []
    num_encoded = 0
    total_pool_size = len(candidates_bound_pool)
    
    for b_start in range(0, total_pool_size, EMBEDDING_BATCH_SIZE):
        _check_wall_clock()
        batch = candidates_bound_pool[b_start : b_start + EMBEDDING_BATCH_SIZE]
        
        if len(fully_evaluated) >= TOP_K:
            score_thresh = fully_evaluated[TOP_K - 1]["score"]
            if batch[0]["score_max"] < score_thresh:
                print(f"  [EARLY STOP TRIGGERED] Pruned {total_pool_size - num_encoded:,} rows.")
                break
                
        batch_texts = [build_candidate_text(item["record"]) for item in batch]
        batch_embeddings = model.encode(batch_texts, batch_size=len(batch_texts), normalize_embeddings=True, show_progress_bar=False)
        batch_semantic_scores = np.clip(np.dot(batch_embeddings, jd_embedding.T).flatten(), 0.0, 1.0)
        num_encoded += len(batch)
        
        for idx, item in enumerate(batch):
            sem_score = float(batch_semantic_scores[idx])
            score_final = (item["s_base"] + 0.30 * sem_score) * item["m_behavior"]
            
            signals = item["record"].get("redrob_signals", {})
            career_history = item["record"].get("career_history", [])
            
            # Bug 1 Fix: Scans for active is_current flag over array positions
            current_job = next(
                (j for j in career_history if isinstance(j, dict) and j.get("is_current")),
                career_history[0] if career_history else {}
            )
            company = "Product Company"
            if isinstance(current_job, dict) and current_job.get("company"):
                company = str(current_job["company"])
                    
            rrr_raw = signals.get("recruiter_response_rate")
            rrr_pct = float(rrr_raw) * 100 if isinstance(rrr_raw, (int, float)) else 0.0
            notice_raw = signals.get("notice_period_days")
            notice_days = int(notice_raw) if isinstance(notice_raw, (int, float)) else 0
            
            title = _safe_get(item["record"], "profile", "current_title", default="AI Engineer")
            location = _safe_get(item["record"], "profile", "location", default="India")
            
            # Extract additional data for richer reasoning
            github_score = signals.get("github_activity_score", -1) if isinstance(signals, dict) else -1
            skills_list = item["record"].get("skills", [])
            core_matched = set()
            for sk in (skills_list if isinstance(skills_list, list) else []):
                if isinstance(sk, dict):
                    sn = str(sk.get("name", "")).lower()
                    for p in CORE_SKILL_PILLARS:
                        if p in sn:
                            core_matched.add(p)
            n_core_skills = len(core_matched)
            
            # Check career depth for ranking/search/retrieval experience
            career_texts = ""
            for j in (career_history if isinstance(career_history, list) else []):
                if isinstance(j, dict):
                    d = j.get("description", "")
                    if isinstance(d, str): career_texts += " " + d.lower()
            has_ranking_exp = any(kw in career_texts for kw in CAREER_DEPTH_KEYWORDS)
            
            # Check if company is a known product company
            is_product_co = company.lower().strip() in PRODUCT_COMPANIES
            
            fully_evaluated.append({
                "candidate_id": item["candidate_id"],
                "score": round(score_final, 10),
                "yoe": item["yoe"],
                "top_skill": item["top_skill"],
                "company": company,
                "rrr_pct": rrr_pct,
                "notice_days": notice_days,
                "title": title,
                "location": location,
                "github_score": github_score,
                "n_core_skills": n_core_skills,
                "has_ranking_exp": has_ranking_exp,
                "is_product_co": is_product_co
            })
            
        fully_evaluated.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    import gc
    del model, candidates_bound_pool, surviving_records
    gc.collect()

    # ── STAGE 4: Final Rank Ordering & Reasoning Generation ──
    print("\n[STAGE 4] Composing ranked dataframe columns...")
    df = pd.DataFrame(fully_evaluated).head(TOP_K)
    df["rank"] = range(1, len(df) + 1)
    
    reasoning_col = []
    for _, row in df.iterrows():
        reasoning_col.append(
            generate_reasoning(
                rank=int(row["rank"]), yoe=float(row["yoe"]), top_skill=str(row["top_skill"]),
                company=str(row["company"]), rrr_pct=float(row["rrr_pct"]), notice_days=int(row["notice_days"]),
                title=str(row.get("title", "AI Engineer")), location=str(row.get("location", "India")),
                github_score=float(row.get("github_score", -1)),
                n_core_skills=int(row.get("n_core_skills", 0)),
                has_ranking_exp=bool(row.get("has_ranking_exp", False)),
                is_product_co=bool(row.get("is_product_co", False))
            )
        )
    df["reasoning"] = reasoning_col
    
    # Structural Safeguards
    scores = df["score"].tolist()
    for i in range(1, len(scores)):
        if scores[i] > scores[i - 1]:
            raise RuntimeError(f"Monotonicity structural failure at index position {i}.")

    # ── STAGE 5: File Export & Format Verification Passes ──
    print("\n[STAGE 5] Saving and executing formatting checks...")
    output_df = df[["candidate_id", "rank", "score", "reasoning"]].copy()
    output_df.to_csv(output_file, index=False)
    
    try:
        from validate_submission import validate_submission
        result = validate_submission(output_file)
        if result and isinstance(result, (list, tuple)) and len(result) > 0:
            raise RuntimeError(f"Submission validation error report:\n" + "\n".join(str(e) for e in result))
        print("  [PASSED] Output formatted cleanly.")
    except ImportError:
        print("  [WARNING] validate_submission tool missing. Bypassed.")

    print(f"\nPipeline execution successfully finalized. Total time: {time.monotonic() - t0:.2f}s")

if __name__ == "__main__":
    main()
