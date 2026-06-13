---
title: Redrob Ranker
emoji: 🏢
colorFrom: yellow
colorTo: indigo
sdk: gradio
sdk_version: 6.18.0
python_version: '3.13'
app_file: app.py
pinned: false
---

# Sentinel-AI — Team `cache_Q`

> High-throughput candidate ranking pipeline for the RedRob AI Talent Search Challenge.

## Overview

This repository contains a **production-ready, zero-network-dependency** Python pipeline that:

1. **Streams** 100,000 candidate profiles from a JSON Lines file
2. **Filters** adversarial honeypot profiles using deterministic arithmetic checks
3. **Scores** candidates with a composite technical + behavioral ranking matrix
4. **Exports** a perfectly formatted, monotonically non-increasing CSV of the top 100 candidates

All processing completes on a standard CPU within a **5-minute wall-clock budget**.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
python rank.py --candidates ./candidates.jsonl --out ./team_cache_Q.csv
```

| Argument        | Default                  | Description                              |
|-----------------|--------------------------|------------------------------------------|
| `--candidates`  | `candidates.jsonl`       | Path to JSONL candidate dataset          |
| `--out`         | `team_cache_Q.csv`       | Output CSV file path                     |

### 3. Validate Output

The pipeline automatically invokes `validate_submission.py` at the end of execution. You can also run it independently:

```bash
python -c "from validate_submission import validate_submission; validate_submission('team_cache_Q.csv')"
```

---

## Output Format

The output CSV (`team_cache_Q.csv`) contains exactly **100 rows** with the following columns:

| Column         | Type    | Description                                          |
|----------------|---------|------------------------------------------------------|
| `candidate_id` | string  | Unique candidate identifier                          |
| `rank`         | integer | Sequential rank from 1 (best) to 100                 |
| `score`        | float   | Composite score, monotonically non-increasing         |
| `reasoning`    | string  | Tiered explanation (Elite / Strong Fit / Borderline)  |

Scores are sorted **descending**; ties are broken by `candidate_id` in **ascending lexicographic** order.

---

## Pipeline Architecture

```
candidates.jsonl
        │
        ▼
 ┌─────────────────────────────────────┐
 │  STAGE 1: Gated Streaming Filter   │
 │  • Honeypot arithmetic checker      │
 │  • Title adjacency guard            │
 │  • Industry verification layer      │
 └──────────────┬──────────────────────┘
                │ surviving candidates
                ▼
 ┌─────────────────────────────────────┐
 │  STAGE 2: Technical Scoring        │
 │  • Experience curve mapping         │
 │  • Core skill intersection          │
 │  • Execution context scoring        │
 └──────────────┬──────────────────────┘
                │ S_tech
                ▼
 ┌─────────────────────────────────────┐
 │  STAGE 3: Behavioral Calibration   │
 │  • Temporal inactivity decay        │
 │  • Response rate index              │
 │  • Notice period alignment          │
 │  • Geographic weighting             │
 │  • GitHub velocity scoring          │
 │  • Assessment validation            │
 └──────────────┬──────────────────────┘
                │ M_behavior
                ▼
 ┌─────────────────────────────────────┐
 │  STAGE 4: Aggregation & Reasoning  │
 │  Score_final = S_tech × M_behavior  │
 │  Sort → Top 100 → Rank → Explain   │
 └──────────────┬──────────────────────┘
                │
                ▼
 ┌─────────────────────────────────────┐
 │  STAGE 5: Validation & Export      │
 │  • Format check                     │
 │  • Monotonicity assertion           │
 │  • validate_submission() gate       │
 └─────────────────────────────────────┘
                │
                ▼
         team_cache_Q.csv
```

---

## Repository Structure

```
rehob.ai/
├── rank.py                              # Main pipeline script
├── app.py                               # Gradio UI for HuggingFace Spaces
├── requirements.txt                     # Pinned Python dependencies
├── submission_metadata.yaml             # Challenge metadata declarations
├── validate_submission.py               # Provided submission validator
├── candidates.jsonl                     # Input dataset (not committed)
├── team_cache_Q.csv                     # Output submission (generated)
└── README.md                            # This file
```

---

## Environment

- **Python**: 3.10+
- **OS**: Any (tested on Windows 11, Ubuntu 22.04)
- **Network**: Not required — fully offline execution
- **Hardware**: Standard CPU (no GPU needed)

---

## Team

**Team ID**: `cache_Q`
**Challenge**: RedRob AI Talent Search — Candidate Ranking Pipeline

---

## License

This project is submitted as original work for the RedRob AI Talent Search Challenge.
