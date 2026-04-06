"""
LLM Judge for FYP Medical Activity Analysis System
=====================================================
Evaluates output quality from the home-activity-based medical analysis pipeline.

Score 1  : Symptom-Evidence Validation
           Verifies detected symptoms against raw data in data1.json,
           validates quantitative claims in collectiveEvidence.

Score 2  : Disease-Symptom RAG Validation
           Uses the existing FAISS RAG to verify disease-symptom medical
           relationships; validates reasoning quality.

Score 3  : Confidence Calibration
           Checks whether all confidence labels are well-calibrated using
           Evidence Strength scoring and Expected Calibration Error (ECE).

Overall  : Weighted combination of the three scores.

Usage
-----
python llm_judge.py --data data1.json --output out1.json
python llm_judge.py --data data1.json --output out1.json --rag-path .rag
python llm_judge.py --data data1.json --output out1.json --no-rag
"""

import json
import re
import os
import sys
import time
import logging
import argparse
from typing import Dict, List, Optional
from collections import defaultdict
from datetime import datetime

import requests
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("llm_judge.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Best free reasoning models on OpenRouter – April 2026
# Primary: qwen3-32b (strong reasoning); fallbacks in descending preference
JUDGE_MODELS = [
    "qwen/qwen3.6-plus:free",
    "stepfun/step-3.5-flash:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
]

CONFIDENCE_PROBS = {
    "Very Likely": 0.90,
    "Likely": 0.70,
    "Possible": 0.50,
    "Unlikely": 0.25,
    "Very Unlikely": 0.10,
}

JUDGE_SCORE_WEIGHTS = {
    "symptom_evidence": 0.45,
    "disease_symptom_rag": 0.40,
    "confidence_calibration": 0.15,
}

DEFAULT_RAG_CONFIG = {
    "faiss_path": ".rag",
    "model_name": "pritamdeka/S-PubMedBert-MS-MARCO",
    "embedding_type": "sentence_transformer",
}


# ===========================================================================
# OpenRouter Client
# ===========================================================================
class OpenRouterClient:
    """OpenRouter wrapper with multi-model fallback and JSON parsing."""

    def __init__(self, api_key: str, models: List[str] = None):
        self.api_key = api_key
        self.models = models or JUDGE_MODELS
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "FYP LLM Judge",
        }

    def chat(
        self,
        prompt: str,
        system: str = "You are a rigorous medical AI evaluator. Always respond in valid JSON.",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> Optional[str]:
        """Try each model in order; return the first successful response."""
        for model in self.models:
            for attempt in range(max_retries):
                try:
                    payload = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    resp = requests.post(
                        OPENROUTER_API_URL,
                        headers=self.headers,
                        json=payload,
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        content = (
                            resp.json()
                            .get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )
                        if content and content.strip():
                            logger.info("Model %s responded (attempt %d)", model, attempt + 1)
                            return content.strip()
                    elif resp.status_code == 429:
                        logger.warning("Rate limit on %s. Waiting 10 s ...", model)
                        time.sleep(10)
                    else:
                        logger.warning("Model %s failed: %s", model, resp.status_code)
                        break  # try next model
                except Exception as exc:
                    logger.warning("Error with %s attempt %d: %s", model, attempt + 1, exc)
                    if attempt < max_retries - 1:
                        time.sleep(3)
        logger.error("All models exhausted.")
        return None

    @staticmethod
    def parse_json(response: str) -> Optional[Dict]:
        """Robustly extract JSON from an LLM response."""
        if not response:
            return None
        # 1) direct
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        # 2) markdown code block
        for pat in [
            r"```json\s*([\s\S]+?)\s*```",
            r"```\s*([\s\S]+?)\s*```",
            r"(\{[\s\S]+\})",
        ]:
            m = re.search(pat, response, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass
        logger.error("Cannot parse JSON from response: %s", response[:300])
        return None


# ===========================================================================
# Data Statistics Engine  (algorithmic ground-truth from data1.json)
# ===========================================================================
class DataStatisticsEngine:
    """
    Computes verifiable ground-truth statistics from raw data1.json.

    Algorithms
    ----------
    * Per-location counts and durations (mean, per-day average)
    * Sleep/wake pattern extraction (first/last activity per day)
    * Late-night (>= 22:00) and early-morning (<= 06:00) activity flags
    * Numeric claim verifier: compares LLM evidence text against real stats
    """

    def __init__(self, raw_data: Dict):
        self.raw_data = raw_data
        self.stats = self._compute_stats()

    # ------------------------------------------------------------------
    def _compute_stats(self) -> Dict:
        stats: Dict = {"per_day": {}, "global": {}}
        global_loc: Dict = defaultdict(lambda: {"total_entries": 0, "total_duration": 0.0})

        for date, entries_list in self.raw_data.items():
            day_locs: Dict = defaultdict(
                lambda: {"count": 0, "total_duration": 0.0, "entry_durations": []}
            )
            all_times: List[int] = []

            for entry in entries_list:
                loc = entry.get("location", "Unknown")
                dur = float(entry.get("durationMins", entry.get("duration_mins", 0)))
                start = entry.get("start", "")

                day_locs[loc]["count"] += 1
                day_locs[loc]["total_duration"] += dur
                day_locs[loc]["entry_durations"].append(dur)
                global_loc[loc]["total_entries"] += 1
                global_loc[loc]["total_duration"] += dur

                if start:
                    ts = start.split(" ")[-1] if " " in start else start[-4:]
                    try:
                        t = int(ts[:4]) if len(ts) >= 4 else int(ts)
                        all_times.append(t)
                    except ValueError:
                        pass

            sorted_t = sorted(all_times)
            late_night = [t for t in all_times if t >= 2200 or t <= 59]
            early_morn = [t for t in all_times if 0 <= t <= 600]

            stats["per_day"][date] = {
                "locations": dict(day_locs),
                "first_activity": sorted_t[0] if sorted_t else None,
                "last_activity": sorted_t[-1] if sorted_t else None,
                "total_entries": len(entries_list),
                "late_night_entries": len(late_night),
                "early_morning_entries": len(early_morn),
                "has_late_night": len(late_night) > 0,
                "has_early_morning": len(early_morn) > 0,
            }

        n_days = max(len(stats["per_day"]), 1)
        for loc, g in global_loc.items():
            g["avg_entries_per_day"] = g["total_entries"] / n_days
            g["avg_duration_per_entry"] = g["total_duration"] / max(g["total_entries"], 1)
        stats["global"] = dict(global_loc)
        return stats

    # ------------------------------------------------------------------
    def verify_evidence_claims(self, evidence_text: str) -> Dict:
        """
        Extract numeric claims from collectiveEvidence and cross-check
        against ground-truth statistics.

        Returns verified / failed / unverifiable lists plus a 0-1 score.
        """
        res: Dict = {
            "verified": [],
            "failed": [],
            "unverifiable": [],
            "verification_score": 0.0,
        }

        def _chk(name: str, claimed: float, actual: float, tol: float = 0.15) -> None:
            if actual <= 0:
                res["unverifiable"].append(f"{name}: no ground-truth data")
                return
            ok = abs(claimed - actual) / actual <= tol
            bucket = "verified" if ok else "failed"
            res[bucket].append(
                f"{name}: claimed={claimed:.1f}, actual={actual:.1f} "
                f"({'OK' if ok else 'FAIL'})"
            )

        # Kitchen entries / day
        k_avg = self.stats["global"].get("Kitchen", {}).get("avg_entries_per_day", 0)
        m = re.search(r"[Kk]itchen\s+averages?\s+(\d+(?:\.\d+)?)\s+entries", evidence_text)
        if m:
            _chk("Kitchen avg entries/day", float(m.group(1)), k_avg)

        # Lounge entries / day
        l_avg = self.stats["global"].get("Lounge", {}).get("avg_entries_per_day", 0)
        m = re.search(r"[Ll]ounge\s+averages?\s+(\d+(?:\.\d+)?)\s+entries", evidence_text)
        if m:
            _chk("Lounge avg entries/day", float(m.group(1)), l_avg)

        # Hallway entries / day
        h_avg = self.stats["global"].get("Hallway", {}).get("avg_entries_per_day", 0)
        m = re.search(r"[Hh]allway\s+averages?\s+(\d+(?:\.\d+)?)\s+entries", evidence_text)
        if m:
            _chk("Hallway avg entries/day", float(m.group(1)), h_avg)

        # Late-night consistency claim
        late_days = sum(1 for d in self.stats["per_day"].values() if d["has_late_night"])
        total_days = len(self.stats["per_day"])
        m = re.search(r"(consistent|all)\s+\d+\s+days.{0,60}past\s+22", evidence_text, re.I)
        if m:
            if late_days == total_days:
                res["verified"].append(f"Late-night all {total_days} days: OK")
            else:
                res["failed"].append(
                    f"Late-night claim: only {late_days}/{total_days} days have activity"
                )

        # Duration claims  e.g. "35-minute duration"
        dur_claims = re.findall(
            r"(\d+)-minute\s+(?:duration|visit|entry|stay|activity|bathroom)",
            evidence_text,
            re.I,
        )
        for claim in dur_claims:
            cd = float(claim)
            found = any(
                any(abs(d - cd) <= 2 for d in ls["entry_durations"])
                for day in self.stats["per_day"].values()
                for ls in day["locations"].values()
            )
            bucket = "verified" if found else "unverifiable"
            res[bucket].append(f"{claim}-min duration: {'found in data' if found else 'not found'}")

        total = len(res["verified"]) + len(res["failed"])
        res["verification_score"] = len(res["verified"]) / max(total, 1)
        return res

    # ------------------------------------------------------------------
    def get_summary(self) -> str:
        """Compact text summary suitable for LLM prompts."""
        lines = ["=== GROUND-TRUTH DATA STATISTICS ==="]
        lines.append(f"Total days: {len(self.stats['per_day'])}")
        for loc, g in sorted(
            self.stats["global"].items(), key=lambda x: -x[1]["total_entries"]
        ):
            lines.append(
                f"  {loc}: {g['avg_entries_per_day']:.1f} entries/day avg, "
                f"{g['avg_duration_per_entry']:.1f} min/entry avg"
            )
        lines.append("\nPer-day sleep/activity patterns:")
        for date, d in sorted(self.stats["per_day"].items()):
            lines.append(
                f"  {date}: first={d['first_activity']}, last={d['last_activity']}, "
                f"late_night_entries={d['late_night_entries']}, "
                f"early_morn_entries={d['early_morning_entries']}"
            )
        return "\n".join(lines)


# ===========================================================================
# RAG Interface  (reuses existing FAISS setup from server.py)
# ===========================================================================
class RAGInterface:
    """
    Wraps the project's existing FAISS RAG.
    Uses the same AutoEmbedding + CrossEncoder reranker as server.py.
    Falls back gracefully when the database is unavailable.
    """

    def __init__(self, config: Dict = None):
        self.config = config or DEFAULT_RAG_CONFIG
        self.available = False
        self._load_rag()

    def _load_rag(self) -> None:
        try:
            import torch
            from auto_embed import AutoEmbedding
            from langchain_community.vectorstores import FAISS
            from sentence_transformers import CrossEncoder

            device = "cuda" if torch.cuda.is_available() else "cpu"
            embeddings = AutoEmbedding(
                self.config["model_name"],
                self.config["embedding_type"],
                model_kwargs={"device": device},
            )
            self.rag_db = FAISS.load_local(
                self.config["faiss_path"],
                embeddings,
                allow_dangerous_deserialization=True,
            )
            self.retriever = self.rag_db.as_retriever(search_kwargs={"k": 10})
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            self.available = True
            logger.info("FAISS RAG loaded from %s", self.config["faiss_path"])
        except Exception as exc:
            logger.warning("FAISS RAG unavailable (%s). Using LLM-only mode.", exc)

    def retrieve_for_disease(self, disease: str, symptoms: List[str]) -> List[str]:
        """
        Retrieve and rerank top medical passages for a disease + symptom set.
        """
        if not self.available:
            return []

        queries = [
            f"{disease} clinical symptoms diagnosis",
            f"{disease} behavioral symptoms sleep activity",
            f"differential diagnosis {disease} {' '.join(symptoms[:2])}",
        ] + [f"{disease} {s}" for s in symptoms[:3]]

        seen: Dict = {}
        for q in queries:
            try:
                docs = self.retriever.invoke(q)
                for doc in docs:
                    h = hash(doc.page_content[:200])
                    if h not in seen:
                        seen[h] = doc
            except Exception as exc:
                logger.warning("Retrieval failed for '%s': %s", q, exc)

        if not seen:
            return []

        docs_list = list(seen.values())
        query_text = f"{disease}: {', '.join(symptoms)}"
        pairs = [(query_text, doc.page_content) for doc in docs_list]
        try:
            scores = self.reranker.predict(pairs)
            ranked = sorted(zip(scores, docs_list), key=lambda x: -x[0])
            return [doc.page_content[:800] for _, doc in ranked[:5]]
        except Exception as exc:
            logger.warning("Reranking failed: %s", exc)
            return [doc.page_content[:800] for doc in docs_list[:5]]


# ===========================================================================
# Score 1: Symptom-Evidence Judge
# ===========================================================================
_SYMPTOM_RUBRIC = """
You are evaluating whether a detected symptom and its reported evidence
are genuinely supported by the actual activity data.

Evaluate on FOUR criteria (0-10 each):

1. EVIDENCE_ACCURACY
   Are numeric claims in collectiveEvidence accurate vs. actual statistics?
   10 = all numbers match (±15%); 5 = partial accuracy; 0 = fabricated

2. PATTERN_VALIDITY
   Does the described behavioral pattern genuinely appear in the data?
   10 = clearly and consistently present; 0 = not present / hallucinated

3. CONFIDENCE_APPROPRIATENESS
   Is the confidence label (Very Likely / Likely / Possible / ...) correct?
   10 = perfectly calibrated; 5 = one level off; 0 = severely wrong

4. CLINICAL_RELEVANCE
   Is this pattern clinically meaningful AND derivable from home monitoring?
   10 = strong relevance; 5 = moderate; 0 = meaningless or impossible

Return ONLY valid JSON (no markdown, no commentary):
{
  "evidence_accuracy": <0-10>,
  "pattern_validity": <0-10>,
  "confidence_appropriateness": <0-10>,
  "clinical_relevance": <0-10>,
  "evidence_issues": ["issue1", "issue2"],
  "strengths": ["strength1"],
  "confidence_verdict": "correct|overestimated|underestimated",
  "suggested_confidence": "Very Likely|Likely|Possible|Unlikely|Very Unlikely",
  "reasoning": "<2-3 sentences>"
}
"""


class SymptomEvidenceJudge:
    """
    Score 1: G-Eval-style rubric scoring for each detected symptom.

    Algorithm
    ---------
    1. DataStatisticsEngine.verify_evidence_claims() – algorithmic numeric check
    2. LLM rubric scoring (evidence_accuracy, pattern_validity, …)
    3. Final score = 0.70 * LLM_composite + 0.30 * statistical_verification
    """

    _WEIGHTS = {
        "evidence_accuracy": 0.30,
        "pattern_validity": 0.30,
        "confidence_appropriateness": 0.20,
        "clinical_relevance": 0.20,
    }

    def __init__(self, client: OpenRouterClient, data_engine: DataStatisticsEngine):
        self.client = client
        self.data_engine = data_engine

    def judge_symptom(self, symptom: Dict) -> Dict:
        activity = symptom.get("Abnormal Activity", "Unknown")
        confidence = symptom.get("confidence", "Unknown")
        definition = symptom.get("definition", "")
        explanation = symptom.get("explanation", "")
        evidence = symptom.get("collectiveEvidence", "")

        # Algorithmic verification
        stat_verify = self.data_engine.verify_evidence_claims(evidence)

        # LLM evaluation
        prompt = (
            _SYMPTOM_RUBRIC
            + f"\n\n=== SYMPTOM BEING EVALUATED ===\n"
            + f"Name: {activity}\n"
            + f"Confidence: {confidence}\n"
            + f"Definition: {definition}\n"
            + f"Explanation: {explanation}\n"
            + f"Collective Evidence: {evidence}\n\n"
            + f"=== ACTUAL DATA STATISTICS ===\n{self.data_engine.get_summary()}\n\n"
            + f"=== ALGORITHMIC PRE-VERIFICATION ===\n"
            + f"Verified: {stat_verify['verified']}\n"
            + f"Failed  : {stat_verify['failed']}\n"
            + f"N/A     : {stat_verify['unverifiable']}\n"
            + f"Auto score: {stat_verify['verification_score']:.2f}\n"
        )
        raw = self.client.chat(prompt, max_tokens=1500)
        result: Dict = self.client.parse_json(raw) or {}

        result["statistical_verification"] = stat_verify
        result["symptom_name"] = activity
        result["original_confidence"] = confidence

        llm_score = sum(
            result.get(k, 5.0) * w for k, w in self._WEIGHTS.items()
        )
        final_01 = 0.70 * llm_score / 10.0 + 0.30 * stat_verify["verification_score"]
        result["final_score_0_10"] = round(final_01 * 10, 2)
        result["final_score_0_1"] = round(final_01, 3)
        return result

    def judge_all(self, symptoms: List[Dict]) -> Dict:
        results = []
        for sym in symptoms:
            logger.info("Score-1 judging: %s", sym.get("Abnormal Activity", "?")[:60])
            results.append(self.judge_symptom(sym))
            time.sleep(1)
        scores = [r.get("final_score_0_10", 5.0) for r in results]
        return {
            "symptom_results": results,
            "aggregate_score_0_10": round(float(np.mean(scores)), 2),
            "min_score": round(float(np.min(scores)), 2),
            "max_score": round(float(np.max(scores)), 2),
            "n_symptoms_evaluated": len(results),
        }


# ===========================================================================
# Score 2: Disease-Symptom RAG Judge
# ===========================================================================
_DISEASE_RUBRIC = """
You are a clinical expert evaluating whether a disease prediction is
medically justified given the patient's detected symptoms AND supporting
medical literature retrieved from a RAG knowledge base.

Evaluate on FIVE criteria (0-10 each):

1. SYMPTOM_DISEASE_ALIGNMENT
   Do the related symptoms clinically match this disease's known presentation?
   10 = textbook match; 5 = partial; 0 = contradictory

2. RAG_EVIDENCE_SUPPORT
   Does the retrieved medical literature confirm this disease-symptom link?
   10 = explicit and strong; 5 = partial; 0 = contradicts or absent

3. REASONING_QUALITY
   Is the clinical reasoning specific, mechanistic, and evidence-based?
   10 = excellent; 5 = generic; 0 = flawed/vague

4. CONFIDENCE_CALIBRATION
   Is the disease-level confidence label appropriate?
   10 = perfect; 5 = defensible; 0 = severely miscalibrated

5. DIFFERENTIAL_VALIDITY
   Is this disease plausibly diagnosable from home behavioral monitoring alone?
   10 = strongly plausible; 5 = possible with caveats; 0 = not inferrable

Return ONLY valid JSON:
{
  "symptom_disease_alignment": <0-10>,
  "rag_evidence_support": <0-10>,
  "reasoning_quality": <0-10>,
  "confidence_calibration": <0-10>,
  "differential_validity": <0-10>,
  "key_supporting_evidence": ["point from RAG"],
  "gaps_or_concerns": ["concern"],
  "confidence_verdict": "correct|overestimated|underestimated",
  "suggested_confidence": "Very Likely|Likely|Possible|Unlikely|Very Unlikely",
  "reasoning": "<2-3 sentences>"
}
"""


class DiseaseSymptomRAGJudge:
    """
    Score 2: RAG-augmented evaluation of each disease prediction.

    Algorithm
    ---------
    1. Symptom coverage metric: fraction of all symptoms claimed by disease
    2. Weighted symptom confidence aggregation
    3. RAG retrieval + CrossEncoder reranking (via RAGInterface)
    4. LLM rubric scoring with RAG context injected
    5. Final = 0.80 * LLM_composite + 0.20 * coverage_score
    """

    _WEIGHTS = {
        "symptom_disease_alignment": 0.30,
        "rag_evidence_support": 0.25,
        "reasoning_quality": 0.20,
        "confidence_calibration": 0.15,
        "differential_validity": 0.10,
    }

    def __init__(self, client: OpenRouterClient, rag: RAGInterface):
        self.client = client
        self.rag = rag

    def _coverage_score(
        self, predicted: List[str], all_syms: List[str]
    ) -> float:
        if not all_syms:
            return 0.0
        covered = {s.lower() for s in predicted}
        total = {s.lower() for s in all_syms}
        return len(covered & total) / len(total)

    def judge_disease(
        self,
        disease: Dict,
        all_symptoms: List[str],
        sym_confidences: Dict[str, str],
    ) -> Dict:
        name = disease.get("disease", "Unknown")
        related = disease.get("relatedAbnormalActivities", [])
        reasoning = disease.get("reasoning", "")
        confidence = disease.get("confidence", "Unknown")

        coverage = self._coverage_score(related, all_symptoms)
        conf_vals = [
            CONFIDENCE_PROBS.get(sym_confidences.get(s, "Possible"), 0.50)
            for s in related
        ]
        avg_sym_conf = float(np.mean(conf_vals)) if conf_vals else 0.50

        # RAG retrieval
        rag_docs = self.rag.retrieve_for_disease(name, related)
        rag_ctx = (
            "\n\n---\n\n".join(rag_docs[:3])
            if rag_docs
            else "No RAG documents retrieved. Use general medical knowledge."
        )

        prompt = (
            _DISEASE_RUBRIC
            + f"\n\n=== DISEASE PREDICTION ===\n"
            + f"Disease: {name}\n"
            + f"Confidence: {confidence}\n"
            + f"Related Symptoms: {json.dumps(related, indent=2)}\n"
            + f"Reasoning: {reasoning}\n\n"
            + f"=== SYMPTOM CONFIDENCE MAP ===\n"
            + json.dumps({s: sym_confidences.get(s, "?") for s in related}, indent=2)
            + f"\nSymptom coverage of all detected symptoms: {coverage:.1%}\n"
            + f"Weighted avg symptom confidence: {avg_sym_conf:.2f}\n\n"
            + f"=== RAG MEDICAL LITERATURE ===\n{rag_ctx}\n"
        )
        raw = self.client.chat(prompt, max_tokens=1500)
        result: Dict = self.client.parse_json(raw) or {}

        result["disease_name"] = name
        result["original_confidence"] = confidence
        result["symptom_coverage"] = round(coverage, 3)
        result["weighted_symptom_confidence"] = round(avg_sym_conf, 3)
        result["rag_docs_retrieved"] = len(rag_docs)

        llm_score = sum(result.get(k, 5.0) * w for k, w in self._WEIGHTS.items())
        final = 0.80 * llm_score + 0.20 * coverage * 10.0
        result["final_score_0_10"] = round(final, 2)
        result["final_score_0_1"] = round(final / 10.0, 3)
        return result

    def judge_all(self, diseases: List[Dict], symptoms: List[Dict]) -> Dict:
        all_sym_names = [s.get("Abnormal Activity", "") for s in symptoms]
        sym_confs = {
            s.get("Abnormal Activity", ""): s.get("confidence", "Possible")
            for s in symptoms
        }
        results = []
        for d in diseases:
            logger.info("Score-2 judging: %s", d.get("disease", "?")[:60])
            results.append(self.judge_disease(d, all_sym_names, sym_confs))
            time.sleep(1)
        scores = [r.get("final_score_0_10", 5.0) for r in results]
        return {
            "disease_results": results,
            "aggregate_score_0_10": round(float(np.mean(scores)), 2),
            "min_score": round(float(np.min(scores)), 2),
            "max_score": round(float(np.max(scores)), 2),
            "n_diseases_evaluated": len(results),
        }


# ===========================================================================
# Score 3: Confidence Calibration Analyser
# ===========================================================================
class ConfidenceCalibrationAnalyser:
    """
    Algorithmic confidence calibration check.

    Algorithm
    ---------
    1. Evidence Strength (ES): quantifies raw data support per symptom
       ES = f(multi-day evidence, numeric density, location specificity)
    2. For diseases: use LLM judge score from Score-2 as proxy probability
    3. Calibration Error = |assigned_prob - estimated_true_prob|
    4. Expected Calibration Error (ECE) = mean over all items
    5. Calibration Score (0-10) = max(0, 10 - ECE * 20)
    """

    def __init__(self, data_engine: DataStatisticsEngine):
        self.data_engine = data_engine

    def _evidence_strength(self, symptom: Dict) -> float:
        """Compute algorithmic evidence strength in [0, 1]."""
        evidence = symptom.get("collectiveEvidence", "")
        explanation = symptom.get("explanation", "")
        combined = evidence + " " + explanation

        score = 0.50  # base

        # Multi-day evidence
        day_refs = set(re.findall(r"Day\s+\d+", combined, re.I))
        score += min(len(day_refs) * 0.10, 0.20)

        # Numeric density
        nums = re.findall(r"\d+(?:\.\d+)?", evidence)
        if len(nums) >= 3:
            score += 0.10
        if len(nums) >= 6:
            score += 0.10

        # Location mentions
        locs = ["Kitchen", "Lounge", "Bedroom", "Bathroom", "Fridge", "Hallway"]
        loc_count = sum(1 for loc in locs if loc.lower() in evidence.lower())
        score += min(loc_count * 0.05, 0.15)

        # Penalise thin evidence
        if len(evidence) < 80:
            score -= 0.15

        return min(max(score, 0.0), 1.0)

    def analyse(
        self,
        symptoms: List[Dict],
        sym_results: List[Dict],
        dis_results: List[Dict],
    ) -> Dict:
        items = []

        for sym, jr in zip(symptoms, sym_results):
            conf = sym.get("confidence", "Possible")
            assigned_p = CONFIDENCE_PROBS.get(conf, 0.50)
            es = self._evidence_strength(sym)
            judge_01 = jr.get("final_score_0_1", 0.50)
            est_p = 0.60 * es + 0.40 * judge_01
            err = abs(est_p - assigned_p)
            items.append({
                "name": sym.get("Abnormal Activity", "?")[:55],
                "type": "symptom",
                "assigned_confidence": conf,
                "assigned_probability": assigned_p,
                "evidence_strength": round(es, 3),
                "judge_score_0_1": round(judge_01, 3),
                "estimated_true_probability": round(est_p, 3),
                "calibration_error": round(err, 3),
                "verdict": jr.get("confidence_verdict", "correct"),
                "suggested": jr.get("suggested_confidence", conf),
            })

        for dr in dis_results:
            conf = dr.get("original_confidence", "Possible")
            assigned_p = CONFIDENCE_PROBS.get(conf, 0.50)
            judge_01 = dr.get("final_score_0_1", 0.50)
            err = abs(judge_01 - assigned_p)
            items.append({
                "name": dr.get("disease_name", "?")[:55],
                "type": "disease",
                "assigned_confidence": conf,
                "assigned_probability": assigned_p,
                "judge_score_0_1": round(judge_01, 3),
                "estimated_true_probability": round(judge_01, 3),
                "calibration_error": round(err, 3),
                "verdict": dr.get("confidence_verdict", "correct"),
                "suggested": dr.get("suggested_confidence", conf),
            })

        ece = float(np.mean([i["calibration_error"] for i in items])) if items else 0.0
        calib_score = max(0.0, 10.0 - ece * 20.0)

        by_level: Dict = defaultdict(list)
        for i in items:
            by_level[i["assigned_confidence"]].append(i["calibration_error"])

        misclassified = [
            i for i in items
            if i["verdict"] != "correct" or i["calibration_error"] > 0.30
        ]

        return {
            "calibration_items": items,
            "expected_calibration_error": round(ece, 3),
            "calibration_score_0_10": round(calib_score, 2),
            "level_breakdown": {
                k: {"count": len(v), "mean_error": round(float(np.mean(v)), 3)}
                for k, v in by_level.items()
            },
            "misclassified_count": len(misclassified),
            "misclassified_items": misclassified,
            "total_evaluated": len(items),
        }


# ===========================================================================
# Main Orchestrator
# ===========================================================================
_SUMMARY_SYSTEM = (
    "You are a senior clinical AI evaluator writing a final assessment report. "
    "Be concise, specific, and clinically constructive. "
    "Always respond in valid JSON."
)

_SUMMARY_PROMPT_TEMPLATE = """
You are providing a final summary for a medical AI pipeline evaluation.

SCORES:
- Symptom-Evidence Score : {s1:.2f}/10
- Disease-RAG Score      : {s2:.2f}/10
- Calibration Score      : {s3:.2f}/10
- Overall Weighted Score : {overall:.2f}/10

FLAGGED SYMPTOM ISSUES (score < 6):
{sym_issues}

FLAGGED DISEASE ISSUES (score < 6):
{dis_issues}

Provide a concise clinical assessment in JSON:
{{
  "overall_quality": "Excellent|Good|Acceptable|Poor|Unacceptable",
  "key_strengths": ["s1", "s2", "s3"],
  "key_weaknesses": ["w1", "w2", "w3"],
  "clinical_safety_concerns": [],
  "improvement_recommendations": ["r1", "r2", "r3"],
  "deployment_readiness": "Ready|Needs Improvement|Not Ready",
  "summary": "<3-4 sentences>"
}}
"""


class MedicalLLMJudge:
    """
    Orchestrator: runs all three scoring modules and produces a
    comprehensive evaluation JSON + human-readable text report.
    """

    def __init__(
        self,
        api_key: str,
        rag_config: Dict = None,
        use_rag: bool = True,
    ):
        self.client = OpenRouterClient(api_key, JUDGE_MODELS)
        rag_cfg = rag_config or DEFAULT_RAG_CONFIG
        self.rag = RAGInterface(rag_cfg) if use_rag else RAGInterface.__new__(RAGInterface)
        if not use_rag:
            self.rag.available = False

    def evaluate(self, data1_path: str, out1_path: str) -> Dict:
        logger.info("=" * 60)
        logger.info("LLM JUDGE EVALUATION STARTED")
        logger.info("=" * 60)

        with open(data1_path, encoding="utf-8") as f:
            raw_data = json.load(f)
        with open(out1_path, encoding="utf-8") as f:
            output = json.load(f)

        # 兼容 snake_case, lowercase 和原始大驼峰 key
        symptoms = (
            output.get("enhanced_symptoms") or 
            output.get("enhancedSymptoms") or 
            output.get("enhancedsymptoms", [])
        )
        diseases = (
            output.get("disease_predictions") or 
            output.get("diseasePredictions") or 
            output.get("diseasepredictions", [])
        )
        metadata: Dict = output.get("analysisMetadata", {})

        logger.info("Symptoms: %d  |  Diseases: %d", len(symptoms), len(diseases))

        data_engine = DataStatisticsEngine(raw_data)

        # Score 1
        logger.info("\n--- Score 1: Symptom-Evidence Validation ---")
        s1_judge = SymptomEvidenceJudge(self.client, data_engine)
        s1_res = s1_judge.judge_all(symptoms)

        # Score 2
        logger.info("\n--- Score 2: Disease-Symptom RAG Validation ---")
        s2_judge = DiseaseSymptomRAGJudge(self.client, self.rag)
        s2_res = s2_judge.judge_all(diseases, symptoms)

        # Score 3
        logger.info("\n--- Score 3: Confidence Calibration ---")
        calib = ConfidenceCalibrationAnalyser(data_engine)
        calib_res = calib.analyse(
            symptoms, s1_res["symptom_results"], s2_res["disease_results"]
        )

        # Overall
        w = JUDGE_SCORE_WEIGHTS
        s1 = s1_res["aggregate_score_0_10"]
        s2 = s2_res["aggregate_score_0_10"]
        s3 = calib_res["calibration_score_0_10"]
        overall = s1 * w["symptom_evidence"] + s2 * w["disease_symptom_rag"] + s3 * w["confidence_calibration"]

        # LLM overall summary
        sym_issues = [
            {"symptom": r["symptom_name"], "score": r.get("final_score_0_10")}
            for r in s1_res["symptom_results"]
            if r.get("final_score_0_10", 10) < 6.0
        ]
        dis_issues = [
            {"disease": r["disease_name"], "score": r.get("final_score_0_10")}
            for r in s2_res["disease_results"]
            if r.get("final_score_0_10", 10) < 6.0
        ]
        summary_prompt = _SUMMARY_PROMPT_TEMPLATE.format(
            s1=s1, s2=s2, s3=s3, overall=overall,
            sym_issues=json.dumps(sym_issues, indent=2),
            dis_issues=json.dumps(dis_issues, indent=2),
        )
        summary_raw = self.client.chat(summary_prompt, system=_SUMMARY_SYSTEM, max_tokens=1200)
        summary = self.client.parse_json(summary_raw) or {"summary": "Summary unavailable."}

        report = {
            "judge_metadata": {
                "evaluation_timestamp": datetime.now().isoformat(),
                "data_file": data1_path,
                "output_file": out1_path,
                "judge_models": JUDGE_MODELS,
                "rag_available": self.rag.available,
                "pipeline_metadata": metadata,
            },
            "scores": {
                "score1_symptom_evidence": {
                    "score_0_10": s1,
                    "weight": w["symptom_evidence"],
                    "weighted_contribution": round(s1 * w["symptom_evidence"], 3),
                    "description": "Validates symptom detection vs raw activity data",
                },
                "score2_disease_symptom_rag": {
                    "score_0_10": s2,
                    "weight": w["disease_symptom_rag"],
                    "weighted_contribution": round(s2 * w["disease_symptom_rag"], 3),
                    "description": "Validates disease predictions via RAG medical knowledge",
                },
                "score3_confidence_calibration": {
                    "score_0_10": s3,
                    "weight": w["confidence_calibration"],
                    "weighted_contribution": round(s3 * w["confidence_calibration"], 3),
                    "expected_calibration_error": calib_res["expected_calibration_error"],
                    "description": "Checks calibration of all confidence labels",
                },
                "overall_score_0_10": round(overall, 2),
                "overall_score_pct": f"{overall * 10:.1f}%",
                "grade": _grade(overall),
            },
            "symptom_evaluation": s1_res,
            "disease_evaluation": s2_res,
            "confidence_calibration": calib_res,
            "overall_summary": summary,
        }
        return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _grade(score: float) -> str:
    if score >= 9:
        return "A+ (Excellent)"
    if score >= 8:
        return "A (Very Good)"
    if score >= 7:
        return "B (Good)"
    if score >= 6:
        return "C (Acceptable)"
    if score >= 5:
        return "D (Needs Improvement)"
    return "F (Poor)"


def format_report(report: Dict) -> str:
    """Generate a human-readable text report."""
    sep = "=" * 70
    thin = "-" * 70
    lines = [
        sep,
        "   FYP MEDICAL ANALYSIS SYSTEM  —  LLM JUDGE EVALUATION REPORT",
        sep,
        f"Evaluated : {report['judge_metadata']['evaluation_timestamp']}",
        f"RAG       : {'Available' if report['judge_metadata']['rag_available'] else 'Unavailable (LLM-only mode)'}",
        "",
    ]

    sc = report["scores"]
    lines += [
        thin, "SCORE SUMMARY", thin,
        f"  Score 1  Symptom-Evidence Validation   : {sc['score1_symptom_evidence']['score_0_10']:.2f}/10  (weight {sc['score1_symptom_evidence']['weight']:.0%})",
        f"  Score 2  Disease-Symptom RAG Validation : {sc['score2_disease_symptom_rag']['score_0_10']:.2f}/10  (weight {sc['score2_disease_symptom_rag']['weight']:.0%})",
        f"  Score 3  Confidence Calibration         : {sc['score3_confidence_calibration']['score_0_10']:.2f}/10  (weight {sc['score3_confidence_calibration']['weight']:.0%})",
        f"  {'─'*50}",
        f"  OVERALL SCORE  :  {sc['overall_score_0_10']:.2f}/10  [{sc['grade']}]",
        "",
    ]

    lines += [thin, "SYMPTOM-LEVEL RESULTS", thin]
    for r in report["symptom_evaluation"]["symptom_results"]:
        v = r.get("confidence_verdict", "correct")
        sug = r.get("suggested_confidence", "")
        lines.append(f"  [{r.get('final_score_0_10',0):4.1f}/10]  {r.get('symptom_name','?')[:58]}")
        conf_line = f"           Confidence: {r.get('original_confidence','?')} → {v}"
        if v != "correct":
            conf_line += f"  (suggest: {sug})"
        lines.append(conf_line)
        for issue in r.get("evidence_issues", [])[:2]:
            lines.append(f"           ⚠ {issue}")
    lines.append("")

    lines += [thin, "DISEASE-LEVEL RESULTS", thin]
    for r in report["disease_evaluation"]["disease_results"]:
        v = r.get("confidence_verdict", "correct")
        lines.append(f"  [{r.get('final_score_0_10',0):4.1f}/10]  {r.get('disease_name','?')[:58]}")
        lines.append(
            f"           Coverage: {r.get('symptom_coverage',0):.0%} of symptoms | "
            f"RAG docs: {r.get('rag_docs_retrieved',0)}"
        )
        conf_line = f"           Confidence: {r.get('original_confidence','?')} → {v}"
        if v != "correct":
            conf_line += f"  (suggest: {r.get('suggested_confidence','')})"
        lines.append(conf_line)
        for concern in r.get("gaps_or_concerns", [])[:1]:
            lines.append(f"           ⚠ {concern}")
    lines.append("")

    cal = report["confidence_calibration"]
    lines += [
        thin, "CONFIDENCE CALIBRATION", thin,
        f"  ECE (Expected Calibration Error) : {cal['expected_calibration_error']:.3f}",
        f"  Calibration Score                : {cal['calibration_score_0_10']:.2f}/10",
        f"  Miscalibrated items              : {cal['misclassified_count']}/{cal['total_evaluated']}",
    ]
    for item in cal["misclassified_items"][:4]:
        lines.append(
            f"  → [{item['type']}] {item['name'][:42]} : "
            f"{item['assigned_confidence']} → suggest {item['suggested']}"
        )
    lines.append("")

    if "overall_summary" in report:
        s = report["overall_summary"]
        lines += [
            thin, "OVERALL ASSESSMENT", thin,
            f"  Quality            : {s.get('overall_quality','N/A')}",
            f"  Deployment Status  : {s.get('deployment_readiness','N/A')}",
            f"  Summary            : {s.get('summary','N/A')}",
        ]
        if s.get("key_strengths"):
            lines.append("\n  Strengths:")
            for st in s["key_strengths"]:
                lines.append(f"    + {st}")
        if s.get("key_weaknesses"):
            lines.append("\n  Weaknesses:")
            for wk in s["key_weaknesses"]:
                lines.append(f"    - {wk}")
        if s.get("improvement_recommendations"):
            lines.append("\n  Recommendations:")
            for rec in s["improvement_recommendations"]:
                lines.append(f"    → {rec}")
        if s.get("clinical_safety_concerns"):
            lines.append("\n  ⚠ CLINICAL SAFETY CONCERNS:")
            for c in s["clinical_safety_concerns"]:
                lines.append(f"    !! {c}")
    lines += ["", sep]
    return "\n".join(lines)


# ===========================================================================
# CLI Entry Point
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Judge for FYP Medical Activity Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", default="data1.json", help="Raw input data file")
    parser.add_argument("--output", default="out1.json", help="Pipeline output file")
    parser.add_argument("--api-key", default=None, help="OpenRouter API key")
    parser.add_argument("--rag-path", default=".rag", help="Path to FAISS RAG database")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG (LLM-only mode)")
    parser.add_argument("--report-json", default="judge_report.json", help="JSON output path")
    parser.add_argument("--report-txt", default="judge_report.txt", help="Text output path")
    args = parser.parse_args()

    api_key = "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"

    rag_config = {**DEFAULT_RAG_CONFIG, "faiss_path": args.rag_path}

    judge = MedicalLLMJudge(
        api_key=api_key,
        rag_config=rag_config,
        use_rag=not args.no_rag,
    )

    report = judge.evaluate(args.data, args.output)

    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("JSON report saved to %s", args.report_json)

    txt = format_report(report)
    with open(args.report_txt, "w", encoding="utf-8") as f:
        f.write(txt)
    logger.info("Text report saved to %s", args.report_txt)

    print("\n" + txt)


if __name__ == "__main__":
    main()
