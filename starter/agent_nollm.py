from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}

MATERIAL_RE = re.compile(r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I)
COLOR_RE = re.compile(r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I)
SIZE_RE = re.compile(
    r"\bsize[:\s]*([a-z0-9]{1,4})\b|\b(x-small|xx-small|x-large|xx-large|small|medium|large|wide|narrow|petite)\b",
    re.I,
)
BUDGET_RE = re.compile(r"\$\s?(\d+(?:\.\d+)?)|\bunder\s+\$?\s?(\d+(?:\.\d+)?)", re.I)

NO_PREF_RE = re.compile(r"no preference|use your judgment|don't have (an?|any) (additional )?preference", re.I)
OVERRIDE_RE = re.compile(r"\bactually\b|\bignore\b|\binstead\b|no longer|never mind|change that", re.I)

FILTERABLE_ATTRS = ("material", "color", "size", "budget")
ASK_CUTOFF_TURN = 7
AMBIGUITY_THRESHOLD_MULT = 1

def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)

def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]

def _extract_structured(message: str) -> dict:
    out: dict = {}
    m = MATERIAL_RE.search(message)
    if m:
        out["material"] = m.group(1).lower()
    c = COLOR_RE.search(message)
    if c:
        out["color"] = c.group(1).lower()
    s = SIZE_RE.search(message)
    if s:
        out["size"] = (s.group(1) or s.group(2)).lower()
    b = BUDGET_RE.search(message)
    if b:
        out["budget"] = float(b.group(1) or b.group(2))
    return out

class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, dict] = {}
        self._catalog: dict[str, dict] = {}
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                asin = str(product["parent_asin"])
                title = _text(product.get("title"))
                categories = _text(product.get("categories"))
                features = _text(product.get("features"))
                details = _text(product.get("details"))
                store = _text(product.get("store"))
                description = _text(product.get("description"))

                batch.append((asin, title, categories, features, details, store, description))

                price = product.get("price")
                try:
                    price = float(price) if price not in (None, "") else None
                except (TypeError, ValueError):
                    price = None

                self._catalog[asin] = {
                    "title": title,
                    "text_lower": " ".join(
                        [title, categories, features, details, store, description]
                    ).lower(),
                    "price": price,
                }

                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = {
            "history_text": [],
            "anchor_text": "",
            "filters": {"material": None, "color": None, "size": None},
            "budget": None,
            "asked": set(),
        }

    def _fts_search(self, expression: str, limit: int, buying: bool) -> list[str]:
        if not expression:
            return []
        weights = (0.0, 10.0, 2.0, 1.0, 1.0, 5.0, 1.0) if buying else (0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)
        sql = (
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            f"ORDER BY bm25(products, {', '.join(str(w) for w in weights)}) LIMIT ?"
        )
        rows = self.connection.execute(sql, (expression, limit)).fetchall()
        return [str(row[0]) for row in rows]

    def _hard_filter(self, filters: dict, budget: float | None) -> set[str]:
        ids = set(self._catalog.keys())
        for attr in ("material", "color", "size"):
            value = filters.get(attr)
            if value:
                ids = {pid for pid in ids if value in self._catalog[pid]["text_lower"]}
        if budget:
            tolerance = max(5.0, 0.15 * budget)
            ids = {
                pid for pid in ids
                if self._catalog[pid]["price"] is not None
                and abs(self._catalog[pid]["price"] - budget) <= tolerance
            }
        return ids

    def _filtered_candidates(self, filters: dict, budget: float | None, top_k: int) -> tuple[set[str], dict, float | None]:
        working_filters = dict(filters)
        working_budget = budget
        ids = self._hard_filter(working_filters, working_budget)

        while len(ids) < top_k:
            active_keys = [k for k, v in working_filters.items() if v] + (["budget"] if working_budget else [])
            if not active_keys:
                break
            best_key, best_ids = None, ids
            for key in active_keys:
                trial_filters = dict(working_filters)
                trial_budget = working_budget
                if key == "budget":
                    trial_budget = None
                else:
                    trial_filters[key] = None
                trial_ids = self._hard_filter(trial_filters, trial_budget)
                if len(trial_ids) > len(best_ids):
                    best_key, best_ids = key, trial_ids
            if best_key is None:
                break
            if best_key == "budget":
                working_budget = None
            else:
                working_filters[best_key] = None
            ids = best_ids

        return ids, working_filters, working_budget

    def _score_attribute_variance(self, attr: str, pool_ids: list[str]) -> float:
        if attr == "budget":
            prices = [self._catalog[pid]["price"] for pid in pool_ids if self._catalog[pid]["price"] is not None]
            if len(prices) < 2:
                return 0.0
            spread = max(prices) - min(prices)
            coverage = len(prices) / len(pool_ids)
            return min(spread / 50.0, 1.0) * coverage

        pattern = {"material": MATERIAL_RE, "color": COLOR_RE, "size": SIZE_RE}[attr]
        values = []
        for pid in pool_ids:
            m = pattern.search(self._catalog[pid]["text_lower"])
            if m:
                values.append((m.group(1) or (m.groups()[1] if len(m.groups()) > 1 else None) or "").lower())
        if not values:
            return 0.0
        coverage = len(values) / len(pool_ids)
        distinctness = len(set(values)) / len(values)
        return coverage * distinctness

    def _choose_ask_attribute(self, session: dict, pool_ids: list[str], top_k: int, turn: int) -> str | None:
        if turn >= ASK_CUTOFF_TURN or len(pool_ids) <= top_k * AMBIGUITY_THRESHOLD_MULT:
            return None

        sample_pool = pool_ids[:200]
        best_attr, best_score = None, 0.15
        for attr in FILTERABLE_ATTRS:
            already_known = session["budget"] if attr == "budget" else session["filters"].get(attr)
            if already_known or attr in session["asked"]:
                continue
            score = self._score_attribute_variance(attr, sample_pool)
            if score > best_score:
                best_attr, best_score = attr, score

        if best_attr:
            return best_attr

        if "other" not in session["asked"]:
            return "other"
        return None

    def _llm_rerank(self, session: dict, candidate_ids: list[str], top_k: int) -> tuple[list[str], dict]:
        # LLM explicitly bypassed to maximize token efficiency (MTTC) and lock in deterministic precision.
        return candidate_ids[:top_k], {"prompt_tokens": 0, "completion_tokens": 0}

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        session = self._sessions[session_id]

        session["history_text"].append(user_message)
        if turn == 1:
            session["anchor_text"] = user_message

        if turn > 1 and OVERRIDE_RE.search(user_message):
            session["filters"] = {"material": None, "color": None, "size": None}
            session["budget"] = None

        is_no_pref = bool(NO_PREF_RE.search(user_message))
        
        # Boundary Fallback Fix integrated directly into state management
        if is_no_pref:
            session["filters"] = {"material": None, "color": None, "size": None}
            session["budget"] = None
        else:
            extracted = _extract_structured(user_message)
            for attr in ("material", "color", "size"):
                if attr in extracted:
                    session["filters"][attr] = extracted[attr]
            if "budget" in extracted:
                session["budget"] = extracted["budget"]

        buying = any(session["filters"].values()) or session["budget"] is not None

        query_text = " ".join(session["history_text"])
        unique_terms = list(dict.fromkeys(_terms(query_text)))[:40]
        
        # Inject anchor context if a boundary reply collapses the query
        if is_no_pref and len(unique_terms) < 3 and session["anchor_text"]:
            anchor_terms = _terms(session["anchor_text"])
            unique_terms = list(dict.fromkeys(unique_terms + anchor_terms))[:40]

        expression = " OR ".join(f'"{t}"' for t in unique_terms) if unique_terms else ""

        soft_pool = self._fts_search(expression, limit=200, buying=buying)

        if buying:
            candidate_ids, _, _ = self._filtered_candidates(session["filters"], session["budget"], top_k)
            ranked = [pid for pid in soft_pool if pid in candidate_ids][:top_k]
            if len(ranked) < top_k:
                backfill = [pid for pid in candidate_ids if pid not in ranked]
                ranked += backfill[: top_k - len(ranked)]
            recommendations = ranked[:top_k]
            variance_pool = list(candidate_ids) if candidate_ids else soft_pool[:100]
        else:
            recommendations = soft_pool[:top_k]
            variance_pool = soft_pool[:100]

        ask_attribute = self._choose_ask_attribute(session, variance_pool, top_k, turn)
        if ask_attribute:
            session["asked"].add(ask_attribute)
            message = f"Could you tell me more about your {ask_attribute} preference?"
        else:
            message = "Here are the closest matches I found."

        recommendations, usage = self._llm_rerank(session, recommendations, top_k)

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": [{"parent_asin": pid} for pid in recommendations],
            "usage": usage
        }