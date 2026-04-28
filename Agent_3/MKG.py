from __future__ import annotations

import os
import re
import pandas as pd
from dotenv import load_dotenv

load_dotenv()   

_gliner_model = None
_gliner_model_id: str | None = None

_DEFAULT_GLINER_LABELS: tuple[str, ...] = (
    "disease",
    "disorder",
    "medication",
    "drug",
    "symptom",
    "sign",
    "procedure",
    "treatment",
    "diagnostic test",
    "lab test",
    "vital sign",
    "anatomical structure",
    "allergy",
)

_DEFAULT_CHUNK_CHARS = 2500


def _get_gliner(model_id: str | None = None):
    """Lazy-load GLiNER so PrimeKG usage doesn't pull HF weights."""
    global _gliner_model, _gliner_model_id
    mid = model_id or os.environ.get("MKG_GLINER_MODEL", "urchade/gliner_mediumv2.1")
    if _gliner_model is None or _gliner_model_id != mid:
        from gliner import GLiNER

        _gliner_model = GLiNER.from_pretrained(mid)
        _gliner_model_id = mid
    return _gliner_model


def _chunk_text(text: str, max_chars: int = _DEFAULT_CHUNK_CHARS) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    parts = re.split(r"(?<=[.!?\n])\s+", text)
    chunks: list[str] = []
    cur: list[str] = []
    n = 0
    for p in parts:
        if not p:
            continue
        sep = 1 if cur else 0
        if n + len(p) + sep > max_chars and cur:
            chunks.append(" ".join(cur))
            cur = [p]
            n = len(p)
        else:
            cur.append(p)
            n += len(p) + sep
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def extract_medical_terms(
    text: str,
    *,
    labels: tuple[str, ...] | None = None,
    threshold: float = 0.35,
    model_id: str | None = None,
    max_chunk_chars: int = _DEFAULT_CHUNK_CHARS,
) -> list[str]:
    """
    Extract unique entity surface forms (lowercased), preserving first-seen order, using GLiNER.

    Notes:
    - This is best-effort NER for prompting/querying (not a clinical-grade ontology mapper).
    - GLiNER weights are downloaded from Hugging Face on first run.
    """
    lab = labels or _DEFAULT_GLINER_LABELS
    model = _get_gliner(model_id)
    seen: set[str] = set()
    ordered: list[str] = []
    for chunk in _chunk_text(text, max_chunk_chars):
        ents = model.predict_entities(chunk, list(lab), threshold=threshold)
        for e in ents:
            t = (e.get("text") or "").strip()
            if not t:
                continue
            key = t.lower()
            if key not in seen:
                seen.add(key)
                ordered.append(key)
    return ordered


class PrimeKGExplorer:
    """
    Lightweight loader/query helper for the local PrimeKG CSVs in `Agent_3/mkg/`.

    Expected columns (from `mkg/kg.csv`):
    - relation, display_relation
    - x_id, x_type, x_name, x_source (and optionally x_index)
    - y_id, y_type, y_name, y_source (and optionally y_index)
    """

    def __init__(self, filepath: str | None = None):
        print("Loading PrimeKG... (this may take a minute)")
        base_dir = os.path.dirname(__file__)
        default_path = os.path.join(base_dir, "mkg", "kg.csv")
        path = filepath or os.environ.get("PRIMEKG_CSV") or default_path

        # PrimeKG is large. Only load the columns we actually use.
        usecols = [
            "relation",
            "display_relation",
            "x_type",
            "x_name",
            "y_type",
            "y_name",
            "x_id",
            "y_id",
        ]
        self.df = pd.read_csv(path, low_memory=False, usecols=lambda c: c in set(usecols))
        self.filepath = path

    def query_term(self, term: str) -> pd.DataFrame:
        """Seed rows: any edge where ``x_name`` or ``y_name`` contains `term` (case-insensitive)."""
        return self.df[
            (self.df["x_name"].str.contains(term, case=False, na=False))
            | (self.df["y_name"].str.contains(term, case=False, na=False))
        ]

    def get_related_triples(
        self,
        term: str,
        *,
        hops: int = 1,
        max_rows: int = 20_000,
    ) -> pd.DataFrame:
        """
        Collect **all graph-related triples** for a name query by expanding along ``x_id``/``y_id``.

        1. **Seed**: same as :meth:`query_term` (substring match on names).
        2. For each of ``hops`` expansion steps, take the current node set *S* (all endpoints seen
           so far) and add **every** row in the table whose ``x_id`` or ``y_id`` appears in *S*.

        This is a k-round incident-edge closure (not a plain substring filter on 8M rows: it
        follows the KG). Use ``hops=0`` for seed edges only.

        If a hop matches more than ``max_rows`` (minus what is already collected), rows are taken
        in **sorted index order** until the cap is reached; the node set *S* is still updated from
        *all* included expansion rows so the next hop sees a consistent graph.

        Result rows are ordered **seed (name-matched) first**, then expansion, so prompts are not
        dominated by unrelated low-index edges in the CSV.
        """
        seed = self.query_term(term)
        if seed.empty:
            return seed
        if hops < 0:
            raise ValueError("hops must be >= 0")
        if hops == 0:
            return seed.head(max_rows) if len(seed) > max_rows else seed

        # Node set S always reflects the *full* seed (even if we cannot keep every seed row in
        # the result due to max_rows) so the expansion mask is faithful.
        S: set[str] = set()
        S.update(seed["x_id"].astype(str).tolist())
        S.update(seed["y_id"].astype(str).tolist())

        from_seed: list[object] = []
        for i in sorted(seed.index.tolist()):
            if len(from_seed) >= max_rows:
                break
            from_seed.append(i)
        collected: set[object] = set(from_seed)
        if len(collected) >= max_rows:
            out = self.df.loc[from_seed]
            return out if len(out) <= max_rows else out.head(max_rows)

        from_exp: list[object] = []
        xs = self.df["x_id"].astype(str)
        ys = self.df["y_id"].astype(str)
        for _ in range(hops):
            m = xs.isin(S) | ys.isin(S)
            if not m.any():
                break
            hit_idx = set(self.df.index[m].tolist()) - collected
            if not hit_idx:
                break
            n_left = max_rows - len(collected)
            if n_left <= 0:
                break
            to_take = sorted(hit_idx)[:n_left]
            add = self.df.loc[to_take]
            S = S | set(add["x_id"].astype(str)) | set(add["y_id"].astype(str))
            for i in to_take:
                from_exp.append(i)
                collected.add(i)
            if len(collected) >= max_rows:
                break

        # Keep seed (name-matched) rows first so prompts are not dominated by low CSV row ids.
        out = self.df.loc[from_seed + from_exp]
        return out if len(out) <= max_rows else out.head(max_rows)

    def query_disease(
        self, disease_name: str, *, hops: int = 1, max_rows: int = 20_000
    ) -> pd.DataFrame:
        """Related triples for a disease (or any) name; see :meth:`get_related_triples`."""
        return self.get_related_triples(disease_name, hops=hops, max_rows=max_rows)

    def get_treatments(self, disease_name: str) -> pd.DataFrame:
        """
        Return drugs that *treat* a disease.

        In your `mkg/kg.csv`, `relation == "indication"` is typically `drug` -> `disease`.
        """
        mask = (
            (self.df["relation"] == "indication")
            & (self.df["x_type"] == "drug")
            & (self.df["y_type"] == "disease")
            & (self.df["y_name"].str.contains(disease_name, case=False, na=False))
        )
        cols = [c for c in ["x_name", "relation", "y_name", "x_id", "y_id"] if c in self.df.columns]
        return self.df.loc[mask, cols].rename(columns={"x_name": "drug", "y_name": "disease"})

    def format_context(
        self, term: str, *, max_rows: int = 40, hops: int = 1, max_unique_lines: int | None = None
    ) -> str:
        """
        Format a KG context block from **related** triples (graph expansion), not name-only matches.
        """
        cap = max_unique_lines if max_unique_lines is not None else max_rows
        df = self.get_related_triples(term, hops=hops, max_rows=max_rows)
        if df is None or len(df) == 0:
            return "No PrimeKG matches found."

        lines: list[str] = []
        seen: set[tuple[str, str, str]] = set()
        for _, row in df.iterrows():
            x = str(row.get("x_name", "")).strip()
            rel = str(row.get("relation", "")).strip()
            y = str(row.get("y_name", "")).strip()
            if not (x and rel and y):
                continue
            key = (x, rel, y)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {x} — {rel} — {y}")
            if len(lines) >= cap:
                break
        return "\n".join(lines) if lines else "No PrimeKG matches found."


# --- Example ---
if __name__ == "__main__":
    # Uses PRIMEKG_CSV if set, otherwise `Agent_3/mkg/kg.csv`.
    explorer = PrimeKGExplorer()

    disease = "Multiple Sclerosis"
    sample = "The patient has relapsing-remitting multiple sclerosis; continue interferon."
    medical_terms = extract_medical_terms(sample)
    # related_triples = explorer.get_related_triples(medical_terms, hops=1, max_rows=500)
    for term in medical_terms:
        formatted_context = explorer.format_context(term, max_rows=10, hops=1, max_unique_lines=12)
        print(f"Term: {term}")
        print(formatted_context)
        print("-" * 100)

    # print("\n=== Related triples (graph expansion, hops=1, max_rows=500) ===")
    # try:
    #     rel = explorer.get_related_triples(disease, hops=1, max_rows=500)
    #     print(f"row count: {len(rel)}")
    #     print(rel.head(5).to_string(index=False))
    # except Exception as e:
    #     print("get_related_triples failed:", e)

    # print("\n=== Treatments (indication) ===")
    # try:
    #     print(explorer.get_treatments(disease).head(10).to_string(index=False))
    # except Exception as e:
    #     print("Failed to fetch treatments:", e)

    # print("\n=== Context sample (related triples, hops=1) ===")
    # print(explorer.format_context(disease, max_rows=200, hops=1, max_unique_lines=12))

    # print("\n=== Raw query sample (related triples, first 5 rows) ===")
    # print(explorer.query_disease(disease, hops=1, max_rows=500).head(5).to_string(index=False))

    # print("\n=== GLiNER extracted terms (sample) ===")
    # sample = "The patient has relapsing-remitting multiple sclerosis; continue interferon."
    # try:
    #     print(extract_medical_terms(sample)[:15])
    # except Exception as e:
    #     print("GLiNER extraction failed (model download or import issue):", e)
