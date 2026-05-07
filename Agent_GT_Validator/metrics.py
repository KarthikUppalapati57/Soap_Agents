from __future__ import annotations

import re
from typing import Iterable, Mapping

from .schemas import RecallScore, SoapSection, SectionAlignment, StructuralAlignmentScore


def _norm_entity(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def entity_recall(gt: Iterable[str], gen: Iterable[str]) -> RecallScore:
    gt_set = {_norm_entity(x) for x in gt if _norm_entity(x)}
    gen_set = {_norm_entity(x) for x in gen if _norm_entity(x)}
    overlap = sorted(gt_set & gen_set)
    gt_only = sorted(gt_set - gen_set)
    gen_only = sorted(gen_set - gt_set)
    recall = (len(overlap) / len(gt_set)) if gt_set else 0.0
    return RecallScore(
        gt_count=len(gt_set),
        gen_count=len(gen_set),
        overlap_count=len(overlap),
        recall=float(recall),
        gt_only=gt_only,
        gen_only=gen_only,
        overlap=overlap,
    )


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def token_f1(a: str, b: str) -> float:
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    sa = set(ta)
    sb = set(tb)
    inter = len(sa & sb)
    if inter == 0:
        return 0.0
    p = inter / len(sb)
    r = inter / len(sa)
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def _lcs_len(a: list[str], b: list[str]) -> int:
    # DP LCS, O(n*m) but sections are small.
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]


def rouge_l_f1(a: str, b: str) -> float:
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    lcs = _lcs_len(ta, tb)
    if lcs == 0:
        return 0.0
    p = lcs / len(tb)
    r = lcs / len(ta)
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def structural_alignment(
    gt_sections: Mapping[SoapSection, str],
    gen_sections: Mapping[SoapSection, str],
) -> StructuralAlignmentScore:
    by: list[SectionAlignment] = []
    sec_order: tuple[SoapSection, ...] = ("Subjective", "Objective", "Assessment", "Plan", "Other")
    for sec in sec_order:
        gt = (gt_sections.get(sec) or "").strip()
        gen = (gen_sections.get(sec) or "").strip()
        gt_present = bool(gt)
        gen_present = bool(gen)
        if gt_present or gen_present:
            tf1 = token_f1(gt, gen)
            rl = rouge_l_f1(gt, gen)
        else:
            tf1 = 1.0
            rl = 1.0
        by.append(
            SectionAlignment(
                section=sec,
                gt_present=gt_present,
                gen_present=gen_present,
                token_f1=tf1,
                rouge_l_f1=rl,
            )
        )

    present_pairs = [
        (gt_sections.get(s) or "", gen_sections.get(s) or "")
        for s in sec_order
        if (gt_sections.get(s) or "").strip() or (gen_sections.get(s) or "").strip()
    ]
    if present_pairs:
        overall_tf1 = sum(token_f1(a, b) for a, b in present_pairs) / len(present_pairs)
        overall_rl = sum(rouge_l_f1(a, b) for a, b in present_pairs) / len(present_pairs)
    else:
        overall_tf1 = 1.0
        overall_rl = 1.0

    assessment_score = next((x.token_f1 for x in by if x.section == "Assessment"), None)
    plan_score = next((x.token_f1 for x in by if x.section == "Plan"), None)
    return StructuralAlignmentScore(
        overall_token_f1=float(overall_tf1),
        overall_rouge_l_f1=float(overall_rl),
        by_section=by,
        assessment_alignment_score=assessment_score,
        plan_alignment_score=plan_score,
    )

