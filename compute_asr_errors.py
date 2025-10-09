#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import re
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple
import numpy as np

from jiwer import wer
from pypinyin import pinyin, Style
import pkuseg

# --- Chinese word segmentation for WER ---
_SEG = pkuseg.pkuseg(model_name="default")

def segment_zh(text: str):
    """Segment Mandarin text for WER using pkuseg, after punctuation cleanup."""
    t = normalize(text)
    # Remove spaces so pkuseg sees contiguous Chinese; it handles Latin as well.
    t = t.replace(" ", "")
    if not t:
        return []
    try:
        return _SEG.cut(t)
    except Exception:
        # Fallback: per-char tokens if the segmenter fails for any reason
        return list(t)

# -------------------------
# Normalization utilities
# -------------------------
_PUNCT_RE = re.compile(r"[\s\.\,\!\?\:\;\-—_\(\)\[\]\{\}“”\"'`·、，。！？：；…《》〈〉／/]+")
def normalize(text: str) -> str:
    if text is None:
        return ""
    # Lowercase for safety (does nothing for CJK), strip punctuation/whitespace clusters
    t = text.strip().lower()
    t = _PUNCT_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def char_tokens(text: str) -> List[str]:
    # For Chinese, treat each non-space character as a token (makes WER more informative)
    return [ch for ch in normalize(text).replace(" ", "")]

# -------------------------
# Edit distance + alignment
# -------------------------
def _levenshtein_alignment(seq_a: List[str], seq_b: List[str]) -> Tuple[int, List[Tuple[str, str]]]:
    """
    Returns (distance, alignment) where alignment is a list of (a,b) with None as gap.
    """
    n, m = len(seq_a), len(seq_b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    back = [[None]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        dp[i][0] = i
        back[i][0] = "U"  # up -> deletion from a
    for j in range(1, m+1):
        dp[0][j] = j
        back[0][j] = "L"  # left -> insertion to a

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost_sub = dp[i-1][j-1] + (0 if seq_a[i-1] == seq_b[j-1] else 1)
            cost_del = dp[i-1][j] + 1
            cost_ins = dp[i][j-1] + 1
            best = min(cost_sub, cost_del, cost_ins)
            dp[i][j] = best
            if best == cost_sub:
                back[i][j] = "D"  # diag
            elif best == cost_del:
                back[i][j] = "U"  # up
            else:
                back[i][j] = "L"  # left

    # backtrack
    i, j = n, m
    alignment = []
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "D":
            alignment.append((seq_a[i-1], seq_b[j-1]))
            i -= 1; j -= 1
        elif move == "U":
            alignment.append((seq_a[i-1], None))
            i -= 1
        elif move == "L":
            alignment.append((None, seq_b[j-1]))
            j -= 1
        else:
            break
    alignment.reverse()
    return dp[n][m], alignment

def edit_distance(seq_a: List[str], seq_b: List[str]) -> int:
    d, _ = _levenshtein_alignment(seq_a, seq_b)
    return d

# -------------------------
# Pinyin + tone handling
# -------------------------
_DIGIT_RE = re.compile(r"\d")

def to_pinyin_syllables(s: str) -> List[str]:
    """
    Convert Chinese text to list of Pinyin with tone numbers (TONE3).
    Non-Chinese letters/numbers are kept as single tokens (after normalization).
    """
    t = normalize(s)
    # Separate CJK vs non-CJK: for simplicity, feed whole string to pypinyin; it will
    # pass-through latin tokens when style=NORMAL; we use TONE3 to keep digits.
    # pinyin(...) returns list of list; flatten.
    syls = []
    for syl in pinyin(t.replace(" ", ""), style=Style.TONE3, strict=False, neutral_tone_with_five=True):
        if syl and syl[0]:
            syls.append(syl[0])
    return syls

def split_base_and_tone(syl: str) -> Tuple[str, str]:
    # base without digits; tone digit if present, else "5" (neutral)
    tone_digits = "".join(_DIGIT_RE.findall(syl))
    tone = tone_digits[-1] if tone_digits else "5"
    base = _DIGIT_RE.sub("", syl)
    return base, tone

def pinyin_error_rate(ref_text: str, hyp_text: str) -> float:
    ref = to_pinyin_syllables(ref_text)
    hyp = to_pinyin_syllables(hyp_text)
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    dist = edit_distance(ref, hyp)
    return dist / max(1, len(ref))

def tone_error_rate(ref_text: str, hyp_text: str) -> float:
    ref = to_pinyin_syllables(ref_text)
    hyp = to_pinyin_syllables(hyp_text)
    if not ref:
        return 0.0
    _, align = _levenshtein_alignment(ref, hyp)
    mismatches = 0
    count_ref_positions = 0
    for r, h in align:
        if r is None:
            # insertion into hyp; counts against ref length implicitly
            continue
        count_ref_positions += 1
        if h is None:
            # deletion -> consider as tone error (no matched syllable)
            mismatches += 1
            continue
        r_base, r_tone = split_base_and_tone(r)
        h_base, h_tone = split_base_and_tone(h)
        if r_base == h_base:
            if r_tone != h_tone:
                mismatches += 1
        else:
            # wrong syllable -> counts as tone error as well (strict)
            mismatches += 1
    return mismatches / max(1, count_ref_positions)

# -------------------------
# I/O helpers
# -------------------------
def load_references(path: str) -> Dict[str, str]:
    """
    Returns dict mapping id like '001a' -> text.
    If CSV: expects header with columns id,text.
    If directory: reads *.txt files by filename stem.
    """
    refs = {}
    if os.path.isdir(path):
        for fn in os.listdir(path):
            if fn.endswith(".txt"):
                key = os.path.splitext(fn)[0]
                with open(os.path.join(path, fn), "r", encoding="utf-8") as f:
                    refs[key] = f.read().strip()
    else:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                k = row["pair_num"].strip() + row["label"].strip()
                refs[k] = row["text"].strip()
    return refs

def read_txt_or_none(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None

def collect_ids(refs: Dict[str, str]) -> List[str]:
    """
    Returns sorted base ids like ['001','002',...] inferred from ref keys ending with 'a' or 'b'.
    """
    base_ids = set()
    for k in refs.keys():
        if len(k) >= 2 and k[-1] in ("a", "b"):
            base_ids.add(k[:-1])
    return sorted(base_ids)

def compute_metrics(ref_words, hyp_words) -> Dict[str, float]:
    """
    Returns dict with keys: wer, cer, pinyin, tone
    """
    if len(ref_words) == 0:
        wer_val = 0.0 if len(hyp_words) == 0 else 1.0
    else:
        wer_val = wer(" ".join(ref_words), " ".join(hyp_words))
    cer_val = edit_distance(list("".join(ref_words)), list("".join(hyp_words)))
    cer_den = max(1, len(ref_words))
    cer_rate = cer_val / cer_den
    pinyin_rate = pinyin_error_rate("".join(ref_words), "".join(hyp_words))
    tone_rate = tone_error_rate("".join(ref_words), "".join(hyp_words))
    return {
        "wer": round(wer_val, 2),
        "cer": round(cer_rate, 2),
        "pinyin": round(pinyin_rate, 2),
        "tone": round(tone_rate, 2),
    }


# -------------------------
# Metrics per (model, a/b)
# -------------------------
def compute_metrics_for_item(
    item_base: str,
    refs: Dict[str, str],
    model_dir: str,
) -> Dict[str, Dict[str, float]]:
    """
    Returns:
      {
        'a': {'wer': x, 'cer': y, 'pinyin': z, 'tone': t},
        'b': {...}
      }
    Missing transcripts are treated as empty string (max error).
    """
    out = {}
    for ab in ("a", "b"):
        key = f"{item_base}{ab}"
        ref = refs.get(key, "")
        hyp = read_txt_or_none(os.path.join(model_dir, f"{key}.txt")) or ""
        # WER: for Chinese, use character tokens as "words"
        ref_words = char_tokens(ref)
        metrics = {"wer": [], "cer": [], "pinyin": [], "tone": []}
        hyps = hyp.split("\n")
        for hyp in hyps:
            hyp_words = char_tokens(hyp)
            for key in metrics:
                result = compute_metrics(ref_words, hyp_words)
                metrics[key].append(result[key])
        
        out[ab] = {key: round(np.mean(metric),2) for key, metric in metrics.items()}
    return out

# -------------------------
# Writers
# -------------------------
def write_metric_csv(
    out_path: str,
    items: List[str],
    whisper_metrics: Dict[str, Dict[str, float]],
    firered_metrics: Dict[str, Dict[str, float]],
    metric_name: str,
):
    """
    Writes CSV with columns:
      item, whisper_a, whisper_b, firered_a, firered_b
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item", "whisper_a", "whisper_b", "firered_a", "firered_b"])
        for item in items:
            w = whisper_metrics[item]
            r = firered_metrics[item]
            row = [
                item,
                w["a"][metric_name],
                w["b"][metric_name],
                r["a"][metric_name],
                r["b"][metric_name],
            ]
            writer.writerow(row)

# TODO: add tone into the table
# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Compute WER, CER, Pinyin and Tone errors from two ASR outputs.")
    ap.add_argument("--ref", required=True, help="Path to references: CSV (id,text) or directory with *.txt")
    ap.add_argument("--whisper_dir", required=True, help="Directory containing whisper transcripts (001a.txt etc.)")
    ap.add_argument("--firered_dir", required=True, help="Directory containing FireRed transcripts (001a.txt etc.)")
    ap.add_argument("--out_dir", required=True, help="Where to write CSVs")
    args = ap.parse_args()

    refs = load_references(args.ref)
    items = collect_ids(refs)
    if not items:
        raise SystemExit("No items found in references. Expect ids like 001a / 001b.")

    whisper_metrics = {}
    firered_metrics = {}

    for item in items:
        whisper_metrics[item] = compute_metrics_for_item(item, refs, args.whisper_dir)
        firered_metrics[item] = compute_metrics_for_item(item, refs, args.firered_dir)

    write_metric_csv(os.path.join(args.out_dir, "wer.csv"), items, whisper_metrics, firered_metrics, "wer")
    write_metric_csv(os.path.join(args.out_dir, "cer.csv"), items, whisper_metrics, firered_metrics, "cer")
    write_metric_csv(os.path.join(args.out_dir, "pinyin_error.csv"), items, whisper_metrics, firered_metrics, "pinyin")
    write_metric_csv(os.path.join(args.out_dir, "tone_error.csv"), items, whisper_metrics, firered_metrics, "tone")

    print(f"Done. Wrote: {args.out_dir}/wer.csv, cer.csv, pinyin_error.csv, tone_error.csv")

if __name__ == "__main__":
    main()
