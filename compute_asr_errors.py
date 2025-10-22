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
    return dist / max(1, len(ref)), ref, hyp

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
            mismatches += 1
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
def read_id_text_pairs(path: str) -> Dict[str, str]:
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
            if ".csv" in path:
                reader = csv.DictReader(f)
                for row in reader:
                    uttid, ref_rext = row.split()
                    refs[uttid] = ref_rext
            elif (".text" in path) or (".txt" in path):
                for line in f:
                    line = line.strip()  # remove newline & spaces
                    if not line:  # skip empty lines
                        continue
                    id, text = line.split(" ", 1)  # split only on the first space
                    if ".wav" in id:
                        id = id.replace(".wav", "")
                    refs[id] = text
    return refs

def get_mini_pairs_ref(refs: Dict[str, str]) -> Dict[str, str]:
    mini_pairs = {}
    for i in range(int(len(refs)/2)):
        ref_a = refs[f"{(i+1):03d}a"]
        ref_b = refs[f"{(i+1):03d}b"]
        ref_a_pinyin = to_pinyin_syllables(ref_a)
        ref_b_pinyin = to_pinyin_syllables(ref_b)
        mini_pairs[f"{(i+1):03d}"] = [i for i, (a, b) in enumerate(zip(ref_a_pinyin, ref_b_pinyin)) if a != b]
    return mini_pairs

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

def log_metrics(uttid, ref, hyp, metrics, log_path):
    """
    Appends a row to the debug metrics log file.
    Columns: ref, hyp, wer, cer, pinyin, tone
    """
    file_exists = os.path.isfile(log_path)
    with open(log_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["ID", "ref", "hyp", "wer", "cer", "pinyin", "tone", "ref_pinyin", "hyp_pinyin"])
        writer.writerow([
            uttid,
            ref, 
            hyp, 
            metrics["wer"], 
            metrics["cer"], 
            metrics["pinyin"], 
            metrics["tone"], 
            " ".join(metrics.get("ref_pinyin", "")), 
            " ".join(metrics.get("hyp_pinyin", ""))
            ]
        )
        
def compute_metrics(uttid, ref_words, hyp_words, debug_log_path, add_mapping=False) -> Dict[str, float]:
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
    pinyin_rate, ref_pinyin, hyp_pinyin = pinyin_error_rate("".join(ref_words), "".join(hyp_words))
    tone_rate = tone_error_rate("".join(ref_words), "".join(hyp_words))
    result = {
        "wer": round(wer_val, 3),
        "cer": round(cer_rate, 3),
        "pinyin": round(pinyin_rate, 3),
        "tone": round(tone_rate, 3),
    }
    log_result = result.copy()
    log_result["ref_pinyin"] = ref_pinyin
    log_result["hyp_pinyin"] = hyp_pinyin
    # Log for debugging
    log_metrics(uttid, "".join(ref_words), "".join(hyp_words), log_result, debug_log_path)
    # Add tone mapping.
    if not add_mapping:
        return result
    # Some pairs may have different lengths after pinyin conversion; align first.
    _, ref_hyp_align = _levenshtein_alignment(ref_pinyin, hyp_pinyin)
    def pair_is_not_none(syl_pair):
        return syl_pair[0] is not None and syl_pair[1] is not None
    ref_tone = [int(split_base_and_tone(syl[0])[1]) for syl in ref_hyp_align if pair_is_not_none(syl)]
    hyp_tone = [int(split_base_and_tone(syl[1])[1]) for syl in ref_hyp_align if pair_is_not_none(syl)]
    result["tone_mapping"] = np.vstack((np.array(ref_tone), np.array(hyp_tone)))
    return result


# -------------------------
# Metrics per (model, a/b)
# -------------------------
def compute_model_output_metrics(ref_path: str, hyp_path: str, debug_log_path: str) -> Dict[str, float]:
    """
    Returns:
      {
        'a': {'wer': x, 'cer': y, 'pinyin': z, 'tone': t},
        'b': {...}
      }
    Missing transcripts are treated as empty string (max error).
    """
    metrics = {}
    refs = read_id_text_pairs(ref_path)
    hyps = read_id_text_pairs(hyp_path)
    for uttid, hyp_text in hyps.items():
        # WER: for Chinese, use character tokens as "words"
        if uttid not in refs:
            raise ValueError(f"Utterance id {uttid} not found in references.")
        ref_text = refs[uttid]
        ref_words = char_tokens(ref_text)
        hyp_words = char_tokens(hyp_text)
        result = compute_metrics(uttid, ref_words, hyp_words, debug_log_path=debug_log_path, add_mapping=True)
        for key in result:
            if isinstance(result[key], float):
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(result[key])
            elif key == "tone_mapping":
                if "tone_mapping" not in metrics:
                    metrics["tone_mapping"] = np.empty((2,0), dtype=int)
                metrics["tone_mapping"] = np.hstack((metrics["tone_mapping"], result["tone_mapping"]))

    return metrics

def get_metrics_mean_values(metrics: Dict[str, float]):
    means = {}
    for metric_name, metric in metrics.items():
        if isinstance(metric, list) and len(metric) > 0 and isinstance(metric[0], float):
            means[metric_name] = np.mean(metric)
    return means

# -------------------------
# Writers
# -------------------------
def write_metric_csv(
    out_dir: str,
    out_file_name: str,
    metric_names: str,
    metrics: Dict[str, float],
    metric_means: Dict[str, float],
):
    """
    Writes CSV with columns:
      item, whisper_a, whisper_b, firered_a, firered_b
    """
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    file_path = os.path.join(out_dir, f"{out_file_name}.csv")
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item", "value"])
        for metric_name in metric_names:
            mean_value = metric_means.get(metric_name, 0.0)
            writer.writerow([metric_name, f"{mean_value:.4f}"])
    print(f"✅ Wrote metrics to {file_path}")

def compute_and_write_confusion_matrix(metrics: dict, out_path: str):
    """
    Computes confusion matrix from tone_mapping and writes to CSV.
    """
    tone_mapping = np.empty((2,0), dtype=int)
    if "tone_mapping" not in metrics:
        raise ValueError("No tone_mapping found in metrics.")
    tone_mapping = metrics["tone_mapping"]
    ref_tones = tone_mapping[0]
    hyp_tones = tone_mapping[1]
    max_tone = max(np.max(ref_tones), np.max(hyp_tones))
    matrix = np.zeros((max_tone, max_tone))
    for r, h in zip(ref_tones, hyp_tones):
        matrix[r-1, h-1] += 1.0  # tones are 1-based
    # Normalize rows to percentages
    for i in range(max_tone):
        row_sum = np.sum(matrix[i, :])
        if row_sum > 0:
            matrix[i, :] = (matrix[i, :] / row_sum)
    # Write to CSV
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["Ref\\Hyp"] + [str(i) for i in range(1, max_tone+1)]
        writer.writerow(header)
        for i in range(max_tone):
            row = [str(i+1)] + [f"{v*100:.2f}" for v in matrix[i].tolist()]
            writer.writerow(row)
    print(f"✅ Wrote confusion matrix to {out_path}")
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

    output_dir = args.out_dir
    os.makedirs(output_dir, exist_ok=True)
    whisper_metrics = compute_model_output_metrics(args.ref, args.whisper_dir, os.path.join(output_dir, "debug_metrics_log_whisper.csv"))
    firered_metrics = compute_model_output_metrics(args.ref, args.firered_dir, os.path.join(output_dir, "debug_metrics_log_firered.csv"))

    whisper_means = get_metrics_mean_values(whisper_metrics)
    firered_means = get_metrics_mean_values(firered_metrics)

    metrics_names = list(whisper_means.keys())
    write_metric_csv(args.out_dir, "output_whisper", metrics_names, whisper_metrics, whisper_means)    
    write_metric_csv(args.out_dir, "output_fireredasr", metrics_names, firered_metrics, firered_means)

    # Confusion matrices
    compute_and_write_confusion_matrix(whisper_metrics, os.path.join(args.out_dir, "whisper_tone_confusion_matrix.csv"))
    compute_and_write_confusion_matrix(firered_metrics, os.path.join(args.out_dir, "firered_tone_confusion_matrix.csv"))

if __name__ == "__main__":
    main()

