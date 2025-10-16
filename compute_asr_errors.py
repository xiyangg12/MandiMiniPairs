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

def log_metrics(ref, hyp, metrics, pair_id, log_path="debug_metrics_log.csv"):
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
            pair_id,
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
        
def compute_metrics(ref_words, hyp_words, pair_id, file_name) -> Dict[str, float]:
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
    log_metrics("".join(ref_words), "".join(hyp_words), log_result, pair_id, file_name)
    return result


# -------------------------
# Metrics per (model, a/b)
# -------------------------
def compute_metrics_for_item(
    pair_id: str,
    refs: Dict[str, str],
    model_dir: str,
    mini_pairs: Dict[str, List[int]],
    debug_log_file_name="debug_metrics_log.csv"
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
    pair_char_index = mini_pairs[pair_id]
    for ab in ("a", "b"):
        pair_id_ab = f"{pair_id}{ab}"
        ref = refs.get(pair_id_ab, "")
        hyp = read_txt_or_none(os.path.join(model_dir, f"{pair_id_ab}.txt")) or ""
        # WER: for Chinese, use character tokens as "words"
        ref_words = char_tokens(ref)
        metrics = {}
        hyps = hyp.split("\n")
        for hyp in hyps:
            hyp_words = char_tokens(hyp)
            result = compute_metrics(ref_words, hyp_words, pair_id=pair_id_ab, file_name=debug_log_file_name)
            # Dedicated metrics for mini pairs
            _, alignment = _levenshtein_alignment(ref_words, hyp_words)
            ref_words_aligned = [r for r, h in alignment]
            hyp_words_aligned = [h for r, h in alignment]
            hyp_words_mini_pairs = [hyp_words_aligned[index] for index in pair_char_index]
            ref_words_mini_pairs = [ref_words_aligned[index] for index in pair_char_index]
            if None in ref_words_mini_pairs or None in hyp_words_mini_pairs:
                result_mini_pairs = {k : 1.0 for k in result_mini_pairs}
            else:
                result_mini_pairs = compute_metrics(ref_words_mini_pairs, hyp_words_mini_pairs, pair_id=pair_id_ab, file_name=debug_log_file_name)
                result_mini_pairs = {f"mini_{k}": v for k, v in result_mini_pairs.items()}
            result.update(result_mini_pairs)
            for key in result:
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(result[key])
        
        out[ab] = {key: round(np.mean(metric),3) for key, metric in metrics.items()}
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
    extra_metric_name: str,
    whisper_means: Dict[str, Dict[str, float]],
    firered_means: Dict[str, Dict[str, float]],
):
    """
    Writes CSV with columns:
      item, whisper_a, whisper_b, firered_a, firered_b
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        first_row = ["item", "whisper_a(%)", "whisper_b(%)", "firered_a(%)", "firered_b(%)"]
        if extra_metric_name:
            first_row += [f"whisper_a_{extra_metric_name}(%)", f"whisper_b_{extra_metric_name}(%)", f"firered_a_{extra_metric_name}(%)", f"firered_b_{extra_metric_name}(%)"]
        writer.writerow(first_row)
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
            if extra_metric_name:
                row += [
                    w["a"][extra_metric_name],
                    w["b"][extra_metric_name],
                    r["a"][extra_metric_name],
                    r["b"][extra_metric_name],
                ]
            row = [f"{v*100:.1f}" if isinstance(v, float) else v for v in row]
            writer.writerow(row)
        writer.writerow([""])
        writer.writerow([
            "Mean",
            f"{whisper_means[metric_name]['a']*100:.1f}",
            f"{whisper_means[metric_name]['b']*100:.1f}",
            f"{firered_means[metric_name]['a']*100:.1f}",
            f"{firered_means[metric_name]['b']*100:.1f}",
        ] + ([
            f"{whisper_means[extra_metric_name]['a']*100:.1f}",
            f"{whisper_means[extra_metric_name]['b']*100:.1f}",
            f"{firered_means[extra_metric_name]['a']*100:.1f}",
            f"{firered_means[extra_metric_name]['b']*100:.1f}",
        ] if extra_metric_name else []))
        a = 1

def get_metrics_mean_values(
    metrics_names: List[str],
    whisper_metrics: Dict[str, Dict[str, float]],
    firered_metrics: Dict[str, Dict[str, float]]
):
    whisper_means = {}
    firered_means = {}
    for metric in metrics_names:
        whisper_a_vals = [whisper_metrics[item]["a"].get(metric, 0.0) for item in whisper_metrics]
        whisper_b_vals = [whisper_metrics[item]["b"].get(metric, 0.0) for item in whisper_metrics]
        firered_a_vals = [firered_metrics[item]["a"].get(metric, 0.0) for item in firered_metrics]
        firered_b_vals = [firered_metrics[item]["b"].get(metric, 0.0) for item in firered_metrics]
        whisper_means[metric] = {
            "a": round(np.mean(whisper_a_vals), 3),
            "b": round(np.mean(whisper_b_vals), 3)
        }
        firered_means[metric] = {
            "a": round(np.mean(firered_a_vals), 3),
            "b": round(np.mean(firered_b_vals), 3)
        }
    return whisper_means, firered_means
    
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
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    os.remove("debug_metrics_log_whisper.csv") if os.path.exists("debug_metrics_log_whisper.csv") else None
    os.remove("debug_metrics_log_firered.csv") if os.path.exists("debug_metrics_log_firered.csv") else None

    refs = load_references(args.ref)
    items = collect_ids(refs)
    mini_pairs = get_mini_pairs_ref(refs)
    if not items:
        raise SystemExit("No items found in references. Expect ids like 001a / 001b.")

    whisper_metrics = {}
    firered_metrics = {}

    for item in items:
        whisper_metrics[item] = compute_metrics_for_item(item, refs, args.whisper_dir, mini_pairs, debug_log_file_name="debug_metrics_log_whisper.csv")
        firered_metrics[item] = compute_metrics_for_item(item, refs, args.firered_dir, mini_pairs, debug_log_file_name="debug_metrics_log_firered.csv")

    metrics_names = ["wer", "cer", "pinyin", "tone", "mini_wer", "mini_cer", "mini_pinyin", "mini_tone"]
    whisper_means, firered_means = get_metrics_mean_values(metrics_names, whisper_metrics, firered_metrics)

    write_metric_csv(os.path.join(args.out_dir, "wer.csv"), items, whisper_metrics, firered_metrics, "wer", "mini_wer", whisper_means, firered_means)
    write_metric_csv(os.path.join(args.out_dir, "cer.csv"), items, whisper_metrics, firered_metrics, "cer", "mini_cer", whisper_means, firered_means)
    write_metric_csv(os.path.join(args.out_dir, "pinyin_error.csv"), items, whisper_metrics, firered_metrics, "pinyin", "mini_pinyin", whisper_means, firered_means)
    write_metric_csv(os.path.join(args.out_dir, "tone_error.csv"), items, whisper_metrics, firered_metrics, "tone", "mini_tone", whisper_means, firered_means)

    print(f"Done. Wrote: {args.out_dir}/wer.csv, cer.csv, pinyin_error.csv, tone_error.csv")

if __name__ == "__main__":
    main()

