# -*- coding: utf-8 -*-
# pip install soundfile praatio regex
import os, re, math, shutil
from pathlib import Path
import soundfile as sf
from praatio import tgio

# ---------- CONFIG (EDIT THESE) ----------
AUDIO_DIR = "/Users/lixiyang/Desktop/mandi_mini_pairs/mandi_minipair_data/raw_data/mandarin"
TG_DIR    = "/Users/lixiyang/Desktop/mandi_mini_pairs/mandi_minipair_data/raw_data/mandarin_textgrid"
OUT_DIR   = "/Users/lixiyang/Desktop/mandi_mini_pairs/mandi_minipair_data/seg_pair25"
EXPECTED_SPEAKERS = 25    # for audit only

REF_FILE  = "/Users/lixiyang/Desktop/mandi_mini_pairs/mandi_minipair_data/mandarin_sst.txt"
TIER_NAME_PREFERRED = None
SKIP_LABELS = {"sil", "sp", "pause"}
MIN_DUR = 0.15            # seconds; skip very short
SILENCE_PADDING = 0.5     # seconds on each side

# ----------------------------------------

_non_cjk = re.compile(r"[^\u4e00-\u9fff，。！？、“”‘’（）《》——…·：；、\s]")
_ws = re.compile(r"\s+")

def norm_chi(s: str) -> str:
    return _ws.sub("", _non_cjk.sub("", s or "")).strip()

def build_lookup(path: str):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = norm_chi(line.strip())
            if t:
                lines.append(t)
    lookup = {}
    i, pair = 0, 1
    while i < len(lines):
        a = lines[i]; lookup[a] = (f"{pair:03d}", "a"); i += 1
        if i < len(lines):
            b = lines[i]; lookup[b] = (f"{pair:03d}", "b"); i += 1
        else:
            print(f"[WARN] Odd count: pair {pair:03d} has only 'a'.")
        pair += 1
    return lookup

def pick_tier(tg: tgio.Textgrid):
    if TIER_NAME_PREFERRED and TIER_NAME_PREFERRED in tg.tierNameList:
        t = tg.tierDict[TIER_NAME_PREFERRED]
        if t.tierType == tgio.INTERVAL_TIER:
            return t
    for name in tg.tierNameList:
        t = tg.tierDict[name]
        if t.tierType == tgio.INTERVAL_TIER:
            return t
    raise RuntimeError("No IntervalTier found.")

def cut_wav(wav_path: str, start: float, end: float, out_path: str) -> float:
    audio, sr = sf.read(wav_path)
    s = max(0, int(math.floor((start - SILENCE_PADDING) * sr)))
    e = min(len(audio), int(math.ceil((end + SILENCE_PADDING) * sr)))
    if e <= s:
        return 0.0
    dur = (e - s) / sr
    if dur < MIN_DUR:
        return 0.0
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, audio[s:e], sr)
    return dur

def main():
    # lookup = build_lookup(REF_FILE)

    counters = {}        # pair_id -> next index int (1-based)
    cleaned_pair = set() # which pair dirs cleaned already
    per_pair_counts = {} # pair_id -> saved count

    for wav in sorted(Path(AUDIO_DIR).glob("*.wav")):
        base = wav.stem
        tg_path = Path(TG_DIR, base + ".TextGrid")
        if not tg_path.exists():
            print(f"[SKIP] Missing TextGrid for {base}")
            continue

        tg = tgio.openTextgrid(str(tg_path))
        tier = pick_tier(tg)

        for itv in tier.entryList:
            lab = (itv.label or "").strip()
            if not lab or lab in SKIP_LABELS:
                continue
            text = norm_chi(lab)
            if not text or text in SKIP_LABELS or text not in lookup:
                continue

            pair_num, ab = lookup[text]
            pair_id = f"{pair_num}{ab}"
            pair_dir = Path(OUT_DIR, pair_id)

            # clean this pair folder once per run
            if pair_id not in cleaned_pair:
                if pair_dir.exists():
                    shutil.rmtree(pair_dir)
                pair_dir.mkdir(parents=True, exist_ok=True)
                counters[pair_id] = 1
                cleaned_pair.add(pair_id)

            idx = counters[pair_id]
            out_wav = pair_dir / f"{idx:04d}.wav"

            dur = cut_wav(str(wav), itv.start, itv.end, str(out_wav))
            if dur == 0.0:
                continue

            counters[pair_id] = idx + 1
            per_pair_counts[pair_id] = per_pair_counts.get(pair_id, 0) + 1
            print(f"✓ {pair_id}/{idx:04d}.wav  [{itv.start:.2f}-{itv.end:.2f}s]")

    # ---- Audit ----
    if not per_pair_counts:
        print("\n⚠️ No segments saved. Check tier name, labels, or reference text.")
        return

    print("\n=== Audit (files per pair_id) ===")
    for pid in sorted(per_pair_counts.keys()):
        n = per_pair_counts[pid]
        status = "OK" if n == EXPECTED_SPEAKERS else ("MISSING" if n < EXPECTED_SPEAKERS else "EXTRA")
        print(f"{pid}: {n}/{EXPECTED_SPEAKERS}  [{status}]")

if __name__ == "__main__":
    main()
