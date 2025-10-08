# -*- coding: utf-8 -*-
# pip install pandas soundfile praatio regex
import os, re, math, csv
from pathlib import Path
import pandas as pd
import soundfile as sf
from praatio import tgio

# ---------- CONFIG (EDIT THESE) ----------
REF_FILE  = "/Users/lixiyang/Desktop/mandi_mini_pairs/mandi_minipair_data/mandarin_sst.txt"
AUDIO_DIR = "/Users/lixiyang/Desktop/mandi_mini_pairs/mandi_minipair_data/long_audio"            # contains XIA_022_M_CMN_SST.wav, ...
TG_DIR    = "/Users/lixiyang/Desktop/mandi_mini_pairs/mandi_minipair_data/textgrid"            # contains XIA_022_M_CMN_SST.TextGrid, ...
OUT_DIR   = "/Users/lixiyang/Desktop/mandi_mini_pairs/mandi_minipair_data/segments"
TIER_NAME_PREFERRED = None                 # e.g., "sentence"; or None to take first IntervalTier
SKIP_LABELS = {"sil", "sp", "pause"}       # labels to ignore (like in your screenshot)
MIN_DUR = 0.15                             # seconds; skip shorter intervals
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# keep only Chinese + basic CJK punct; drop pinyin/latin and whitespace
_non_cjk = re.compile(r"[^\u4e00-\u9fff，。！？、“”‘’（）《》——…·：；、\s]")
_ws = re.compile(r"\s+")

def norm_chi(s: str) -> str:
    return _ws.sub("", _non_cjk.sub("", s or "")).strip()

def build_lookup(path: str):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = norm_chi(line.strip())
            if t: lines.append(t)
    lookup = {}      # text -> (num, 'a'/'b')
    id2text = {}     # "001a" -> text
    i, pair = 0, 1
    while i < len(lines):
        a = lines[i]; lookup[a] = (f"{pair:03d}", "a"); id2text[f"{pair:03d}a"] = a; i += 1
        if i < len(lines):
            b = lines[i]; lookup[b] = (f"{pair:03d}", "b"); id2text[f"{pair:03d}b"] = b; i += 1
        else:
            print(f"[WARN] Odd count: pair {pair:03d} has only 'a'.")
        pair += 1

    # Save lookup table to CSV
    lookup_csv_path = Path(OUT_DIR, "lookup_table.csv")
    with open(lookup_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_num", "label", "text"])
        for seg_id, text in id2text.items():
            pair_num = seg_id[:-1]
            label = seg_id[-1]
            writer.writerow([pair_num, label, text])
    print(f"✅ Loaded {len(id2text)//2} pairs ({len(id2text)} lines) from {path}\nLookup table: {lookup_csv_path}")
    return lookup, id2text

def pick_tier(tg: tgio.Textgrid):
    if TIER_NAME_PREFERRED and TIER_NAME_PREFERRED in tg.tierNameList:
        t = tg.tierDict[TIER_NAME_PREFERRED]
        if t.tierType == tgio.INTERVAL_TIER: return t
    # fallback: first interval tier
    for name in tg.tierNameList:
        t = tg.tierDict[name]
        if t.tierType == tgio.INTERVAL_TIER: return t
    raise RuntimeError("No IntervalTier found.")

def cut_wav(wav_path: str, start: float, end: float, out_path: str) -> float:
    audio, sr = sf.read(wav_path)
    s = max(0, int(math.floor(start * sr)))
    e = min(len(audio), int(math.ceil(end * sr)))
    if e <= s: return 0.0
    dur = (e - s)/sr
    if dur < MIN_DUR: return 0.0
    sf.write(out_path, audio[s:e], sr)
    return dur

def main():
    lookup, _ = build_lookup(REF_FILE)
    rows = []

    for wav in sorted(Path(AUDIO_DIR).glob("*.wav")):
        base = wav.stem
        tg_path = Path(TG_DIR, base + ".TextGrid")
        if not tg_path.exists():
            print(f"[SKIP] Missing TextGrid for {base}")
            continue

        tg = tgio.openTextgrid(str(tg_path))
        tier = pick_tier(tg)
        spk_dir = Path(OUT_DIR, base); spk_dir.mkdir(parents=True, exist_ok=True)

        for itv in tier.entryList:
            start, end, lab = itv.start, itv.end, (itv.label or "").strip()
            if not lab or lab in SKIP_LABELS: continue
            text = norm_chi(lab)
            if not text or text in SKIP_LABELS: continue
            if text not in lookup:              # not one of your minimal-pair sentences
                continue

            pair_num, ab = lookup[text]
            seg_id = f"{pair_num}{ab}"
            out_wav = spk_dir / f"{seg_id}.wav"
            out_txt = spk_dir / f"{seg_id}.txt"

            dur = cut_wav(str(wav), start, end, str(out_wav))
            if dur == 0.0: continue

            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(text + "\n")

            rows.append({
                "speaker": base, "pair_num": pair_num, "label": ab, "pair_id": seg_id,
                "start": round(start, 3), "end": round(end, 3), "duration": round(dur, 3),
                "text": text
            })
            print(f"✓ {base} {seg_id} [{start:.2f}-{end:.2f}s]")

    if rows:
        df = pd.DataFrame(rows).sort_values(["speaker", "pair_num", "label"])
        csv_path = Path(OUT_DIR, "segments.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
        print(f"\n✅ Wrote {len(df)} segments\nCSV: {csv_path}")
    else:
        print("⚠️ No matches saved. Double-check tier name, SKIP_LABELS, or reference text.")

if __name__ == "__main__":
    main()
