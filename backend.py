"""Signal processing, analysis, and report generation for the ECG/PPG dashboard."""

from __future__ import annotations

import csv
import io
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy.ndimage import uniform_filter1d
from scipy.stats import skew

SIGNAL_DEFAULTS = {
    "ECG": {"low": 0.5, "high": 40.0, "order": 4, "min_bpm": 30, "max_bpm": 220},
    "PPG": {"low": 0.5, "high": 8.0, "order": 4, "min_bpm": 30, "max_bpm": 200},
}

FILTER_TYPES = {"Band-pass": "bandpass", "High-pass": "highpass", "Low-pass": "lowpass"}

NOTCH_OPTIONS = {"Off": None, "50 Hz": 50.0, "60 Hz": 60.0}

MAX_TEMPLATE_BEATS = 3_000    # beats used for the morphology template / SQI

TIME_TOKENS = {"time", "t", "timestamp", "times", "sec", "secs", "seconds", "ms",
               "millis", "msec", "datetime", "date", "elapsed", "us"}

INDEX_TOKENS = {"sample", "samples", "index", "idx", "n", "no", "count", "unnamed"}

PPG_TOKENS = {"ppg", "pleth", "bvp", "ir", "red", "green", "spo2", "pulse", "ppg1", "ppg2"}

ECG_TOKENS = {"ecg", "ekg", "lead", "mlii", "ii", "v1", "v2", "v5", "avr", "avl", "avf"}

ICONS = {"ok": "✅", "warn": "⚠️", "info": "ℹ️"}

def is_num(value) -> bool:
    return value is not None and np.isfinite(value)

def fmt(value, digits: int = 1, unit: str = "") -> str:
    if not is_num(value):
        return "—"
    return f"{value:,.{digits}f}" + (f" {unit}" if unit else "")

def fmt_duration(seconds: float) -> str:
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _tokens(name) -> set[str]:
    return set(re.split(r"[^a-z0-9]+", str(name).lower())) - {""}

def _looks_numeric(value) -> bool:
    try:
        float(str(value).replace(",", "."))
        return True
    except ValueError:
        return False

def numeric_fraction(col: pd.Series) -> float:
    if pd.api.types.is_numeric_dtype(col):
        return float(col.notna().mean()) if len(col) else 0.0
    return float(pd.to_numeric(col, errors="coerce").notna().mean()) if len(col) else 0.0

def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    """Read a CSV with delimiter / header / decimal-comma sniffing."""
    text = raw[:65536].decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    skip = 0
    while skip < len(lines) and (not lines[skip].strip() or lines[skip].lstrip().startswith("#")):
        skip += 1
    sample = "\n".join(lines[skip:skip + 50])

    try:
        sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    decimal = "," if sep == ";" and re.search(r"\d,\d", sample) else "."

    def _read(header, sep_=sep):
        return pd.read_csv(io.BytesIO(raw), sep=sep_, decimal=decimal, skiprows=skip,
                           header=header, skipinitialspace=True, encoding="utf-8-sig",
                           engine="python" if sep_ == r"\s+" else "c")

    df = _read(0)
    if df.shape[1] == 1 and re.search(r"\d[ \t]+-?\d", sample):   # whitespace separated
        sep = r"\s+"
        df = _read(0, sep)
    if all(_looks_numeric(c) for c in df.columns):                  # no header row
        df = _read(None, sep)
        df.columns = [f"column_{i + 1}" for i in range(df.shape[1])]
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")                                # trailing delimiters
    return df

def guess_columns(df: pd.DataFrame, numeric_cols: list[str]) -> tuple[str | None, str | None]:
    """Best guess for (time column, signal column)."""
    time_col = None
    for c in df.columns:
        tok = _tokens(c)
        if tok & TIME_TOKENS and not tok & (ECG_TOKENS | PPG_TOKENS):
            time_col = c
            break
    if time_col is None and numeric_cols and not _tokens(numeric_cols[0]) & INDEX_TOKENS:
        first = pd.to_numeric(df[numeric_cols[0]], errors="coerce").to_numpy(float)
        d = np.diff(first[np.isfinite(first)])
        if d.size > 10 and np.all(d > 0) and np.std(d) / np.mean(d) < 0.25 and np.mean(d) != 1:
            time_col = numeric_cols[0]

    candidates = [c for c in numeric_cols if c != time_col and not _tokens(c) & INDEX_TOKENS]
    candidates = candidates or [c for c in numeric_cols if c != time_col] or numeric_cols
    sig_col = next((c for c in candidates if _tokens(c) & (ECG_TOKENS | PPG_TOKENS)),
                   candidates[0] if candidates else None)
    return time_col, sig_col

def guess_signal_type(name: str) -> str:
    return "PPG" if _tokens(name) & PPG_TOKENS else "ECG"

def to_seconds(col: pd.Series, unit: str) -> tuple[np.ndarray, str]:
    """Convert a time column to seconds. Returns (seconds, unit used)."""
    num = pd.to_numeric(col, errors="coerce")
    if num.notna().mean() >= 0.9:
        t = num.to_numpy(float)
        if unit == "Auto":
            d = np.diff(np.sort(t[np.isfinite(t)]))
            d = d[d > 0]
            step = float(np.median(d)) if d.size else 1.0
            unit = "s" if step < 0.1 else ("ms" if step < 100 else "µs")
        scale = {"s": 1.0, "ms": 1e-3, "µs": 1e-6}[unit]
        return t * scale, unit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dt = pd.to_datetime(col, errors="coerce")
    if dt.notna().mean() >= 0.9:
        return (dt - dt.min()).dt.total_seconds().to_numpy(float), "date-time"
    raise ValueError(f"Column '{col.name}' is neither numeric time nor date-time text.")

def estimate_fs(t: np.ndarray) -> float:
    t = np.sort(t[np.isfinite(t)])
    d = np.diff(t)
    d = d[d > 0]
    if d.size == 0:
        return float("nan")
    normal = d[d <= 1.5 * np.median(d)]
    return float(1.0 / normal.mean())

def _fill_nans(x: np.ndarray, notes: list[str]) -> np.ndarray:
    bad = ~np.isfinite(x)
    if bad.all():
        raise ValueError("The signal column contains no numeric values.")
    if bad.any():
        idx = np.arange(x.size)
        x = x.copy()
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
        notes.append(f"{bad.sum():,} missing / non-numeric samples ({bad.mean():.1%}) were linearly interpolated.")
    return x

def prepare_signal(t_sec: np.ndarray | None, x: np.ndarray, fs_manual: float | None):
    """Return (t, x, fs, notes) on a uniform time grid starting at 0 s."""
    notes: list[str] = []
    x = np.asarray(x, dtype=float)

    if t_sec is None:
        fs = float(fs_manual)
        x = _fill_nans(x, notes)
        return np.arange(x.size) / fs, x, fs, notes

    t = np.asarray(t_sec, dtype=float)
    ok = np.isfinite(t)
    if (~ok).any():
        notes.append(f"{(~ok).sum():,} rows without a valid timestamp were dropped.")
    t, x = t[ok], x[ok]
    if t.size < 10:
        raise ValueError("Fewer than 10 rows have a valid timestamp.")
    if np.any(np.diff(t) < 0):
        order = np.argsort(t, kind="stable")
        t, x = t[order], x[order]
        notes.append("Rows were not in time order and have been sorted.")
    keep = np.concatenate(([True], np.diff(t) > 0))
    if (~keep).any():
        notes.append(f"{(~keep).sum():,} duplicate timestamps were removed.")
        t, x = t[keep], x[keep]

    fs = estimate_fs(t)
    x = _fill_nans(x, notes)
    d = np.diff(t)
    gaps = d > 1.5 * np.median(d)
    jitter = float(np.std(d[~gaps]) / np.mean(d[~gaps])) if (~gaps).any() else 0.0
    if gaps.any() or jitter > 0.25:
        step = 1.0 / fs
        tu = np.arange(t[0], t[-1] + 0.5 * step, step)
        x = np.interp(tu, t, x)
        notes.append(
            f"Timestamps were irregular ({gaps.sum():,} gap(s), longest {d.max():.3f} s); "
            f"the signal was linearly resampled onto a uniform {fs:.2f} Hz grid. "
            "Beats inside long gaps cannot be recovered."
        )
        t = tu
    else:
        t = t[0] + np.arange(x.size) / fs
    return t - t[0], x, fs, notes

def make_demo_csv(kind: str, seed: int = 7) -> bytes:
    """Synthetic raw ECG / PPG with baseline wander, mains hum and noise."""
    rng = np.random.default_rng(seed)
    fs = 500.0 if kind == "ECG" else 100.0
    duration = 60.0
    t = np.arange(0, duration, 1 / fs)

    beats, tb = [], 0.4
    while tb < duration - 0.6:  # ~72 BPM with respiratory sinus arrhythmia
        beats.append(tb)
        tb += 60 / 72 + 0.05 * np.sin(2 * np.pi * 0.25 * tb) + rng.normal(0, 0.015)

    clean = np.zeros_like(t)
    if kind == "ECG":
        waves = [(-0.20, 0.15, 0.025), (-0.035, -0.15, 0.010), (0.0, 1.20, 0.012),
                 (0.035, -0.30, 0.012), (0.28, 0.35, 0.045)]  # P, Q, R, S, T
        for b in beats:
            for offset, amp, width in waves:
                clean += amp * np.exp(-0.5 * ((t - b - offset) / width) ** 2)
        mains = 0.08 * np.sin(2 * np.pi * 50 * t)
    else:
        for b in beats:
            d = t - b
            clean += np.where(d < 0, np.exp(-0.5 * (d / 0.07) ** 2), np.exp(-0.5 * (d / 0.14) ** 2))
            clean += 0.35 * np.exp(-0.5 * ((d - 0.32) / 0.08) ** 2)  # diastolic wave
        mains = 0.0
    wander = 0.35 * np.sin(2 * np.pi * 0.15 * t + 0.3) + 0.2 * np.sin(2 * np.pi * 0.04 * t)
    sig = clean + wander + mains + rng.normal(0, 0.04, t.size)
    raw_counts = np.round(2048 + 400 * sig).astype(int)  # looks like 12-bit ADC output
    return pd.DataFrame({"time_s": np.round(t, 4), kind.lower(): raw_counts}).to_csv(index=False).encode()

def apply_filter(x: np.ndarray, fs: float, btype: str, low: float | None,
                 high: float | None, order: int, notch_hz: float | None):
    """Zero-phase Butterworth (+ optional notch). Returns (filtered, sos)."""
    nyq = fs / 2
    if btype in ("highpass", "bandpass") and not (low and 0 < low < nyq):
        raise ValueError(f"Low cut-off must be between 0 and the Nyquist frequency ({nyq:g} Hz).")
    if btype in ("lowpass", "bandpass") and not (high and 0 < high < nyq):
        raise ValueError(f"High cut-off must be below the Nyquist frequency ({nyq:g} Hz).")
    if btype == "bandpass" and low >= high:
        raise ValueError("Low cut-off must be lower than the high cut-off.")

    wn = [low, high] if btype == "bandpass" else (low if btype == "highpass" else high)
    sos = sps.butter(order, wn, btype=btype, fs=fs, output="sos")
    edge = high if btype == "lowpass" else low
    padlen = int(min(x.size - 1, round(3 * fs / edge)))  # generous padding limits edge transients
    y = sps.sosfiltfilt(sos, x, padlen=padlen)
    if notch_hz and notch_hz < nyq:
        b, a = sps.iirnotch(notch_hz, Q=30, fs=fs)
        y = sps.filtfilt(b, a, y)
    return y, sos

def _blocks(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(m == 1), np.flatnonzero(m == -1)

def _merge_close(peaks, amp: np.ndarray, min_dist: int) -> np.ndarray:
    kept: list[int] = []
    for p in peaks:
        if kept and p - kept[-1] < min_dist:
            if amp[p] > amp[kept[-1]]:
                kept[-1] = int(p)
        else:
            kept.append(int(p))
    return np.asarray(kept, dtype=int)

def _detect_ecg(x: np.ndarray, fs: float, max_bpm: float) -> np.ndarray:
    """R-peaks: QRS blocks where the smoothed |gradient| exceeds 1.5x its local mean."""
    grad = np.abs(np.gradient(x))
    smooth = uniform_filter1d(grad, max(1, int(round(0.10 * fs))), mode="nearest")
    local = uniform_filter1d(smooth, max(1, int(round(0.75 * fs))), mode="nearest")
    starts, ends = _blocks(smooth > 1.5 * local)
    if starts.size == 0:
        return np.array([], dtype=int)
    widths = ends - starts
    keep = widths >= 0.4 * widths.mean()
    cand = [s + int(np.argmax(x[s:e])) for s, e in zip(starts[keep], ends[keep])]
    peaks = _merge_close(cand, x, int(round(60 / max_bpm * fs)))
    if peaks.size >= 4:  # drop T-waves / noise blips far smaller than typical R-waves
        r = max(1, int(round(0.1 * fs)))
        amp = np.array([x[p] - x[max(0, p - r):p + r + 1].min() for p in peaks])
        peaks = peaks[amp >= 0.3 * np.median(amp)]
    return peaks

def _elgendi_pass(x: np.ndarray, fs: float, max_bpm: float, beat_window: float) -> np.ndarray:
    y = np.clip(x, 0, None) ** 2
    w1 = max(1, int(round(0.111 * fs)))             # ~ systolic peak duration
    w2 = max(1, int(round(beat_window * fs)))       # ~ one beat duration
    ma_peak = uniform_filter1d(y, w1, mode="nearest")
    ma_beat = uniform_filter1d(y, w2, mode="nearest")
    starts, ends = _blocks(ma_peak > ma_beat + 0.02 * y.mean())
    keep = (ends - starts) >= w1
    cand = [s + int(np.argmax(x[s:e])) for s, e in zip(starts[keep], ends[keep])]
    peaks = _merge_close(cand, x, int(round(60 / max_bpm * fs)))
    if peaks.size >= 4:  # drop diastolic waves: pulse amplitude (peak - preceding foot) too small
        r = max(1, int(round(0.3 * fs)))
        amp = np.array([x[p] - x[max(0, p - r):p + 1].min() for p in peaks])
        peaks = peaks[amp >= 0.3 * np.median(amp)]
    return peaks

def _detect_ppg(x: np.ndarray, fs: float, max_bpm: float) -> np.ndarray:
    """Systolic peaks: Elgendi et al. (2013) two-moving-average method.

    The beat-length window (667 ms in the paper) is shortened on a second pass
    when the first pass finds a fast pulse, so small beats at high rates are kept.
    """
    peaks = _elgendi_pass(x, fs, max_bpm, 0.667)
    if peaks.size >= 4:
        window = float(np.clip(0.9 * np.median(np.diff(peaks)) / fs, 0.3, 0.667))
        if window < 0.667:
            peaks = _elgendi_pass(x, fs, max_bpm, window)
    return peaks

def detect_beats(filtered: np.ndarray, fs: float, kind: str, polarity: str, max_bpm: float):
    """Return (peak indices, inverted?). Peaks are aligned to `filtered`."""
    if filtered.size < fs * 2 or np.ptp(filtered) <= 1e-9 * max(1.0, float(np.max(np.abs(filtered)))):
        return np.array([], dtype=int), False
    pad = int(min(filtered.size - 1, 6 * fs))
    if kind == "PPG":   # detector expects a 0.5-8 Hz pulse wave centred on zero
        sos = sps.butter(2, [0.5, min(8.0, 0.45 * fs)], "bandpass", fs=fs, output="sos")
    else:               # only centre it - QRS energy must survive
        sos = sps.butter(2, 0.5, "highpass", fs=fs, output="sos")
    xc = sps.sosfiltfilt(sos, filtered, padlen=pad)

    if polarity == "Auto":
        score = skew(xc) if kind == "ECG" else skew(np.diff(xc))  # PPG: fast upstroke, slow decay
        inverted = bool(np.isfinite(score) and score < 0)
    else:
        inverted = polarity == "Inverted"
    xo = -xc if inverted else xc

    peaks = _detect_ecg(xo, fs, max_bpm) if kind == "ECG" else _detect_ppg(xo, fs, max_bpm)
    # refine onto the displayed (user-filtered) signal so markers sit on the visible peaks
    target = -filtered if inverted else filtered
    r = max(1, int(round(0.05 * fs)))
    refined = [max(0, p - r) + int(np.argmax(target[max(0, p - r):p + r + 1])) for p in peaks]
    return np.unique(np.asarray(refined, dtype=int)), inverted

def analyze_rr(peaks: np.ndarray, fs: float, min_bpm: float, max_bpm: float) -> dict:
    nan = float("nan")
    out = {"n_beats": int(peaks.size), "mean_hr": nan, "median_hr": nan, "min_hr": nan,
           "max_hr": nan, "sdnn": nan, "rmssd": nan, "pnn50": nan, "sd1": nan, "sd2": nan,
           "median_rr": nan, "longest_rr": nan, "longest_rr_at": nan, "excluded_pct": nan,
           "outlier_pct": nan, "irregular": False,
           "n_intervals": 0, "n_excluded": 0, "rr": np.array([]), "t_rr": np.array([]),
           "valid": np.array([], dtype=bool), "hr_inst": np.array([])}
    if peaks.size < 3:
        return out

    tp = peaks / fs
    rr = np.diff(tp)
    plausible = (rr >= 60 / max_bpm) & (rr <= 60 / min_bpm)
    local = (pd.Series(np.where(plausible, rr, np.nan))
             .rolling(7, center=True, min_periods=2).median().to_numpy())
    fallback = np.nanmedian(rr[plausible]) if plausible.any() else np.median(rr)
    local = np.where(np.isfinite(local), local, fallback)
    regular = plausible & (np.abs(rr - local) / local <= 0.20)
    outlier_pct = float(100 * (~regular).mean())
    # Sinus-like rhythm: drop outliers (ectopics / missed or extra detections).
    # Irregular rhythm (>20 % outliers, e.g. AF): every plausible interval is real data.
    irregular = outlier_pct > 20
    valid = plausible if irregular else regular

    out.update(rr=rr, t_rr=tp[1:], valid=valid, hr_inst=60 / rr, n_intervals=int(rr.size),
               n_excluded=int((~valid).sum()), excluded_pct=float(100 * (~valid).mean()),
               outlier_pct=outlier_pct, irregular=irregular,
               longest_rr=float(rr.max() * 1000), longest_rr_at=float(tp[:-1][np.argmax(rr)]))
    nn = rr[valid]
    if nn.size == 0:
        return out
    hr_nn = 60 / nn
    # min / max over a 5-beat moving average: single-beat extremes are mostly noise
    hr_smooth = pd.Series(hr_nn).rolling(5, center=True, min_periods=min(3, nn.size)).mean().dropna()
    out.update(mean_hr=float(60 / nn.mean()), median_hr=float(np.median(hr_nn)),
               min_hr=float(hr_smooth.min()), max_hr=float(hr_smooth.max()),
               median_rr=float(np.median(nn)))
    if nn.size >= 2:
        out["sdnn"] = float(np.std(nn, ddof=1) * 1000)
    pair = valid[1:] & valid[:-1]          # successive differences only between clean neighbours
    if pair.sum() >= 2:
        a, b = rr[:-1][pair], rr[1:][pair]
        sd = b - a
        out.update(rmssd=float(np.sqrt(np.mean(sd ** 2)) * 1000),
                   pnn50=float(np.mean(np.abs(sd) > 0.05) * 100),
                   sd1=float(np.std((b - a) / np.sqrt(2), ddof=1) * 1000),
                   sd2=float(np.std((b + a) / np.sqrt(2), ddof=1) * 1000))
    return out

def beat_template(x: np.ndarray, peaks: np.ndarray, fs: float, median_rr: float) -> dict | None:
    """Ensemble-average beat and template-correlation signal quality index."""
    if peaks.size < 3 or not is_num(median_rr):
        return None
    pre, post = int(round(0.35 * median_rr * fs)), int(round(0.60 * median_rr * fs))
    sel = peaks[(peaks - pre >= 0) & (peaks + post < x.size)]
    if sel.size < 3:
        return None
    if sel.size > MAX_TEMPLATE_BEATS:
        sel = sel[np.linspace(0, sel.size - 1, MAX_TEMPLATE_BEATS).astype(int)]
    offsets = np.arange(-pre, post + 1)
    segs = x[sel[:, None] + offsets[None, :]]
    segs = segs - segs.mean(axis=1, keepdims=True)
    template = segs.mean(axis=0)
    denom = np.linalg.norm(segs, axis=1) * np.linalg.norm(template) + 1e-12
    corr = segs @ template / denom
    return {"t_ms": offsets / fs * 1000, "segs": segs, "template": template,
            "sd": segs.std(axis=0), "corr": corr, "sqi": float(np.mean(corr))}

def sqi_label(sqi: float) -> str:
    if not is_num(sqi):
        return "n/a"
    return "Good" if sqi >= 0.86 else ("Fair" if sqi >= 0.70 else "Poor")

def welch_db(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    nper = int(min(x.size, max(256, 2 ** int(np.ceil(np.log2(fs * 4))))))
    f, p = sps.welch(x - np.mean(x), fs=fs, nperseg=nper)
    return f, 10 * np.log10(p + 1e-20)

def build_findings(kind: str, res: dict, sqi: float, duration: float) -> list[tuple[str, str]]:
    """List of (level, text); level is 'ok', 'warn' or 'info'."""
    items: list[tuple[str, str]] = []
    rate = "heart rate" if kind == "ECG" else "pulse rate"
    hr = res["mean_hr"]
    if not is_num(hr):
        items.append(("warn", "Too few clean beats were detected to estimate the "
                              f"{rate}. Check the signal type, column, polarity and filter settings."))
        return items

    if hr < 60:
        items.append(("warn", f"Average {rate} of {hr:.0f} BPM is below 60 BPM (bradycardia range "
                              "for a resting adult). This can be normal in athletes or during sleep."))
    elif hr > 100:
        items.append(("warn", f"Average {rate} of {hr:.0f} BPM is above 100 BPM (tachycardia range "
                              "at rest). Expected during exercise, stress, fever or after caffeine."))
    else:
        level = "info" if sqi_label(sqi) == "Poor" else "ok"
        items.append((level, f"Average {rate} of {hr:.0f} BPM is within the typical resting adult "
                             "range (60–100 BPM)."))

    if res["irregular"]:
        items.append(("warn", f"Irregular beat-to-beat timing: {res['outlier_pct']:.0f}% of intervals "
                              "deviate more than 20% from their neighbours. This may reflect an "
                              "arrhythmia (e.g. atrial fibrillation or frequent ectopic beats) or "
                              "missed / extra detections caused by noise. Inspect the trace. Rate "
                              "and variability use all physiologically plausible intervals."))
    elif res["n_excluded"] > 0:
        items.append(("info", f"{res['n_excluded']} of {res['n_intervals']} intervals were excluded "
                              "as likely ectopic beats or detection artifacts."))
    else:
        items.append(("ok", "Regular beat-to-beat timing with no outlier intervals."))

    if is_num(res["longest_rr"]) and res["longest_rr"] > 2000:
        items.append(("warn", f"Longest interval is {res['longest_rr'] / 1000:.1f} s, starting at "
                              f"t = {res['longest_rr_at']:.1f} s: a possible pause or missed beats."))

    label = sqi_label(sqi)
    if label == "Poor":
        items.append(("warn", f"Poor signal quality (SQI {sqi:.2f}). Beat shapes are inconsistent, so "
                              "heart rate and variability values may be unreliable."))
    elif label == "Fair":
        items.append(("info", f"Fair signal quality (SQI {sqi:.2f}). Some beats differ from the "
                              "average beat shape."))
    elif label == "Good":
        items.append(("ok", f"Good signal quality (SQI {sqi:.2f}): consistent beat morphology."))

    if duration < 60:
        items.append(("info", "Recording is shorter than 1 minute, so SDNN and RMSSD are indicative "
                              "only (5-minute recordings are the standard for short-term HRV)."))
    if kind == "PPG":
        items.append(("info", "PPG intervals give pulse-rate variability (PRV), which approximates "
                              "but does not equal ECG-based HRV."))
    return items

def build_report(ctx: dict) -> str:
    res, kind = ctx["res"], ctx["kind"]
    rate = "Heart rate" if kind == "ECG" else "Pulse rate"
    hrv = "HRV" if kind == "ECG" else "PRV"
    lines = [
        f"## {kind} summary report",
        f"*Generated {datetime.now():%d %b %Y, %H:%M} · source `{ctx['file_name']}`, "
        f"column `{ctx['sig_col']}`*",
        "",
        "#### Findings",
        *[f"- {ICONS[level]} {text}" for level, text in ctx["findings"]],
        "",
        f"#### {rate}",
        "| Measure | Value |",
        "|---|---|",
        f"| Average | **{fmt(res['mean_hr'], 1, 'BPM')}** |",
        f"| Median | {fmt(res['median_hr'], 1, 'BPM')} |",
        f"| Minimum / maximum (5-beat average) | {fmt(res['min_hr'], 0)} / {fmt(res['max_hr'], 0, 'BPM')} |",
        f"| Beats detected | {res['n_beats']:,} |",
        f"| Intervals used | {res['n_intervals'] - res['n_excluded']:,} of {res['n_intervals']:,} "
        f"({fmt(res['excluded_pct'], 1, '%')} excluded) |",
        "",
        f"#### Variability ({hrv})",
        "| Measure | Value | Meaning |",
        "|---|---|---|",
        f"| SDNN | {fmt(res['sdnn'], 1, 'ms')} | overall variability of clean intervals |",
        f"| RMSSD | {fmt(res['rmssd'], 1, 'ms')} | short-term (beat-to-beat) variability |",
        f"| pNN50 | {fmt(res['pnn50'], 1, '%')} | successive intervals differing by > 50 ms |",
        f"| Poincaré SD1 / SD2 | {fmt(res['sd1'], 1)} / {fmt(res['sd2'], 1, 'ms')} | short- / long-term spread |",
        f"| Median interval | {fmt(res['median_rr'] * 1000 if is_num(res['median_rr']) else None, 0, 'ms')} | |",
        f"| Longest interval | {fmt(res['longest_rr'], 0, 'ms')} | |",
        "",
        "#### Signal quality",
        "| Measure | Value |",
        "|---|---|",
        f"| Template-correlation SQI | {fmt(ctx['sqi'], 2)} ({sqi_label(ctx['sqi'])}) |",
        f"| Polarity | {ctx['polarity_text']} |",
        "",
        "#### Recording & processing",
        "| Item | Value |",
        "|---|---|",
        f"| Signal type | {kind} |",
        f"| Sampling rate | {ctx['fs']:,.2f} Hz |",
        f"| Duration | {ctx['duration']:,.1f} s ({fmt_duration(ctx['duration'])}) |",
        f"| Samples | {ctx['n_samples']:,} |",
        f"| Filter | Butterworth {ctx['filter_text']} |",
        f"| Mains notch | {ctx['notch_text']} |",
        f"| Plausible HR range | {ctx['min_bpm']}–{ctx['max_bpm']} BPM |",
    ]
    lines += [f"| Preprocessing | {n} |" for n in ctx["notes"]]
    lines += [
        "",
        "> **Disclaimer:** automated analysis for research and education only. It is not a "
        "medical device and must not be used for diagnosis or treatment decisions. Findings "
        "should be reviewed by a qualified clinician.",
    ]
    return "\n".join(lines)

def processed_csv(t: np.ndarray, raw: np.ndarray, filt: np.ndarray, peaks: np.ndarray) -> bytes:
    is_beat = np.zeros(t.size, dtype=np.int8)
    is_beat[peaks] = 1
    df = pd.DataFrame({"time_s": t, "raw": raw, "filtered": filt, "beat": is_beat})
    return df.to_csv(index=False, float_format="%.6g").encode()

def beats_table(res: dict, peaks: np.ndarray, fs: float) -> pd.DataFrame:
    n = peaks.size
    rr_ms = np.full(n, np.nan)
    hr = np.full(n, np.nan)
    status = np.array(["first beat"] + [""] * (n - 1), dtype=object) if n else np.array([], dtype=object)
    if n >= 2 and res["rr"].size:
        rr_ms[1:] = res["rr"] * 1000
        hr[1:] = res["hr_inst"]
        status[1:] = np.where(res["valid"], "used", "excluded")
    return pd.DataFrame({"beat": np.arange(1, n + 1), "time_s": np.round(peaks / fs, 3),
                         "interval_ms": np.round(rr_ms, 1), "rate_bpm": np.round(hr, 1),
                         "status": status})
