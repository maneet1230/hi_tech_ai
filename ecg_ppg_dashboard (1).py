"""
ECG / PPG Diagnostic Dashboard - single-file Streamlit app
==========================================================

Upload a CSV of raw ECG or PPG samples, clean it with a zero-phase Butterworth
filter, compare raw vs. filtered signals, detect beats, compute heart rate and
heart-rate variability, and read / download a summary report.

Run
---
    pip install "streamlit>=1.36" plotly scipy pandas numpy
    streamlit run ecg_ppg_dashboard.py

CSV input
---------
* One column holding the signal, optionally a time column (seconds, ms, us or
  date-time strings). Extra columns are fine - you pick the one to analyse.
* Comma, semicolon, tab or pipe delimited; header row optional; comment lines
  starting with '#' at the top of the file are skipped.
* No time column? Enter the sampling rate in the sidebar.
* Not sure about the format? Choose "Demo signal" in the sidebar and download
  the demo CSV.

Processing pipeline
-------------------
1. Parse + tidy: sort by time, drop duplicate timestamps, interpolate missing
   samples, resample onto a uniform grid if timestamps have gaps or jitter.
2. Butterworth filter (band-, high- or low-pass, order 1-8) applied forward and
   backward (sosfiltfilt) so peaks are not shifted in time. Optional mains notch.
3. Beat detection
   * ECG: QRS detection from the steepness of the signal (smoothed absolute
     gradient vs. an adaptive threshold), R-peak = local maximum in each QRS block.
   * PPG: Elgendi (2013) two-moving-average systolic peak detector.
   Polarity is detected automatically (or set manually).
4. Heart rate from beat-to-beat (RR / pulse) intervals. Physiologically
   implausible intervals and intervals deviating >20 % from their local median
   (ectopic beats / artifacts) are excluded before HR and HRV are computed.
5. Signal quality index (SQI) = mean correlation of each beat with the average
   beat template.

Not a medical device. Results are for research and education only.
"""

from __future__ import annotations

import csv
import inspect
import io
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import signal as sps
from scipy.ndimage import uniform_filter1d
from scipy.stats import skew

# ════════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════════

SIGNAL_DEFAULTS = {
    "ECG": {"low": 0.5, "high": 40.0, "order": 4, "min_bpm": 30, "max_bpm": 220},
    "PPG": {"low": 0.5, "high": 8.0, "order": 4, "min_bpm": 30, "max_bpm": 200},
}
FILTER_TYPES = {"Band-pass": "bandpass", "High-pass": "highpass", "Low-pass": "lowpass"}
NOTCH_OPTIONS = {"Off": None, "50 Hz": 50.0, "60 Hz": 60.0}

COLORS = {
    "raw": "#8a8984",        # neutral grey - context, not the focus
    "filtered": "#2a78d6",   # blue - the cleaned signal
    "peak": "#eb6834",       # orange - detected beats
    "excluded": "#d03b3b",   # red - excluded intervals
    "normal_band": "rgba(12, 163, 12, 0.08)",
}
MAX_PLOT_POINTS = 40_000      # per trace, after min/max decimation
MAX_TEMPLATE_BEATS = 3_000    # beats used for the morphology template / SQI
PLOT_CONFIG = {"displaylogo": False, "scrollZoom": False}

TIME_TOKENS = {"time", "t", "timestamp", "times", "sec", "secs", "seconds", "ms",
               "millis", "msec", "datetime", "date", "elapsed", "us"}
INDEX_TOKENS = {"sample", "samples", "index", "idx", "n", "no", "count", "unnamed"}
PPG_TOKENS = {"ppg", "pleth", "bvp", "ir", "red", "green", "spo2", "pulse", "ppg1", "ppg2"}
ECG_TOKENS = {"ecg", "ekg", "lead", "mlii", "ii", "v1", "v2", "v5", "avr", "avl", "avf"}


# ════════════════════════════════════════════════════════════════════════════
# Small helpers
# ════════════════════════════════════════════════════════════════════════════

def _stretch_kwargs(fn) -> dict:
    """Full-width kwarg that works on both old and new Streamlit versions."""
    try:
        param = inspect.signature(fn).parameters.get("width")
        if param is not None and isinstance(param.default, str):
            return {"width": "stretch"}
    except (TypeError, ValueError):
        pass
    return {"use_container_width": True}


def show_chart(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, config=PLOT_CONFIG, key=key, **_stretch_kwargs(st.plotly_chart))


def show_table(df: pd.DataFrame, **kwargs) -> None:
    st.dataframe(df, hide_index=True, **_stretch_kwargs(st.dataframe), **kwargs)


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


# ════════════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
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


# ════════════════════════════════════════════════════════════════════════════
# Signal processing
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
def welch_db(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    nper = int(min(x.size, max(256, 2 ** int(np.ceil(np.log2(fs * 4))))))
    f, p = sps.welch(x - np.mean(x), fs=fs, nperseg=nper)
    return f, 10 * np.log10(p + 1e-20)


# ════════════════════════════════════════════════════════════════════════════
# Findings & report
# ════════════════════════════════════════════════════════════════════════════

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


ICONS = {"ok": "✅", "warn": "⚠️", "info": "ℹ️"}


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


# ════════════════════════════════════════════════════════════════════════════
# Plots
# ════════════════════════════════════════════════════════════════════════════

def minmax_downsample(t: np.ndarray, y: np.ndarray, max_points: int):
    """Keep each bucket's min and max so peaks survive decimation."""
    n = y.size
    if n <= max_points:
        return t, y
    bucket = int(np.ceil(n / (max_points // 2)))
    m = (n // bucket) * bucket
    yb, tb = y[:m].reshape(-1, bucket), t[:m].reshape(-1, bucket)
    lo, hi = yb.argmin(axis=1), yb.argmax(axis=1)
    first, second = np.minimum(lo, hi), np.maximum(lo, hi)
    rows = np.arange(yb.shape[0])
    t_out = np.column_stack([tb[rows, first], tb[rows, second]]).ravel()
    y_out = np.column_stack([yb[rows, first], yb[rows, second]]).ravel()
    return np.concatenate([t_out, t[m:]]), np.concatenate([y_out, y[m:]])


def _decimated(t: np.ndarray, y: np.ndarray) -> dict:
    td, yd = minmax_downsample(t, y, MAX_PLOT_POINTS)
    return {"x": td, "y": yd}


def _base_layout(fig: go.Figure, height: int, **kwargs) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=64, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
                      **kwargs)
    return fig


def fig_signal(t, raw, filt, peaks, window, overlay: bool, kind: str) -> go.Figure:
    i0, i1 = np.searchsorted(t, window)
    i1 = max(i1, i0 + 2)
    tw = t[i0:i1]
    pk = peaks[(peaks >= i0) & (peaks < i1)]
    peak_name = "R-peaks" if kind == "ECG" else "Systolic peaks"
    peak_marker = dict(color=COLORS["peak"], size=8, line=dict(width=1.5, color="white"))

    if overlay:
        def z(v):
            s = np.std(v)
            return (v - np.mean(v)) / (s if s > 0 else 1)
        rz, fz = z(raw[i0:i1]), z(filt[i0:i1])
        fig = go.Figure()
        fig.add_trace(go.Scattergl(**_decimated(tw, rz), name="Raw",
                                   line=dict(color=COLORS["raw"], width=1)))
        fig.add_trace(go.Scattergl(**_decimated(tw, fz), name="Filtered",
                                   line=dict(color=COLORS["filtered"], width=1.6)))
        fig.add_trace(go.Scattergl(x=t[pk], y=fz[pk - i0], mode="markers", name=peak_name,
                                   marker=peak_marker))
        fig.update_yaxes(title_text="Amplitude (z-score)")
        fig.update_xaxes(title_text="Time (s)")
        return _base_layout(fig, 480, hovermode="x unified")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=("Raw signal", "Filtered signal"))
    fig.add_trace(go.Scattergl(**_decimated(tw, raw[i0:i1]), name="Raw",
                               line=dict(color=COLORS["raw"], width=1)), row=1, col=1)
    fig.add_trace(go.Scattergl(**_decimated(tw, filt[i0:i1]), name="Filtered",
                               line=dict(color=COLORS["filtered"], width=1.6)), row=2, col=1)
    fig.add_trace(go.Scattergl(x=t[pk], y=filt[pk], mode="markers", name=peak_name,
                               marker=peak_marker), row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    return _base_layout(fig, 600, hovermode="x unified")


def fig_hr(res: dict, kind: str) -> go.Figure:
    t_rr, hr, valid = res["t_rr"], res["hr_inst"], res["valid"]
    fig = go.Figure()
    good = hr[valid]
    lo = min(55.0, float(good.min()) - 10) if good.size else 40.0
    hi = max(105.0, float(good.max()) + 10) if good.size else 160.0
    fig.add_hrect(y0=60, y1=100, fillcolor=COLORS["normal_band"], line_width=0, layer="below",
                  annotation_text="60–100 BPM", annotation_position="top left",
                  annotation_font_size=11)
    fig.add_trace(go.Scatter(x=t_rr[valid], y=good, mode="lines+markers",
                             name="Beat-to-beat rate", line=dict(color=COLORS["filtered"], width=1.6),
                             marker=dict(size=5),
                             hovertemplate="t = %{x:.2f} s<br>%{y:.1f} BPM<extra></extra>"))
    if (~valid).any():
        bad = hr[~valid]
        pad = 0.02 * (hi - lo)
        fig.add_trace(go.Scatter(x=t_rr[~valid], y=np.clip(bad, lo + pad, hi - pad), mode="markers",
                                 name="Excluded interval", customdata=bad,
                                 marker=dict(color=COLORS["excluded"], symbol="x", size=9),
                                 hovertemplate="t = %{x:.2f} s<br>%{customdata:.1f} BPM "
                                               "(excluded)<extra></extra>"))
    if is_num(res["mean_hr"]) and t_rr.size:
        fig.add_trace(go.Scatter(x=[0, float(t_rr[-1])], y=[res["mean_hr"]] * 2, mode="lines",
                                 name=f"Average {res['mean_hr']:.0f} BPM", hoverinfo="skip",
                                 line=dict(color=COLORS["peak"], dash="dash", width=1.5)))
    fig.update_yaxes(title_text="BPM", range=[lo, hi])
    fig.update_xaxes(title_text="Time (s)")
    title = "Heart rate over time" if kind == "ECG" else "Pulse rate over time"
    return _base_layout(fig, 420, title=dict(text=title, font=dict(size=14)))


def fig_poincare(res: dict) -> go.Figure:
    rr, valid = res["rr"] * 1000, res["valid"]
    pair = valid[1:] & valid[:-1]
    a, b = rr[:-1][pair], rr[1:][pair]
    fig = go.Figure()
    if a.size:
        lo, hi = float(min(a.min(), b.min())) - 30, float(max(a.max(), b.max())) + 30
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Identity",
                                 line=dict(color=COLORS["raw"], dash="dot", width=1),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=a, y=b, mode="markers", name="Interval pairs",
                                 marker=dict(color=COLORS["filtered"], size=7, opacity=0.6,
                                             line=dict(width=1, color="white")),
                                 hovertemplate="RRₙ %{x:.0f} ms<br>RRₙ₊₁ %{y:.0f} ms<extra></extra>",
                                 showlegend=False))
        fig.update_xaxes(range=[lo, hi])
        fig.update_yaxes(range=[lo, hi], scaleanchor="x", scaleratio=1)
    fig.update_xaxes(title_text="Interval n (ms)")
    fig.update_yaxes(title_text="Interval n+1 (ms)")
    return _base_layout(fig, 420, title=dict(text="Poincaré plot", font=dict(size=14)))


def fig_morphology(morph: dict, kind: str) -> go.Figure:
    t_ms, segs, tmpl, sd = morph["t_ms"], morph["segs"], morph["template"], morph["sd"]
    fig = go.Figure()
    show = segs[np.linspace(0, segs.shape[0] - 1, min(40, segs.shape[0])).astype(int)]
    xs = np.concatenate([np.append(t_ms, np.nan) for _ in show])
    ys = np.concatenate([np.append(s, np.nan) for s in show])
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"Individual beats ({len(show)})",
                             line=dict(color=COLORS["raw"], width=1), opacity=0.35,
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=np.concatenate([t_ms, t_ms[::-1]]),
                             y=np.concatenate([tmpl + sd, (tmpl - sd)[::-1]]),
                             fill="toself", fillcolor="rgba(42, 120, 214, 0.15)",
                             line=dict(width=0), name="±1 SD", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t_ms, y=tmpl, mode="lines", name="Average beat",
                             line=dict(color=COLORS["filtered"], width=2.5),
                             hovertemplate="%{x:.0f} ms<br>%{y:.3g}<extra></extra>"))
    fig.add_vline(x=0, line_dash="dot", line_color=COLORS["peak"], line_width=1)
    anchor = "R-peak" if kind == "ECG" else "systolic peak"
    fig.update_xaxes(title_text=f"Time from {anchor} (ms)")
    fig.update_yaxes(title_text="Amplitude (mean-removed)")
    return _base_layout(fig, 440, title=dict(text="Beat morphology", font=dict(size=14)))


def _cutoff_lines(fig: go.Figure, cutoffs: list[float]) -> None:
    for c in cutoffs:
        fig.add_vline(x=c, line_dash="dash", line_color=COLORS["peak"], line_width=1,
                      annotation_text=f"{c:g} Hz", annotation_position="top",
                      annotation_font_size=10)


def fig_spectrum(raw, filt, fs, cutoffs, notch_hz) -> go.Figure:
    f, p_raw = welch_db(raw, fs)
    _, p_filt = welch_db(filt, fs)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f, y=p_raw, name="Raw", line=dict(color=COLORS["raw"], width=1.2)))
    fig.add_trace(go.Scatter(x=f, y=p_filt, name="Filtered", line=dict(color=COLORS["filtered"], width=1.6)))
    _cutoff_lines(fig, cutoffs + ([notch_hz] if notch_hz and notch_hz < fs / 2 else []))
    top = float(max(p_raw.max(), p_filt.max())) + 5
    floor = max(float(p_filt[1:].min()) - 5, top - 125) if f.size > 1 else top - 125
    fig.update_yaxes(title_text="Power (dB/Hz)", range=[floor, top])
    fig.update_xaxes(title_text="Frequency (Hz)", range=[0, fs / 2])
    return _base_layout(fig, 420, title=dict(text="Power spectrum (Welch)", font=dict(size=14)),
                        hovermode="x unified")


def fig_response(sos, fs, cutoffs) -> go.Figure:
    freqz = getattr(sps, "freqz_sos", None) or sps.sosfreqz
    w, h = freqz(sos, worN=4096, fs=fs)
    gain_db = 20 * np.log10(np.maximum(np.abs(h) ** 2, 1e-10))  # |H|² - forward + backward pass
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w, y=gain_db, name="Effective gain", showlegend=False,
                             line=dict(color=COLORS["filtered"], width=2),
                             hovertemplate="%{x:.2f} Hz<br>%{y:.1f} dB<extra></extra>"))
    fig.add_hline(y=-6, line_dash="dot", line_color=COLORS["raw"], line_width=1,
                  annotation_text="−6 dB", annotation_position="bottom left",
                  annotation_font_size=10)
    _cutoff_lines(fig, cutoffs)
    fig.update_yaxes(title_text="Gain (dB)", range=[-80, 5])
    fig.update_xaxes(title_text="Frequency (Hz)", range=[0, fs / 2])
    return _base_layout(fig, 420, title=dict(text="Butterworth response (zero-phase)",
                                             font=dict(size=14)))


# ════════════════════════════════════════════════════════════════════════════
# Exports
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
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


# ════════════════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════════════════

def show_welcome() -> None:
    st.info("Upload a CSV in the sidebar, or pick **Demo signal** to try the dashboard.")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(
            "#### What it does\n"
            "1. **Cleans** the raw trace with a zero-phase Butterworth filter "
            "(defaults: ECG 0.5–40 Hz, PPG 0.5–8 Hz) and an optional mains notch.\n"
            "2. **Plots** raw vs. filtered signals with detected beats (Plotly: zoom, pan, hover).\n"
            "3. **Detects beats** and computes heart rate, HRV (SDNN, RMSSD, pNN50) and a "
            "signal-quality index.\n"
            "4. **Summarises** everything in a report you can download, with the processed "
            "signal and beat table as CSV."
        )
    with c2:
        st.markdown("#### Expected CSV")
        show_table(pd.DataFrame({"time_s": [0.000, 0.004, 0.008, 0.012],
                                 "ecg": [2051, 2049, 2056, 2073]}))
        st.caption("A time column (s, ms, µs or date-time) is optional; without one, enter the "
                   "sampling rate. Header row and delimiter are detected automatically.")


def main() -> None:
    st.set_page_config(page_title="ECG / PPG Diagnostic Dashboard", page_icon="🫀", layout="wide")
    st.title("ECG / PPG Diagnostic Dashboard")
    st.caption("Raw signal → Butterworth filtering → beat detection → heart rate, variability and "
               "a summary report. For research and education; not a medical device.")

    sb = st.sidebar

    # ── 1. Data ──────────────────────────────────────────────────────────────
    sb.header("1 · Data")
    source = sb.radio("Source", ["Upload CSV", "Demo signal"], horizontal=True,
                      label_visibility="collapsed")
    if source == "Upload CSV":
        upload = sb.file_uploader("CSV file", type=["csv", "txt", "tsv"])
        if upload is None:
            show_welcome()
            return
        raw_bytes, file_name = upload.getvalue(), upload.name
    else:
        demo_kind = sb.selectbox("Demo signal", ["ECG", "PPG"])
        raw_bytes, file_name = make_demo_csv(demo_kind), f"demo_{demo_kind.lower()}.csv"
        sb.download_button("Download demo CSV", raw_bytes, file_name, "text/csv")

    try:
        df = read_csv_bytes(raw_bytes)
    except Exception as exc:  # noqa: BLE001 - surface any parser problem to the user
        st.error(f"Could not read **{file_name}**: {exc}")
        return
    numeric_cols = [c for c in df.columns if numeric_fraction(df[c]) >= 0.9]
    if df.empty or not numeric_cols:
        st.error("No numeric signal column was found in the file.")
        return
    time_guess, sig_guess = guess_columns(df, numeric_cols)

    # ── 2. Columns & sampling ────────────────────────────────────────────────
    sb.header("2 · Columns & sampling")
    time_opts = ["(none)"] + list(df.columns)
    time_col = sb.selectbox("Time column", time_opts, key=f"time_{file_name}",
                            index=time_opts.index(time_guess) if time_guess else 0)
    sig_opts = [c for c in numeric_cols if c != time_col] or numeric_cols
    sig_col = sb.selectbox("Signal column", sig_opts, key=f"sig_{file_name}",
                           index=sig_opts.index(sig_guess) if sig_guess in sig_opts else 0)

    t_sec, fs_manual = None, None
    if time_col != "(none)":
        unit = sb.selectbox("Time unit", ["Auto", "s", "ms", "µs"], key=f"unit_{file_name}",
                            help="Auto guesses from the spacing between timestamps.")
        try:
            t_sec, unit_used = to_seconds(df[time_col], unit)
        except ValueError as exc:
            st.error(str(exc))
            return
        fs_est = estimate_fs(t_sec)
        sb.caption(f"Detected sampling rate ≈ **{fs_est:,.2f} Hz** (time unit: {unit_used})")
        if sb.checkbox("Override sampling rate", key=f"ovr_{file_name}",
                       help="Ignore the timestamps and assume uniform sampling at this rate."):
            fs_manual = sb.number_input("Sampling rate (Hz)", min_value=1.0, max_value=100_000.0,
                                        value=float(round(fs_est, 2)) if is_num(fs_est) else 250.0)
            t_sec = None
    else:
        fs_manual = sb.number_input("Sampling rate (Hz)", min_value=1.0, max_value=100_000.0,
                                    value=250.0, step=1.0, key=f"fs_{file_name}")
        sb.caption("No time column, so heart rate depends on this value: set it to your "
                   "device's sampling rate.")

    x_raw = pd.to_numeric(df[sig_col], errors="coerce").to_numpy(float)
    try:
        t, x, fs, notes = prepare_signal(t_sec, x_raw, fs_manual)
    except ValueError as exc:
        st.error(str(exc))
        return
    duration = x.size / fs
    if duration < 3 or x.size < 30:
        st.error(f"The recording is too short ({duration:.1f} s). At least 3 s is needed.")
        return

    # ── 3. Signal & filter ───────────────────────────────────────────────────
    sb.header("3 · Signal & filter")
    kind = sb.radio("Signal type", ["ECG", "PPG"], horizontal=True,
                    index=0 if guess_signal_type(sig_col) == "ECG" else 1,
                    key=f"kind_{file_name}_{sig_col}")
    d = SIGNAL_DEFAULTS[kind]
    nyq = fs / 2
    top = float(round(nyq * 0.98, 2))
    ftype_label = sb.selectbox("Butterworth type", list(FILTER_TYPES), key=f"ftype_{kind}")
    btype = FILTER_TYPES[ftype_label]
    low = high = None
    if btype in ("highpass", "bandpass"):
        low = sb.number_input("Low cut-off (Hz)", min_value=0.01, max_value=top,
                              value=float(min(d["low"], top / 2)), step=0.1, format="%.2f",
                              key=f"low_{kind}_{fs:.2f}",
                              help="Removes baseline wander and breathing drift.")
    if btype in ("lowpass", "bandpass"):
        high = sb.number_input("High cut-off (Hz)", min_value=0.1, max_value=top,
                               value=float(min(d["high"], round(nyq * 0.9, 2))), step=1.0,
                               format="%.2f", key=f"high_{kind}_{fs:.2f}",
                               help="Removes muscle noise and high-frequency interference.")
    order = sb.slider("Filter order", 1, 8, d["order"], key=f"order_{kind}",
                      help="Higher = steeper roll-off. Applied forward and backward, so the "
                           "effective order is doubled and there is no phase shift.")
    notch_label = sb.selectbox("Mains notch", list(NOTCH_OPTIONS), key=f"notch_{kind}",
                               help="Power-line interference: 50 Hz in India, Europe and most of "
                                    "Asia/Africa; 60 Hz in the Americas.")
    notch_hz = NOTCH_OPTIONS[notch_label]
    if notch_hz and notch_hz >= nyq:
        sb.caption(f"Notch ignored: {notch_hz:g} Hz is above the Nyquist frequency ({nyq:g} Hz).")
        notch_hz = None

    with sb.expander("Beat detection (advanced)"):
        polarity = st.selectbox("Signal polarity", ["Auto", "Normal", "Inverted"],
                                key=f"pol_{kind}",
                                help="Some ECG leads and many raw PPG sensors record the waveform "
                                     "upside-down. Auto detects this from the waveform's shape.")
        min_bpm, max_bpm = st.slider("Plausible rate range (BPM)", 20, 300,
                                     (d["min_bpm"], d["max_bpm"]), key=f"hr_{kind}",
                                     help="Intervals outside this range are treated as artifacts. "
                                          "The maximum also sets the minimum spacing between beats.")

    # ── Processing ───────────────────────────────────────────────────────────
    try:
        filt, sos = apply_filter(x, fs, btype, low, high, order, notch_hz)
    except ValueError as exc:
        st.error(f"Filter settings: {exc}")
        return
    peaks, inverted = detect_beats(filt, fs, kind, polarity, float(max_bpm))
    res = analyze_rr(peaks, fs, min_bpm, max_bpm)
    morph = beat_template(filt, peaks, fs, res["median_rr"])
    sqi = morph["sqi"] if morph else float("nan")
    findings = build_findings(kind, res, sqi, duration)

    cutoffs = [c for c in (low, high) if c]
    band_text = {"bandpass": f"{low:g}–{high:g} Hz" if low and high else "",
                 "highpass": f"> {low:g} Hz" if low else "",
                 "lowpass": f"< {high:g} Hz" if high else ""}[btype]
    filter_text = (f"{ftype_label.lower()} {band_text}, order {order} "
                   f"(zero-phase, effective order {2 * order})")
    polarity_text = f"{'inverted' if inverted else 'normal'} ({'auto-detected' if polarity == 'Auto' else 'set manually'})"

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown(f"**{file_name}** · {kind} · column `{sig_col}` · {fs:,.2f} Hz · "
                f"{fmt_duration(duration)} · {x.size:,} samples")
    if notes:
        with st.expander(f"Preprocessing notes ({len(notes)})"):
            for n in notes:
                st.markdown(f"- {n}")
    if res["n_beats"] < 3:
        st.warning("Fewer than 3 beats were detected. Check the signal type, signal column, "
                   "polarity (Beat detection → advanced) and the filter cut-offs.")

    rate_label = "Average HR" if kind == "ECG" else "Average pulse rate"
    m = st.columns(6)
    m[0].metric(rate_label, fmt(res["mean_hr"], 0, "BPM"),
                help="60 / mean of the clean beat-to-beat intervals.")
    m[1].metric("Range (BPM)", f"{fmt(res['min_hr'], 0)}–{fmt(res['max_hr'], 0)}"
                if is_num(res["min_hr"]) else "—", help="Lowest and highest rate, averaged over 5 beats.")
    m[2].metric("Beats detected", f"{res['n_beats']:,}",
                help=f"{res['n_excluded']} interval(s) excluded as artifacts / ectopics.")
    m[3].metric("SDNN", fmt(res["sdnn"], 0, "ms"), help="Standard deviation of clean intervals.")
    m[4].metric("RMSSD", fmt(res["rmssd"], 0, "ms"),
                help="Root mean square of successive interval differences.")
    m[5].metric("Signal quality", sqi_label(sqi),
                help=f"SQI {fmt(sqi, 2)}: mean correlation of each beat with the average beat "
                     "(1 = identical). Good ≥ 0.86, Fair ≥ 0.70.")

    tab_sig, tab_hr, tab_rep, tab_morph, tab_spec = st.tabs(
        ["Signal", "Heart rate", "Summary report", "Beat morphology", "Spectrum & filter"])

    # ── Signal tab ───────────────────────────────────────────────────────────
    with tab_sig:
        c1, c2 = st.columns([4, 1])
        dur_r = float(round(duration, 2))
        window = c1.slider("Time window (s)", 0.0, dur_r, (0.0, float(min(dur_r, 30.0))),
                           step=0.1 if duration <= 600 else 1.0, key=f"win_{file_name}_{dur_r}")
        overlay = c2.radio("Layout", ["Stacked", "Overlay"], horizontal=True,
                           help="Overlay z-scores both traces so they share one axis.") == "Overlay"
        show_chart(fig_signal(t, x, filt, peaks, window, overlay, kind), key="sig")
        st.caption(f"Butterworth {filter_text} · notch {notch_label.lower()} · polarity "
                   f"{polarity_text} · {res['n_beats']:,} beats marked. Long windows are "
                   "decimated for display (min/max per bucket, peaks preserved).")

    # ── Heart-rate tab ───────────────────────────────────────────────────────
    with tab_hr:
        if res["n_intervals"] == 0:
            st.info("Not enough beats to compute intervals.")
        else:
            c1, c2 = st.columns([2, 1])
            with c1:
                show_chart(fig_hr(res, kind), key="hr")
            with c2:
                show_chart(fig_poincare(res), key="poincare")
                st.caption(f"SD1 {fmt(res['sd1'], 1, 'ms')} (short-term) · SD2 "
                           f"{fmt(res['sd2'], 1, 'ms')} (long-term). A round, scattered cloud "
                           "suggests irregular timing; a tight comet shape is typical of sinus rhythm.")
        table = beats_table(res, peaks, fs)
        with st.expander(f"Beat table ({len(table):,} beats)"):
            show_table(table, height=320)
            st.download_button("Download beats CSV", table.to_csv(index=False).encode(),
                               f"{file_name.rsplit('.', 1)[0]}_beats.csv", "text/csv")

    # ── Report tab ───────────────────────────────────────────────────────────
    with tab_rep:
        report = build_report({
            "res": res, "kind": kind, "file_name": file_name, "sig_col": sig_col, "fs": fs,
            "duration": duration, "n_samples": int(x.size), "notes": notes, "sqi": sqi,
            "findings": findings, "filter_text": filter_text, "notch_text": notch_label,
            "polarity_text": polarity_text, "min_bpm": min_bpm, "max_bpm": max_bpm,
        })
        st.markdown(report)
        stem = file_name.rsplit(".", 1)[0]
        b1, b2, b3 = st.columns(3)
        b1.download_button("Download report (.md)", report.encode(), f"{stem}_report.md",
                           "text/markdown")
        b2.download_button("Download processed signal (.csv)", processed_csv(t, x, filt, peaks),
                           f"{stem}_processed.csv", "text/csv")
        b3.download_button("Download beats (.csv)", table.to_csv(index=False).encode(),
                           f"{stem}_beats.csv", "text/csv", key="beats_rep")

    # ── Morphology tab ───────────────────────────────────────────────────────
    with tab_morph:
        if morph is None:
            st.info("Not enough clean beats to build an average beat.")
        else:
            show_chart(fig_morphology(morph, kind), key="morph")
            st.caption(f"{morph['segs'].shape[0]:,} beats aligned on the detected peak. "
                       f"Signal quality index {morph['sqi']:.2f} ({sqi_label(morph['sqi'])}); "
                       f"{np.mean(morph['corr'] < 0.7):.0%} of beats correlate < 0.7 with the "
                       "average beat.")

    # ── Spectrum tab ─────────────────────────────────────────────────────────
    with tab_spec:
        c1, c2 = st.columns(2)
        with c1:
            show_chart(fig_spectrum(x, filt, fs, cutoffs, notch_hz), key="psd")
        with c2:
            show_chart(fig_response(sos, fs, cutoffs), key="resp")
        st.caption("Left: how much power the filter removed at each frequency (baseline wander "
                   "sits below ~0.5 Hz, mains hum at 50/60 Hz). Right: the filter's effective "
                   "gain. Because it runs forward and backward, the gain is squared, so the "
                   "cut-offs sit at −6 dB rather than −3 dB.")


if __name__ == "__main__":
    main()
