"""Streamlit interface and charts for the ECG/PPG dashboard.

Install dependencies with ``pip install -r requirements.txt`` and run with
``streamlit run frontend.py``.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import signal as sps

import backend as _backend
from backend import (
    FILTER_TYPES,
    NOTCH_OPTIONS,
    SIGNAL_DEFAULTS,
    analyze_rr,
    beat_template,
    beats_table,
    build_findings,
    build_report,
    estimate_fs,
    fmt,
    fmt_duration,
    guess_columns,
    guess_signal_type,
    is_num,
    numeric_fraction,
    sqi_label,
)

COLORS = {
    "raw": "#8a8984",        # neutral grey - context, not the focus
    "filtered": "#2a78d6",   # blue - the cleaned signal
    "peak": "#eb6834",       # orange - detected beats
    "excluded": "#d03b3b",   # red - excluded intervals
    "normal_band": "rgba(12, 163, 12, 0.08)",
}

MAX_PLOT_POINTS = 40_000      # per trace, after min/max decimation

PLOT_CONFIG = {"displaylogo": False, "scrollZoom": False}

read_csv_bytes = st.cache_data(show_spinner=False)(_backend.read_csv_bytes)
to_seconds = st.cache_data(show_spinner=False)(_backend.to_seconds)
prepare_signal = st.cache_data(show_spinner=False)(_backend.prepare_signal)
make_demo_csv = st.cache_data(show_spinner=False)(_backend.make_demo_csv)
apply_filter = st.cache_data(show_spinner=False)(_backend.apply_filter)
detect_beats = st.cache_data(show_spinner=False)(_backend.detect_beats)
welch_db = st.cache_data(show_spinner=False)(_backend.welch_db)
processed_csv = st.cache_data(show_spinner=False)(_backend.processed_csv)

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
