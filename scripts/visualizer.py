# visualizer.py
import matplotlib
matplotlib.use("TkAgg")  # macOS-friendly
import matplotlib.pyplot as plt
import numpy as np
import queue as pyqueue
import time

def run_visualizer(viz_queue, num_fx_channels):
    plt.ion()
    fig = plt.figure(figsize=(9, 4.5))

    # --- Left: VA plane ---
    ax_va = fig.add_subplot(1, 2, 1)
    ax_va.set_title("Valence–Arousal")
    ax_va.set_xlim(-1, 1); ax_va.set_ylim(-1, 1)
    ax_va.axhline(0, lw=0.8); ax_va.axvline(0, lw=0.8)
    ax_va.set_xlabel("Valence"); ax_va.set_ylabel("Arousal")
    dry_pt,  = ax_va.plot([np.nan], [np.nan], "o", ms=8, label="Dry")
    full_pt, = ax_va.plot([np.nan], [np.nan], "s", ms=8, label="Full Mix")
    fx_pts    = [ax_va.plot([np.nan], [np.nan], "^", ms=7, label=f"FX {i}")[0]
                 for i in range(1, num_fx_channels+1)]
    ax_va.legend(loc="lower right", fontsize=8)
    waiting_txt = ax_va.text(0.5, 0.5, "waiting for data…",
                             transform=ax_va.transAxes, ha="center", va="center", alpha=0.6)

    # --- Right: gains ---
    ax_gain = fig.add_subplot(1, 2, 2)
    labels = ["Dry"] + [f"FX{i}" for i in range(1, num_fx_channels+1)] + ["Mix"]
    bars = ax_gain.bar(labels, [0]*(num_fx_channels+2))
    ax_gain.set_ylim(0, 127)
    ax_gain.set_ylabel("Gain (CC value)")
    info_txt = ax_gain.text(0.01, 0.98, "", transform=ax_gain.transAxes, va="top", ha="left")
    rule_txt = ax_gain.text(0.99, 0.98, "", transform=ax_gain.transAxes, va="top", ha="right")

    fig.tight_layout()
    fig.canvas.draw()
    plt.show(block=False)

    expected_bars = len(bars)
    last_state = None
    beat = 0

    while True:
        # drain queue, keep latest
        print('[viz] started')
        got = False
        try:
            while True:
                last_state = viz_queue.get(timeout=0.05)
                got = True
        except pyqueue.Empty:
            pass

        # UI heartbeat in the title so you know it’s repainting
        beat = (beat + 1) % 1000
        ax_va.set_title(f"Valence–Arousal  ·  ♥ {beat:03d}")

        if not last_state:
            fig.canvas.draw_idle()
            plt.pause(0.01)
            continue

        # got data -> hide waiting overlay
        if waiting_txt.get_visible():
            waiting_txt.set_visible(False)

        va = np.array(last_state.get("va", []), dtype=float)
        gains = list(last_state.get("gains", []))
        stress = int(last_state.get("stress", 0))
        attention = int(last_state.get("attention", 0))
        strength = int(last_state.get("strength", 127))
        rule_name = last_state.get("rule", "")

        # Update VA points (use sequences)
        if va.ndim == 2 and va.shape[1] == 2 and len(va) >= 2:
            dry_pt.set_data([float(va[0, 0])],  [float(va[0, 1])])
            full_pt.set_data([float(va[-1, 0])],[float(va[-1, 1])])
            for i, fx in enumerate(fx_pts):
                idx = 1 + i
                if idx < len(va) - 1:
                    fx.set_data([float(va[idx, 0])],[float(va[idx, 1])])
                else:
                    fx.set_data([np.nan],[np.nan])

        # Update gains (pad/trim)
        if len(gains) < expected_bars:
            gains = gains + [0]*(expected_bars - len(gains))
        elif len(gains) > expected_bars:
            gains = gains[:expected_bars]
        for b, h in zip(bars, gains):
            b.set_height(int(h))

        # HUD
        info_txt.set_text(f"Stress: {stress:3d}   Attention: {attention:3d}   Footpedal: {strength:3d}")
        rule_txt.set_text(rule_name)

        fig.canvas.draw_idle()
        plt.pause(0.01)


