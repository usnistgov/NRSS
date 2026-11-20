"""Render the film validation maps, depth errors and signed peak cuts."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter
import numpy as np


TITLES = {
    "hcp_flat": "HCP, flat",
    "hcp_tilted": "HCP, tilted",
    "bcc_flat": "BCC, flat",
    "bcc_tilted": "BCC, tilted",
    "hcp_hann": "HCP, tilted with XY Hann",
}
HCP_COLOR = "xkcd:bright blue"
BCC_COLOR = "xkcd:bright red"
HANN_COLOR = "xkcd:frog green"
ROUTES = ("native", "virtual_1024", "direct")
ROUTE_LABELS = ("Native · 128 planes", "Virtual · 1024 planes", "Continuous reference")
ROUTE_COLORS = (BCC_COLOR, HCP_COLOR, HANN_COLOR)


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save(fig, output, name):
    for ext in ("pdf", "png"):
        fig.savefig(output / f"{name}.{ext}", dpi=100)
    plt.close(fig)


def depth_errors(root, output):
    fig, axes = plt.subplots(1, 3, figsize=(19.2, 10.8), sharey=True, layout="constrained")
    fig.suptitle("High-q intensity error versus transform depth", fontsize=23)
    depths = (128, 256, 512, 1024)
    groups = (("285", "common", "285 eV · 0.95–1.25 nm⁻¹"),
              ("570", "common", "570 eV · 0.95–1.25 nm⁻¹"),
              ("570", "570-only", "570 eV · 1.90–2.40 nm⁻¹"))
    for ax, (energy, band, title) in zip(axes, groups):
        for name, label in TITLES.items():
            measured = {row["mode"]: 100 * float(row["relative_l2"])
                        for row in rows(root / name / energy / "bands.csv") if row["band"] == band}
            colors = HANN_COLOR if name == "hcp_hann" else HCP_COLOR if name.startswith("hcp") else BCC_COLOR
            values = [measured["native"], *[measured[f"virtual_{m}"] for m in depths[1:]]]
            ax.plot(depths, values, marker="o", color=colors, ls="--" if name.endswith("flat") else "-",
                    lw=1.7, label=label)
        ax.set(xscale="log", yscale="log", xticks=depths, xticklabels=[str(m) for m in depths],
               xlabel="Axial transform depth (planes)", title=title)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=.20, which="both")
    axes[0].set_ylabel("Relative L2 intensity error (%)")
    axes[-1].legend(frameon=False, loc="upper right", fontsize=13)
    save(fig, output, "high-q-depth-error")


def maps(root, output, name):
    data = root / name / "570"
    anchor = json.loads((data / "anchor.json").read_text())
    with np.load(data / "detector.npz") as capture:
        qx, qy = capture["qx"], capture["qy"]
        q_perp = np.hypot(qx[None, :], qy[:, None])
        band = (q_perp >= 1.9) & (q_perp < 2.4)
        images = [np.where(band, capture[route] / anchor["intensity"], np.nan) for route in ROUTES]
    shown = np.concatenate([frame[np.isfinite(frame) & (frame > 0)] for frame in images])
    vmax = np.percentile(shown, 99.95)
    norm = LogNorm(vmin=vmax / 10000, vmax=vmax)
    fig, axes = plt.subplots(1, 3, figsize=(19.2, 10.8), sharex=True, sharey=True, layout="constrained")
    fig.suptitle(f"{TITLES[name]} · 570 eV · 1.90–2.40 nm⁻¹", fontsize=22)
    step = qx[1] - qx[0]
    for ax, label, panel in zip(axes, ROUTE_LABELS, images):
        im = ax.imshow(panel, origin="lower", cmap="inferno", norm=norm, interpolation="nearest",
                       extent=(qx[0]-step/2, qx[-1]+step/2, qy[0]-step/2, qy[-1]+step/2), rasterized=True)
        ax.set(title=label, xlim=(-2.43, 2.43), ylim=(-2.43, 2.43), xlabel=r"$q_x$ (nm$^{-1}$)")
    axes[0].set_ylabel(r"$q_y$ (nm$^{-1}$)")
    cbar = fig.colorbar(im, ax=axes, shrink=.69, extend="max")
    cbar.set_label(f"Intensity / direct {tuple(anchor['hkl'])} peak")
    save(fig, output, name + "-outer-band")


def line_cuts(root, output, name):
    data = root / name / "570"
    anchor = json.loads((data / "anchor.json").read_text())
    candidates = [
        row for row in rows(data / "peaks.csv")
        if row["band"] == "570-only" and row["mode"] == "direct" and row["local_peak"] == "True"
        and row["bcc_odd_only"] == "False" and float(row["peak_to_background"]) >= anchor["contrast"]
        and float(row["peak_over_anchor"]) >= 1.e-5
    ]
    classes = (
        ("Near an axial bin", [r for r in candidates if float(r["fraction_to_nearest_axial_bin"]) <= .1]),
        ("Between axial bins", [r for r in candidates if float(r["fraction_to_nearest_axial_bin"]) >= .4]),
    )
    if any(not items for _, items in classes):
        return
    fig, axes = plt.subplots(1, 2, figsize=(19.2, 10.8), sharey=True, layout="constrained")
    fig.suptitle(f"{TITLES[name]} · indexed Bragg cuts at 570 eV", fontsize=22)
    output_rows = []
    with np.load(data / "detector.npz") as capture:
        qx, qy = capture["qx"], capture["qy"]
        for ax, (label, items) in zip(axes, classes):
            spot = max(items, key=lambda row: float(row["peak_over_anchor"]))
            y_index = int(np.argmin(abs(qy - float(spot["qy_found"]))))
            center = int(np.argmin(abs(qx - float(spot["qx"]))))
            cx = slice(center - 10, center + 11)
            offset = qx[cx] - float(spot["qx"])
            triplet = (int(spot["h"]), int(spot["k"]), int(spot["l"]))
            for mode, route_label, color in zip(ROUTES, ROUTE_LABELS, ROUTE_COLORS):
                intensity = capture[mode][y_index, cx] / anchor["intensity"]
                ax.plot(offset, intensity, color=color, lw=1.6, ls="--" if mode == "direct" else "-",
                        label=route_label)
                output_rows.extend(dict(group=label, contributors=spot["contributors"], mode=mode,
                                        qx=float(x), qy=float(qy[y_index]), qx_offset=float(dx), intensity=float(v))
                                   for x, dx, v in zip(qx[cx], offset, intensity))
            ax.axvline(0, color="white", ls=":", lw=1, alpha=.5)
            ax.set(yscale="log", title=f"{label} · {triplet}", xlabel=r"$q_x - q_{x,\mathrm{pred}}$ (nm$^{-1}$)")
            ax.grid(alpha=.18)
    axes[0].set_ylabel(f"Intensity / direct {tuple(anchor['hkl'])} peak")
    axes[0].legend(frameon=False)
    with (output / f"{name}-signed-cuts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    save(fig, output, name + "-signed-cuts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")
    plt.rcParams.update({"font.size": 16, "axes.titlesize": 18, "axes.labelsize": 17})
    depth_errors(args.capture, args.out)
    for case in ("hcp_tilted", "bcc_tilted", "hcp_hann"):
        maps(args.capture, args.out, case)
        line_cuts(args.capture, args.out, case)
