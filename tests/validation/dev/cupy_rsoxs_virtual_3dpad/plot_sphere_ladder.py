"""Export the full-panel sphere campaign as a CSV table and comparison figure."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

from sphere_reference import D, K, RADIUS, ROOTS, SIZES, axis, forward_scale, q_magnitude, sphere_amplitude


def read_run(directory):
    rows, lines = [], {}
    for size in SIZES:
        folder = directory / f"M{size}"
        if not folder.exists():
            continue
        cpu = (folder/"cpu.json").exists()
        if cpu:
            m = json.loads((folder/"cpu.json").read_text())
            row = {"source":"CPU reference", "fill":m["fill"], "cpu_calculation_s":m["calculation_s"],
                   "host_peak_MiB":m["peak_process_MiB_through_calculation"],
                   "host_peak_including_scoring_MiB":m["peak_process_MiB_including_scoring"]}
        else:
            accuracy = json.loads((folder/"accuracy.json").read_text())
            if accuracy.get("failures"):
                raise ValueError(f"Incomplete qualification at {folder}: {accuracy['failures']}")
            m = accuracy["metrics"]
            t = json.loads((folder/"timing.json").read_text())
            mem = json.loads((folder/"memory.json").read_text())
            row = {"source":"NRSS GPU", "arm":accuracy["arm"], "fill":accuracy["fill_id"],
                   "fp32_optical_power_scale":m["fp32_optical_power_scale"],
                   "nominal_voxel_anchor_max_abs_over_I0":m["nominal_voxel_anchor_max_abs_over_I0"],
                   "gpu_first_s":t["first_s"],
                   "gpu_warm_median_s":t["warm_median_s"], "gpu_warm_mad_s":t["warm_mad_s"],
                   "gpu_warm_min_s":min(t["warm_s"]), "gpu_warm_max_s":max(t["warm_s"]),
                   "gpu_worker_peak_MiB":mem["peak_worker_MiB"], "host_peak_MiB":mem["peak_host_rss_MiB"],
                   "device_baseline_MiB":mem["nvml"]["baseline_gpu_used_mib"],
                   "device_peak_MiB":mem["nvml"]["peak_gpu_used_mib"],
                   "device_peak_delta_MiB":mem["nvml"]["peak_gpu_delta_mib"],
                   "route":"+".join(m["routes"]), "revision":t["git"], "gpu":t["gpu_name"]}
            row.update({f"full_voxel_{key}":val for key,val in m.get("full_voxel",{}).items()})
        row.update(M=size, panel_pixels=size**2, valid_pixels=m["valid_pixels"], dq_nm_inv=m["dq_nm_inv"],
                   detector_fp32_MiB=4*size**2/2**20, literal_field_GiB=4*size**3/2**30,
                   dc_over_unrepaired_I0=m["dc_simulated"]/m["unrepaired_I0"],
                   Rg_fit_nm=m["Rg_fit_nm"], guinier_fit_points=m["guinier_nonzero_points"],
                   voxel_anchor_max_abs_over_I0=m["anchor_max_abs_over_I0"])
        row["valid_pixels_per_s"] = row["valid_pixels"]/row.get("cpu_calculation_s", row.get("gpu_warm_median_s"))
        for band, stats in m["bands"].items():
            row[f"{band}_pixels"] = stats["pixels"]
            row[f"{band}_relative_l2"] = stats["relative_l2"]
        for i, entry in enumerate(m["minima"], start=1):
            minimum = entry["resolved_minimum"] or {}
            row[f"min{i}_q_bin_nm_inv"] = minimum.get("q_bin")
            row[f"min{i}_q_fit_nm_inv"] = minimum.get("q_quadratic")
            row[f"min{i}_I_bin_over_I0"] = minimum.get("I_bin_over_I0")
        stored = np.load(folder/"positive_qy.npy")
        if stored.shape != (size//2,):
            raise ValueError(f"Invalid +qy line at {folder}")
        qy = axis(size)[size//2:]
        select = (qy > 0) & (qy < K) & np.isfinite(stored)
        lines[size] = (q_magnitude(0., qy[select]), stored[select]/forward_scale(row["fill"]))
        rows.append(row)
    if not rows or len({(r["source"], r["fill"], r.get("arm")) for r in rows}) != 1:
        raise ValueError("Expected one CPU or GPU campaign with a consistent fill")
    return rows, lines


def plot(rows, lines, out):
    blue, orange, green, violet = "#13658a", "#b45c17", "#13786c", "#77538b"
    sizes = [r["M"] for r in rows]
    def column(key):
        return np.array([np.nan if r.get(key) is None else r[key] for r in rows], dtype=float)
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False,
                         "axes.labelcolor":"#263740", "axes.titlecolor":"#142c38",
                         "xtick.color":"#354954","ytick.color":"#354954",
                         "legend.frameon":False, "lines.linewidth":1.7})
    fig, ((whole, zoom), (errors, guinier), (runtime, memory)) = plt.subplots(3, 2, figsize=(11.6, 10.5))
    fig.subplots_adjust(left=.085, right=.98, top=.89, bottom=.062, wspace=.26, hspace=.49)
    mode = rows[0]["source"]
    medium = "vacuum" if rows[0]["fill"] is None else f"material {rows[0]['fill']}"
    fig.suptitle("Virtual XYZ padding of a 70 nm sphere", fontsize=21, x=.085, ha="left", color="#102b36", y=.97)
    source = mode + (" · "+rows[0]["arm"] if "arm" in rows[0] else "")
    fig.text(.085,.923, f"{source}  |  {medium} exterior  |  128³ source crop  |  1 nm voxels  |  285 eV",
             color="#49636b", fontsize=11)
    sample_sizes = sorted({sizes[0], sizes[len(sizes)//2], sizes[-1]})
    for ax in (whole, zoom):
        q = np.linspace(.0001, 2.1, 12000) if ax is whole else np.linspace(.103,.153,700)
        ax.plot(q, np.maximum(sphere_amplitude(q*RADIUS)**2, 1e-16), color="#222a2e",
                lw=1.05, label="Continuum sphere", zorder=3)
        for n, color, marker in zip(sample_sizes,(orange,blue,green),("o","s",None)):
            q, power = lines[n]
            if ax is zoom:
                use = (q > .09) & (q < .161)
                q, power = q[use], power[use]
            ax.plot(q, np.maximum(power,1e-16), color=color, marker=marker, ms=3.4, lw=1.05,
                    alpha=.9, label=f"{n}³")
        ax.set_xlabel(r"$|\mathbf{q}|$ (nm$^{-1}$)")
        ax.set_yscale("log")
        ax.set_ylabel(r"$I/I_0$ on $+q_y$")
    whole.set(xlim=(0, np.sqrt(2)*K), ylim=(1e-12, 2))
    whole.set_title("A   Full physical q range",loc="left",fontweight="bold")
    whole.legend(ncol=2,fontsize=8,loc="upper right",columnspacing=1)
    zoom.set(xlim=(.107,.15),ylim=(1e-8,.04))
    zoom.axvline(ROOTS[0]/RADIUS,color="#79888e",ls=":",lw=.9)
    zoom.set_title("B   First form-factor minimum",loc="left",fontweight="bold")
    errors.set_title("C   Full-panel error against the continuum",loc="left",fontweight="bold")
    for band, label, c in (("all_non_dc","All non-DC",blue),("lobes","2 ≤ qR ≤ 14",green),
                           ("high_q","q ≥ 1 nm⁻¹",orange)):
        errors.plot(sizes,column(band+"_relative_l2"),"o-",label=label,color=c,ms=4)
    errors.set(yscale="log",ylabel="Band-relative L2")
    errors.legend(fontsize=8)
    guinier.set_title("D   Guinier fit along $+q_y$",loc="left",fontweight="bold")
    guinier.plot(sizes,column("Rg_fit_nm"),"o-",color=blue,label=r"Fit over $0<qR_g\leq1$")
    guinier.axhline(np.sqrt(.6)*RADIUS,color=green,ls="--",label="Continuum radius of gyration")
    guinier.set(ylabel=r"Fitted $R_g$ (nm)",ylim=(27.03,27.67))
    guinier.legend(fontsize=8)
    runtime.set_title("E   Calculation time",loc="left",fontweight="bold")
    memory.set_title("F   Peak memory and detector size",loc="left",fontweight="bold")
    if mode == "CPU reference":
        runtime.plot(sizes,column("cpu_calculation_s"),"o-",color=blue,label="One CPU calculation")
        memory.plot(sizes,column("host_peak_MiB"),"o-",color=violet,label="Process RSS through calculation")
    else:
        runtime.plot(sizes,column("gpu_first_s"),"s--",color=orange,label="First synchronized run")
        runtime.plot(sizes,column("gpu_warm_median_s"),"o-",color=blue,label="Warm median")
        runtime.fill_between(sizes,column("gpu_warm_min_s"),column("gpu_warm_max_s"),color=blue,alpha=.13)
        memory.plot(sizes,column("gpu_worker_peak_MiB"),"o-",color=green,label="Sampled worker VRAM")
        memory.plot(sizes,column("host_peak_MiB"),"s--",color=violet,label="Worker host RSS")
    runtime.set(yscale="log",ylabel="Seconds")
    runtime.legend(fontsize=8)
    memory.plot(sizes,column("detector_fp32_MiB"),"s:",color=blue,label="One full FP32 output")
    memory.set(yscale="log",ylabel="MiB")
    memory.legend(fontsize=8)
    for ax in (errors, guinier, runtime, memory):
        ax.set_xscale("log",base=2)
        ax.set_xticks(sizes)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.set_xlabel("Logical cube side (voxels)")
    for ax in (whole, zoom, errors, guinier, runtime, memory):
        ax.grid(axis="y",alpha=.17)
    fig.savefig(out/"sphere_ladder.png",dpi=165)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args = parser.parse_args()
    rows, lines = read_run(args.directory)
    args.out.mkdir(parents=True,exist_ok=True)
    first = ["M","source","fill","panel_pixels","valid_pixels","dq_nm_inv"]
    fields = first + [k for k in rows[0] if k not in first]
    with (args.out/"sphere_ladder.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row, fill="vacuum" if row["fill"] is None else row["fill"]))
    plot(rows,lines,args.out)


if __name__=="__main__":
    main()
