"""Report matched high-q Bragg spots and redraw the registration figure."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def index(rows, mode=None):
    chosen = [r for r in rows if (mode is None or r["mode"]==mode) and r["band"]!="anchor"]
    keyed = {(r["band"],round(float(r["qx"]),8),round(float(r["qy"]),8)):r for r in chosen}
    if len(keyed)!=len(chosen):
        raise ValueError("Repeated projected spot in one mode")
    return keyed


def count_rows(root,legacy=None,corrected=None):
    output = []
    paired_spots = []
    for f in sorted(root.glob("*/*/peaks.csv")):
        case,energy = f.parent.parent.name,f.parent.name
        data = read_rows(f)
        direct = index(data,"direct")
        if legacy is not None:
            old = index(read_rows(legacy/case/energy/"peaks.csv"))
            new = index(read_rows(corrected/case/energy/"peaks.csv"))
        else:
            old,new = index(data,"legacy"),index(data,"corrected")
        if old.keys()!=direct.keys() or new.keys()!=direct.keys():
            raise ValueError(f"Unequal candidate catalogs for {case} at {energy} eV")
        contrast = 100. if case=="simple_cubic_a30" else 15.
        for band in sorted({key[0] for key in direct}):
            expected = {k for k in direct if k[0]==band}
            valid = {k for k in expected if all(d[k]["support"]=="True" for d in (old,new,direct))}
            for floor in (1e-3,1e-4,1e-5):
                def hits(d):
                    return {k for k in valid if d[k]["local_peak"]=="True"
                            and float(d[k]["peak_to_background"])>=contrast
                            and float(d[k]["peak_over_anchor"])>=floor}
                a,b,t = hits(old),hits(new),hits(direct)
                output.append(dict(case=case,energy=energy,band=band,floor=floor,
                                   candidates=len(expected),common_support=len(valid),reference=len(t),
                                   legacy_count=len(a),corrected_count=len(b),legacy_match=len(a&t),
                                   corrected_match=len(b&t),confirmed_gains=len((b-a)&t),
                                   confirmed_losses=len((a-b)&t),legacy_only_vs_ref=len(a-t),
                                   corrected_only_vs_ref=len(b-t)))
                if floor==1e-4:
                    for key in sorted(expected):
                        reference = direct[key]
                        spot = dict(case=case,energy=energy,band=band,
                                    contributors=reference["contributors"],
                                    q_abs=reference["q_abs"],q_perp=reference["q_perp"],
                                    qx=reference["qx"],qy=reference["qy"],
                                    common_support=key in valid)
                        for label,arm,detected in (("legacy",old,a),("corrected",new,b),("direct",direct,t)):
                            row = arm[key]
                            spot[label+"_detected"] = (key in detected) if key in valid else None
                            for field in ("peak_over_anchor","peak_to_background","qx_found","qy_found"):
                                spot[label+"_"+field] = row[field]
                        paired_spots.append(spot)
    if not output:
        raise ValueError("No per-case peak tables found")
    return output,paired_spots


def plot(root, rows, out):
    line = read_rows(root/"hcp_a45"/"285"/"illustrative_cut.csv")
    fig,(ax,bx) = plt.subplots(1,2,figsize=(11.4,4.65),gridspec_kw={"width_ratios":[1,1.14]})
    fig.subplots_adjust(left=.084,right=.968,bottom=.21,top=.77,wspace=.53)
    fig.text(.084,.933,"High-q Bragg peaks on the unpadded NRSS lattice fixtures",
             size=18.5,weight="semibold")
    fig.text(.084,.86,"CPU calculation on 256 × 1024 × 1024 voxels. Both coordinate conventions sample the same morphology.",
             size=10.5,color="#4d5660")
    blue,orange,grey="#24689a","#b54c1a","#41454a"
    x = np.array([float(r["delta_qx"]) for r in line])
    for key,label,col,style in (("direct","Direct Fourier sum",grey,":"),
                                ("legacy","Legacy coordinates",blue,"--"),
                                ("corrected","FFT-bin coordinates",orange,"-")):
        ax.plot(x,1.e4*np.array([float(r[key]) for r in line]),style,color=col,lw=2.0,label=label)
    ax.axhline(1.,color="#9c8a78",ls=(0,(2,3)),lw=1)
    ax.set(xlim=(-.042,.042),ylim=(0,None),xlabel=r"$q_x-G_x$ (nm$^{-1}$)",
           ylabel=r"Intensity / $I_{100}$  ($10^{-4}$)")
    ax.set_title("HCP (8, 5, −6) at 285 eV",size=11.5,weight="semibold")
    ax.legend(frameon=False,fontsize=8.6,loc="upper left")
    selected = [(case,e,band) for e,band in (("285","common"),("570","570-only"))
                for case in ("hcp_a45","simple_cubic_a30")]
    labels = ["HCP · 285 eV","Simple cubic · 285 eV","HCP · 570 eV","Simple cubic · 570 eV"]
    for i,(case,e,band) in enumerate(selected):
        row=next(r for r in rows if r["case"]==case and r["energy"]==e
                 and r["band"]==band and r["floor"]==1.e-4)
        a,b,t=(row[k] for k in ("legacy_match","corrected_match","reference"))
        y=len(selected)-i-1
        bx.plot([0,t],[y,y],color="#e0e5e9",lw=2,zorder=0)
        bx.scatter(t,y,facecolor="white",edgecolor=grey,s=46,lw=1.5,
                   label="Direct reference" if i==0 else None,zorder=2)
        bx.scatter(a,y+.082,color=blue,marker="s",s=30,label="Legacy" if i==0 else None,zorder=3)
        bx.scatter(b,y-.082,color=orange,marker="o",s=34,label="FFT bins" if i==0 else None,zorder=3)
    bx.set(yticks=[3,2,1,0],yticklabels=labels,xlabel="Reference-confirmed detector spots",
           xlim=(-1,76),ylim=(-.4,3.4))
    bx.set_title("Matched spots in each energy’s high-q band",size=11.5,weight="semibold")
    bx.legend(frameon=False,fontsize=8.6,ncol=3,loc="upper left",bbox_to_anchor=(0.,-.17),
              borderaxespad=0,handletextpad=.3,columnspacing=.8)
    for a in (ax,bx):
        a.spines[["top","right"]].set_visible(False)
        a.tick_params(labelsize=9)
    fig.text(.084,.028,"Detection floor 0.0001 × the direct (1, 0, 0) peak. Bands in transverse q: 0.95–1.25 nm⁻¹ at 285 eV and 1.90–2.40 nm⁻¹ at 570 eV.",
             fontsize=9,color="#4d5660")
    for ext in ("png","pdf"):
        fig.savefig(out/f"NRSS_Bragg_High_q_Registration.{ext}",dpi=180,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference",type=Path,help="CPU root with both cases and both energies")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--legacy",type=Path,help="Public outputs from one path on the historical revision")
    p.add_argument("--corrected",type=Path,help="Public outputs from that path on the corrected revision")
    args=p.parse_args()
    if bool(args.legacy)!=bool(args.corrected):
        p.error("supply both --legacy and --corrected or neither")
    args.out.mkdir(parents=True,exist_ok=True)
    rows,spots=count_rows(args.reference,args.legacy,args.corrected)
    for filename,table in (("bragg_summary.csv",rows),("bragg_matched_spots.csv",spots)):
        with (args.out/filename).open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    if args.legacy is None:
        plot(args.reference,rows,args.out)
