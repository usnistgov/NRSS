"""Plot the replay products and the even-grid DC example."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np

legacy, fixed, meas, band, hint = "#21649a", "#bd4c16", "#30353b", "#e6ecf1", "#633876"


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out/f"{name}.{ext}", dpi=175 if ext=="png" else None, facecolor="white")
    plt.close(fig)


def core_plot(curves, out):
    fig, axes=plt.subplots(3,3,figsize=(12.5,8.35),sharex="col",
                            gridspec_kw={"height_ratios":[1.65,1,1]})
    fig.subplots_adjust(left=.087,right=.984,bottom=.115,top=.812,hspace=.25,wspace=.32)
    fig.text(.087,.96,"CoreShell anisotropy with legacy and FFT-bin coordinates",fontsize=19,weight="semibold")
    fig.text(.087,.92,"CPU replay of the unpadded 32 × 512 × 512 fixture • 101 energies • historical WPIntegrator wedge",fontsize=10.5,color="#43505c")
    handles=[Line2D([],[],ls="--",lw=2,color=legacy,label="Legacy coordinates"),
             Line2D([],[],lw=2,color=fixed,label="FFT-bin coordinates"),
             Line2D([],[],ls="",marker="o",ms=4,color=meas,label="Experimental reference"),
             Patch(color=band,label="Current maximum-error band")]
    fig.legend(handles=handles,ncol=4,loc="upper left",bbox_to_anchor=(.08,.906),frameon=False,fontsize=10)
    for col,e in enumerate((None,284.7,285.2)):
        x=curves["energy"] if e is None else curves["q"]
        fn=(lambda v:v.mean(axis=1)) if e is None else (lambda v:v[np.argmin(abs(curves["energy"]-e))])
        a,b,c,g=[fn(curves[k]) for k in ("legacy","corrected","measured","saved_sim")]
        lim=(.03,.05,.055)[col]
        sim_lim=(.0025,.008,.008)[col]
        top,mid,bottom=axes[:,col]
        top.set_title(["Mean A, 0.02–0.40 nm⁻¹","A(q) at 284.7 eV","A(q) at 285.2 eV"][col],fontweight="semibold",fontsize=12,pad=8)
        top.plot(x,c,"o",ms=2.5,color=meas,alpha=.78,zorder=1)
        top.plot(x,b,c=fixed,lw=2,zorder=2)
        top.plot(x,a,c=legacy,ls=(0,(4,2.6)),lw=1.7,zorder=3)
        mid.axhspan(-lim,lim,color=band)
        mid.plot(x,b-c,c=fixed,lw=1.8)
        mid.plot(x,a-c,c=legacy,ls="--",lw=1.5)
        bottom.axhspan(-sim_lim,sim_lim,color=band)
        bottom.plot(x,b-g,c=fixed,lw=2)
        for ax,threshold in ((mid,lim),(bottom,sim_lim)):
            ax.axhline(0,lw=.65,c="#879098")
            for sign in (1,-1):
                ax.axhline(sign*threshold,lw=.65,c="#a7b0b8",ls=":")
        outside=abs(b-c)>lim
        if outside.any():
            mid.scatter(x[outside],(b-c)[outside],s=14,fc="white",ec=fixed,lw=1.1,zorder=5)
        for ax in (top,mid,bottom):
            ax.spines[["top","right"]].set_visible(False)
            ax.margins(x=.015,y=.15)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.tick_params(labelsize=9)
        if col:
            bottom.set_xlabel(r"$q_\perp$ (nm$^{-1}$)")
        else:
            bottom.set_xlabel("Photon energy (eV)")
    axes[0,0].set_ylabel("Anisotropy A")
    axes[1,0].set_ylabel("Simulation\n− experiment")
    axes[2,0].set_ylabel("Corrected\n− sim golden")
    fig.text(.087,.029,"Shading gives the existing maximum absolute error limits for each reference. Open circles mark the four experimental exceedances.",fontsize=9,color="#43505c")
    save(fig,out,"NRSS_CoreShell_Registration")


def dc_plot(out):
    fig,(ax,ay)=plt.subplots(1,2,figsize=(11.6,4.8),gridspec_kw={"width_ratios":[1,1.15]})
    fig.subplots_adjust(left=.095,right=.98,bottom=.2,top=.77,wspace=.30)
    fig.text(.095,.94,"Where the DC plane lands on an even FFT",fontsize=18,weight="semibold")
    fig.text(.095,.865,"The origin test and the off-axis grating exercise different coefficients.",fontsize=10.5,color="#43505c")
    n=8; index=np.arange(n)
    for row,(axis,color) in enumerate(((index-n//2, fixed),((index-(n-1)/2)*n/(n-1),legacy))):
        ax.hlines(row,axis[0],axis[-1],lw=1,colors="#bbc3cb")
        ax.scatter(axis,np.full(n,row),fc="white",ec=color,lw=1.7,s=35,zorder=2)
        ax.scatter(axis[n//2],row,c=color,marker="D",s=52,zorder=3)
    ax.axvline(0,c="#454b50",ls=":",lw=1.2)
    ax.set_title("Eight samples, one native DC coefficient",fontsize=11.5,fontweight="semibold")
    ax.set_xlim(-4.55,4.45);ax.set_ylim(-.6,1.8);ax.set_yticks([0,1],["FFT bins","Legacy"])
    ax.set_xticks(np.arange(-4,5));ax.set_xlabel(r"Assigned $q$ in units of $2\pi/(Nd)$")
    ax.annotate("DC, index 4",xy=(0,0),xytext=(.7,-.34),fontsize=9,arrowprops={"arrowstyle":"-","lw":.8})
    ax.annotate("DC, index 4",xy=(4/7,1),xytext=(1.17,1.4),fontsize=9,arrowprops={"arrowstyle":"-","lw":.8})
    nz,ny=128,1024
    s=np.linspace(.08,2.5,720)
    dy=2*np.pi*64/ny; dz=2*np.pi/nz
    k=(dy**2+(s*dz)**2)/(2*s*dz)
    old_y=-np.pi+(ny//2+64)*2*np.pi/(ny-1)
    old_z=-old_y**2/(k+np.sqrt(k*k-old_y**2))
    t=(old_z+np.pi)/(2*np.pi/(nz-1))
    plane=nz//2-1
    old=np.maximum(1-abs(t-plane),0.)
    new=np.maximum(1-abs(s-1),0.)
    exact=(np.sinc(s-1)/np.sinc((s-1)/nz))**2
    ay.plot(s,exact,ls=":",lw=1.7,c=meas,label="Direct finite grating")
    ay.plot(s,new,lw=2.2,c=fixed,label="FFT-bin interpolation")
    ay.plot(s,old,ls="--",lw=2,c=legacy,label="Legacy interpolation")
    ay.axvline(1,c="#9aa2a9",lw=.8);ay.scatter([1,1],[1,.489994],c=[fixed,legacy],zorder=4,s=26)
    ay.set(xlim=(s[0],s[-1]),ylim=(-.04,1.10),xlabel=r"Physical Ewald depth $-q_z/\Delta q_z$",ylabel="Normalized peak intensity")
    ay.set_title("Off-axis grating with a peak at depth 1",fontsize=11.5,fontweight="semibold")
    ay.legend(loc="upper right",frameon=False,fontsize=8.7)
    for axis in (ax,ay):
        axis.spines[["top","right"]].set_visible(False)
    fig.text(.095,.055,"Diamonds locate native FFT index zero after Igor reordering. The grating uses a separate lateral column, 64 bins from the beam.",fontsize=9,color="#43505c")
    save(fig,out,"NRSS_DC_and_Grating")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("curves",type=Path);p.add_argument("--out",type=Path,default=Path.cwd());args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    with np.load(args.curves) as curves:
        core_plot(curves,args.out)
    dc_plot(args.out)
