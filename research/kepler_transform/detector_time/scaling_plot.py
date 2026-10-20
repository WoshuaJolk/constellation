"""Plot larger-query campaigns separately for certified simulation and real pixels."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("synthetic", type=Path)
    parser.add_argument("real", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    paths = [args.synthetic, args.real]
    reports = [json.loads(p.read_text()) for p in paths]
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
                         "axes.spines.top":False, "axes.spines.right":False,
                         "text.color":"#172033", "axes.labelcolor":"#334155",
                         "svg.hashsalt":"fkt-larger-orbit-counts"})
    fig, axes = plt.subplots(2,2,figsize=(13,10))
    fig.subplots_adjust(left=.075,right=.95,bottom=.19,top=.78,hspace=.75,wspace=.22)
    fig.suptitle("Does the flatter runtime curve hold?",x=.075,y=.97,
                 ha="left",fontsize=23,weight="bold")
    fig.text(.075,.925,"Measured end-to-end cost through 65,536 orbit queries · lower is better",
             fontsize=13,color="#475569")
    record = {"sources":[], "series":[]}
    for row,(report,path) in enumerate(zip(reports,paths)):
        real = report["kind"] == "real"
        label = "Fourier adapter (no time compression)" if real else "Partial FKT (time compressed)"
        color = "#C36216" if real else "#7C3AED"
        title = ("REAL PIXELS · 3 ZTF exposures over 12.3 minutes" if real else
                 "CONTROLLED STACK · 1,024 simulated 33×33 exposures over 8 days")
        fig.text(.075,.85 if row==0 else .48,title,fontsize=12,weight="bold")
        rows = report["measurements"]
        x = [r["queries"] for r in rows]
        row_record = {"kind":report["kind"],"queries":x,"seconds":{}}
        for col,prefix in enumerate(["first","new_image"]):
            ax = axes[row,col]
            ax.set_title("First use, including preparation" if col==0 else
                         "New image values, reusing preparation",loc="left",pad=12,fontsize=12)
            for method,name,c in [("compressed",label,color),
                                  ("direct","Best measured direct pixels","#087F8C")]:
                key = f"{prefix}_{method}_seconds"
                y = [r["comparison"][key] for r in rows]
                row_record["seconds"][key] = y
                lows, highs = [], []
                for r in rows:
                    if method == "compressed":
                        fixed = report["setup_seconds"] + r["fast_query_setup_seconds"] if prefix=="first" else 0.
                        pieces = [r["fast_both"]] if prefix=="first" else [
                            report["changed_image_preparation"], r["fast_numerator"]]
                    else:
                        chosen = r["comparison"][f"{prefix}_direct_method"]
                        if prefix=="first" and chosen=="stream":
                            fixed, pieces = r["geometry_seconds"], [r["direct_stream"]]
                        elif chosen=="dense":
                            fixed = r["geometry_seconds"]+r["dense_cache"]["setup_seconds"] if prefix=="first" else 0.
                            pieces = [r["dense_cache"]["application"]]
                        else:
                            fixed = 0.
                            pieces = [r["partial_cache"]["application"]]
                            if prefix=="first":
                                fixed = r["geometry_seconds"]+r["partial_cache"]["setup_seconds"]
                                pieces.append(r["direct_stream"])
                    lows.append(fixed+sum(p["min_seconds"] for p in pieces))
                    highs.append(fixed+sum(p["max_seconds"] for p in pieces))
                ax.fill_between(x,lows,highs,color=c,alpha=.12,linewidth=0)
                ax.plot(x,y,"o-",color=c,label=name,linewidth=2.1,markersize=5)
                ax.annotate(f"{y[-1]:.3g} s",(x[-1],y[-1]),xytext=(-5,10),
                            textcoords="offset points",ha="right",fontsize=10,color=c,weight="bold")
            ax.set_xlim(0,max(x)*1.06)
            ax.set_ylim(bottom=0,top=max(
                row_record["seconds"][f"{prefix}_direct_seconds"]+
                row_record["seconds"][f"{prefix}_compressed_seconds"])*1.22)
            ax.set_xticks([0,16384,32768,49152,65536])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f"{v:,.0f}"))
            ax.tick_params(axis="x",labelsize=9)
            ax.set_xlabel("Number of orbit queries")
            ax.set_ylabel("Compute time (seconds)")
            ax.grid(axis="y",color="#E2E8F0")
            ax.set_axisbelow(True)
        axes[row,0].legend(loc="upper left",frameon=False,fontsize=9)
        record["series"].append(row_record)
        record["sources"].append({"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    fig.text(.075,.065,
             "Points use measured medians; shading spans application samples (not confidence intervals). Setup measured once. Panel scales differ.\n"
             "Direct methods retain up to 3 GB of cache, recomputing uncached PSF factors. Both routes reuse information on new images.\n"
             "The real-data row tests the Fourier image adapter only. It does not validate time compression on real ephemerides.",
             fontsize=10,color="#475569",linespacing=1.7)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for extension in ["png","svg","pdf"]:
        fig.savefig(args.output.with_suffix("."+extension),dpi=180,facecolor="white")
    plt.close(fig)
    args.output.with_suffix(".json").write_text(json.dumps(record,indent=2)+"\n")


if __name__=="__main__":
    main()
