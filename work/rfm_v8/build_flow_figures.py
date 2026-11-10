"""Export accepted finite-annulus fields and unshifted probe histories."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
sys.path.insert(0,str(HERE))
from galerkin_minimal_model import RecurrenceGalerkin, case, np
from fd_full import FullFD
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

OUT = WORK.parent / "outputs" / "rfm_v8_figures"
SOURCES = {
    "RFM41, rank 40": HERE / "thin_t72_qualification_output" / "trajectory_015.npz",
    "Legendre36, rank 40": HERE / "legendre_rank_switch_output" / "leg36_k80_q76.npz",
    "FD176": HERE / "thin_t72_qualification_output" / "trajectory_002.npz",
    "FD448 reference": WORK / "rfm_v7/cpu_formal_point1_pow2_chain_output/fd448_k144_rate512_pow2.npz",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"figure.facecolor":"white","savefig.facecolor":"white"})
    arrays = {}
    for label,path in SOURCES.items():
        with np.load(path,allow_pickle=False) as raw:
            arrays[label] = {name:raw[name].copy() for name in ("states","times","probe_times","transverse","vorticity")}
            arrays[label]["config"] = json.loads(str(raw["config"]))
            arrays[label]["metadata"] = json.loads(str(raw["metadata"]))
    rfm = arrays["RFM41, rank 40"]
    model = RecurrenceGalerkin(dt=rfm["metadata"]["dt"], **rfm["config"])
    if model.width != 40:
        raise ArithmeticError("Representative rank differs from its independently verified source")
    # The radial log grid is for visualization only. Qualification uses the
    # three separate frozen evaluation grids, never these plotting pixels.
    radial_edges = np.geomspace(case.A,case.R,226)
    angular_edges = np.linspace(0.,2*np.pi,601)
    radius = np.sqrt(radial_edges[:-1]*radial_edges[1:])
    theta = .5*(angular_edges[:-1]+angular_edges[1:])
    s = np.log(radius/case.A)
    x,y = radial_edges[:,None]*np.cos(angular_edges),radial_edges[:,None]*np.sin(angular_edges)
    fig,axes = plt.subplots(1,3,figsize=(12,4.1),layout="constrained")
    for axis,time in zip(axes,(24.,48.,72.),strict=True):
        slot = np.flatnonzero(rfm["times"]==time)
        if len(slot)!=1:
            raise ArithmeticError("Expected exact physical snapshot time")
        omega = model.modal(rfm["states"][slot[0]],case.speed(time),s)[2]
        field = case.reconstruct(omega,theta)
        mesh = axis.pcolormesh(x,y,field,cmap="RdBu_r",vmin=-5,vmax=5,shading="flat",rasterized=True)
        axis.add_patch(Circle((0,0),case.A,facecolor="#303c48",edgecolor="white",lw=.8,zorder=5))
        axis.set(xlim=(-1.25,5.8),ylim=(-3.0,3.0),aspect="equal",xlabel="x / D",title=f"t = {time:g}")
        axis.set_ylabel("y / D")
        axis.grid(alpha=.12)
    fig.colorbar(mesh,ax=axes,label="Vorticity  ω D / U",shrink=.79,pad=.02)
    fig.suptitle("Re = 100: unconstrained alternating wake in a finite annulus",fontsize=13)
    fig.savefig(OUT/"finite_annulus_vorticity.png",dpi=240)
    plt.close(fig)
    colors={"RFM41, rank 40":"#147d92","Legendre36, rank 40":"#c76b2c","FD176":"#7758a2","FD448 reference":"#202c37"}
    styles={"RFM41, rank 40":"-","Legendre36, rank 40":"--","FD176":":","FD448 reference":"-"}
    fig=plt.figure(figsize=(10,6.3),layout="constrained")
    grid=fig.add_gridspec(3,1,height_ratios=(1,1,.15))
    axes=[fig.add_subplot(grid[0]),fig.add_subplot(grid[1])]
    legend_axis=fig.add_subplot(grid[2])
    legend_axis.axis("off")
    for label in reversed(list(SOURCES)):
        value=arrays[label]
        if value["probe_times"].shape!=(577,) or not np.array_equal(value["probe_times"],np.arange(577)*.125):
            raise ArithmeticError("Probe histories are not aligned at their original absolute times")
        for axis,key in zip(axes,("transverse","vorticity"),strict=True):
            axis.plot(value["probe_times"],value[key][:,1],label=label,color=colors[label],
                      linestyle=styles[label],lw=1.1 if "reference" not in label else 2.,alpha=.88)
    axes[0].axvspan(4,12,color="#c5cdd5",alpha=.4,label="Prescribed transverse trigger")
    axes[0].set(xlim=(0,72),ylabel="Tangential velocity  uθ / U",title="Original absolute times; no phase alignment")
    axes[1].set(xlim=(24,72),ylabel="Vorticity  ω D / U",xlabel="Time  t U / D",title="Sustained late-time shedding after the trigger")
    for axis in axes:
        axis.grid(alpha=.2)
        axis.axhline(0,color="#63717c",lw=.6,alpha=.4)
    handles,labels=axes[0].get_legend_handles_labels()
    legend_axis.legend(handles,labels,ncols=3,loc="center",fontsize=8,frameon=False)
    fig.suptitle("Independent probe at (x, y) = (2.5D, 0)",fontsize=13)
    fig.savefig(OUT/"unshifted_wake_probes.png",dpi=240)
    plt.close(fig)
    report=dict(scope="Visualization of accepted physical trajectories, not a replacement for frozen numerical scoring",
                source_sha256={p.relative_to(WORK).as_posix():digest(p) for p in SOURCES.values()},
                script_sha256=digest(Path(__file__)),snapshot_times=[24,48,72],probe_position=[2.5,0],
                plot_grid=dict(radial=225,angular=600,explicit_polar_cell_edges=True),vorticity_display_clip=[-5,5],
                finite_outer_radius=6.,cylinder_radius=.5,phase_alignment=False,
                outputs={p.name:digest(p) for p in OUT.glob("*.png")})
    (OUT/"provenance.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report["outputs"]))


if __name__ == "__main__":
    main()
