"""Matched-time radial error profiles and angular error spectra from frozen CPU controls.

The CPU controls are those hash-bound to the accepted GPU qualification. No new
reference solution, phase alignment, field fitting, or time integration occurs.
Run: python work/rfm_review_v3/profiles_bridge/analyze_profiles.py
"""
from review_common import *
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TARGETS=(('rfm41_k80','combined'),('leg36_k80_q76','combined'),('fd184_k80','fd'),
         ('rfm41_k80','uniform'),('leg36_k80_q76','uniform'))
LABELS={'rfm41_k80':'RFM41 (rank 40)','leg36_k80_q76':'Legendre36 (rank 40)','fd184_k80':'FD184'}

def csv_write(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def padded(values, k):
    out=[]
    for value in values:
        a=np.zeros((len(value),k+1),dtype=complex); a[:,:value.shape[1]]=value
        a[:,0]=a[:,0].real
        out.append(a)
    return out

def main():
    refs=references(); grids=base.grid_set(); s,theta,w=grids['layer144']; r=case.radius(s)
    allradial=[]; allmodal=[]; summary=[]; globalscores=[]; curves={}
    # Modal Parseval sums are exact in theta for these finite spectra, independent
    # of the 457-point layer scoring grid. The radial quadrature is unchanged.
    for name,arm in TARGETS:
        sol=control(name,arm)
        for refname,ref in refs.items():
            for ti,t in enumerate(TIMES):
                reference=padded(ref.modal(ti,s),144); candidate=padded(sol.modal(ti,s),144)
                mul=np.ones(145);mul[1:]=2
                for field,indices in (('velocity',(0,1)),('vorticity',(2,))):
                    e=sum(np.abs(candidate[i]-reference[i])**2 for i in indices)*mul
                    norm=sum(np.abs(reference[i])**2 for i in indices)*mul
                    radial=e.sum(axis=1); spectral=w@e; denom=float(w@norm.sum(axis=1))
                    energy=float(w@radial);globalrms=np.sqrt(denom/w.sum())
                    curve=np.sqrt(radial)/globalrms
                    modal=np.sqrt(spectral/denom)
                    prefix=dict(name=name,arm=arm,reference=refname,time=t,field=field)
                    for i,rr in enumerate(r):
                        allradial.append(dict(**prefix,radius=float(rr),log_radius=float(s[i]),radial_weight=float(w[i]),
                            local_angular_rms_error=float(np.sqrt(radial[i])),
                            rms_error_over_global_reference_rms=float(curve[i]),
                            contribution_to_squared_relative_error=float(w[i]*radial[i]/denom)))
                    for k,val in enumerate(modal):
                        allmodal.append(dict(**prefix,mode=k,sqrt_relative_error_energy=float(val),
                            contribution_to_squared_relative_error=float(spectral[k]/denom)))
                    summary.append(dict(**prefix,relative_error=float(np.sqrt(energy/denom)),
                        peak_profile_radius=float(r[np.argmax(curve)]),
                        fraction_error_energy_r_le_1=float((w*radial)[r<=1].sum()/energy),
                        fraction_error_energy_r_le_2p5=float((w*radial)[r<=2.5].sum()/energy),
                        fraction_error_energy_r_ge_4=float((w*radial)[r>=4].sum()/energy),
                        fraction_error_energy_r_ge_5p9=float((w*radial)[r>=5.9].sum()/energy),
                        fraction_error_energy_modes_0_to_8=float(spectral[:9].sum()/energy),
                        fraction_error_energy_modes_9_to_24=float(spectral[9:25].sum()/energy),
                        fraction_error_energy_modes_25_to_80=float(spectral[25:81].sum()/energy),
                        fraction_error_energy_modes_above_80=float(spectral[81:].sum()/energy)))
                    curves[(name,arm,refname,t,field)]=(curve,modal)
            # The max table is calculated using the original three grids, not the
            # profile-only grid. It records argmax separately for each field.
            scores=case.score(sol,case.reference_fields(ref,grids),grids)
            globalscores.extend(dict(name=name,arm=arm,reference=refname,**row) for row in scores)
        print('PROFILE',name,arm,flush=True)
    csv_write(HERE/'radial_profiles.csv',allradial)
    csv_write(HERE/'angular_error_spectra.csv',allmodal)
    csv_write(HERE/'profile_energy_summary.csv',summary)
    csv_write(HERE/'physical_errors_all_times.csv',globalscores)
    extrema=[]
    for name,arm in TARGETS:
        rows=[row for row in globalscores if (row['name'],row['arm'])==(name,arm)]
        extrema.append(dict(name=name,arm=arm,metrics={metric:{key:max(rows,key=lambda x:x[metric])[key]
            for key in ('reference','grid','time',metric)} for metric in METRICS}))
    inputs={f'{name}:{arm}':job(name,arm)['control'] for name,arm in TARGETS}
    write(HERE/'profiles_summary.json',dict(status='complete',physical_times=TIMES,targets=TARGETS,
        reference_source='Existing hash-bound angular_fine and time_fine FD448/K144 trajectories, rates384/512; no new references',
        reference_inputs=FROZEN['early_references']|{'late':FROZEN['late_reference']['reference_hashes']},
        inputs=inputs,source_sha256={p.name:sha(p) for p in (Path(__file__),HERE/'review_common.py')},
        profile_grid='layer144 radial quadrature, exact angular Parseval sums',
        definition='Radial curve is local angular RMS error/global domain RMS of reference; angular curve is sqrt(mode error energy/reference total energy). Velocity uses polar components (ur,utheta): integrated vector-error energy is invariant under the orthogonal Cartesian transformation, but modal allocation depends on representation. Squares integrate/sum to field relative error squared. Neither divides by a local vanishing velocity or vorticity.',
        argmax=extrema,energy_summary=summary,plots_reference='time_fine',
        interpretation_scope='Descriptive localization, not causal proof of layer-basis benefit; switching error, radial approximation, boundary treatment and reference error are not separately identified.'))
    make_plots(curves,r)

def make_plots(curves,r):
    # Native width 160 mm: the paper need not shrink the 9-point labels.
    plt.rcParams.update({'font.size':9,'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42})
    out=HERE/'figures';out.mkdir(exist_ok=True)
    for kind in ('radial','angular'):
        fig,axs=plt.subplots(4,2,figsize=(6.3,7.7),sharex=True,sharey='col',layout='constrained')
        for row,t in enumerate((8,24,48,72)):
            for col,field in enumerate(('velocity','vorticity')):
                ax=axs[row,col]
                for (name,arm),color in zip(TARGETS[:3],('#0072B2','#D55E00','#009E73')):
                    a,b=curves[(name,arm,'time_fine',t,field)]
                    if kind=='radial':
                        ax.semilogy(r,np.maximum(100*a,1e-8),color=color,label=LABELS[name],linewidth=1.25)
                    else:
                        ax.semilogy(np.arange(145),np.maximum(100*b,1e-9),color=color,label=LABELS[name],linewidth=1.25)
                ax.set_title(f'{field.capitalize()}, t = {t}')
                ax.grid(alpha=.25)
                if row==3: ax.set_xlabel('Radius r / D' if kind=='radial' else 'Angular mode k')
                if kind=='radial': ax.set_xlim(.5,6)
                else: ax.set_xlim(0,144)
        fig.supylabel('Local RMS / reference RMS (%)' if kind=='radial' else 'Mode error contribution (%)',fontsize=9)
        handles,labels=axs[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside upper center',ncol=3,frameon=False)
        for ext in ('pdf','png'):
            fig.savefig(out/f'review_{kind}_errors.{ext}',dpi=220)
        plt.close(fig)

if __name__=='__main__': main()
