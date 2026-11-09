"""Post-pulse wake probes, sign reversals and Fourier reflection diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


def _series(value, time):
    value = np.asarray(value, dtype=float)
    if value.ndim == 1:
        value = value[:, None]
    if value.ndim != 2 or len(value) != len(time) or not np.isfinite(value).all():
        raise ValueError('Expected finite time-major probe arrays')
    return value


def _spectral_peak(time, value):
    """Approximate the dominant positive frequency on a uniform grid."""
    dt = float(np.median(np.diff(time)))
    if np.max(np.abs(np.diff(time)-dt)) > 1e-6*dt:
        return None
    signal = (value-np.mean(value))*np.hanning(len(value))
    power = np.abs(np.fft.rfft(signal))**2
    if len(power) < 4 or max(power[1:]) == 0:
        return None
    k = int(np.argmax(power[1:])+1)
    offset = 0.
    if k+1 < len(power):
        a,b,c = np.log(np.maximum(power[k-1:k+2], np.finfo(float).tiny))
        curvature = a-2*b+c
        if curvature < 0:
            offset = np.clip(.5*(a-c)/curvature,-.5,.5)
    return float((k+offset)/(dt*len(time)))


def _cycles(time, raw, amplitude_floor):
    centre = float(np.median(raw))
    value = raw-centre
    swing = float((np.quantile(raw,.95)-np.quantile(raw,.05))/2)
    idx = np.flatnonzero((value[:-1] < 0.) & (value[1:] >= 0.))
    cross = time[idx] - value[idx]*np.diff(time)[idx]/np.diff(value)[idx]
    periods = np.diff(cross)
    amplitudes = []
    for left,right in zip(cross[:-1],cross[1:]):
        between = raw[(time>=left)&(time<=right)]
        if len(between)>1:
            amplitudes.append(float(np.ptp(between)/2))
    valid = len(amplitudes)==len(periods) and len(periods)>=3
    period = float(np.median(periods)) if valid else None
    cv = float(np.std(periods)/np.mean(periods)) if valid else None
    growth = float(np.median(amplitudes[-2:])/np.median(amplitudes[:2])) if valid and min(amplitudes)>0 else None
    minimum,maximum = float(raw.min()),float(raw.max())
    both_signs = minimum <= -amplitude_floor/4 and maximum >= amplitude_floor/4
    prominence = max(.3*swing,amplitude_floor/4)
    highs = find_peaks(raw,prominence=prominence)[0]
    lows = find_peaks(-raw,prominence=prominence)[0]
    extrema = sorted([(i,1) for i in highs]+[(i,-1) for i in lows])
    alternations = sum(a!=b for (_,a),(_,b) in zip(extrema,extrema[1:]))
    persistent = bool(valid and both_signs and min(amplitudes)>=amplitude_floor
                      and cv<=.10 and growth is not None and growth>=.80 and alternations>=5)
    saturated = bool(persistent and .90<=growth<=1.10)
    return dict(centre=centre,minimum=minimum,maximum=maximum,positive_and_negative=both_signs,
        robust_half_swing=swing,positive_crossing_times=cross.tolist(),periods=periods.tolist(),
        complete_cycles=len(periods),median_period=period,period_cv=cv,cycle_half_swing=amplitudes,
        late_over_early_amplitude=growth,peak_trough_alternations=alternations,
        spectral_frequency=_spectral_peak(time,raw),persistent=persistent,saturated=saturated)


def _phase_lags(time, values, period):
    if period is None:
        return [None]*(values.shape[1]-1)
    angle=2*np.pi*(time-time[0])/period
    matrix=np.column_stack((np.ones_like(angle),np.cos(angle),np.sin(angle)))
    coeff=np.linalg.lstsq(matrix,values,rcond=None)[0]
    phase=coeff[1]-1j*coeff[2]
    return [float(-np.angle(b*np.conj(a))*period/(2*np.pi)) if abs(a*b)>0. else None
            for a,b in zip(phase[:-1],phase[1:])]


def analyze(time, transverse, vorticity=None, *, x=(1.5,2.5,3.5), after=36., diameter=1., speed=1.,
            velocity_floor=.005, vorticity_floor=.02):
    """Summarize post-pulse oscillations without forcing phase alignment.

    Velocity and vorticity floors are in units of U and U/D respectively.
    The default diagnostic requires three complete stable-frequency cycles,
    positive/negative excursions, and no substantial decay at two probes.
    """
    time=np.asarray(time,dtype=float)
    if (time.ndim!=1 or len(time)<4 or not np.isfinite(time).all() or
        np.any(np.diff(time)<=0) or diameter<=0 or speed<=0 or velocity_floor<=0 or vorticity_floor<=0):
        raise ValueError('Expected increasing finite times and positive scales')
    probes=np.asarray(x,dtype=float)
    v=_series(transverse,time)
    w=None if vorticity is None else _series(vorticity,time)
    if (probes.shape!=(v.shape[1],) or not np.isfinite(probes).all()
        or np.any(np.diff(probes)<=0) or (w is not None and w.shape!=v.shape)):
        raise ValueError('Provide one increasing streamwise probe coordinate per column')
    mask=time>=after
    if np.count_nonzero(mask)<4:
        raise ValueError('Fewer than four samples in the diagnostic time window')
    t,vy=time[mask],v[mask]
    omega=None if w is None else w[mask]
    velocity_stats=[_cycles(t,col,velocity_floor*speed) for col in vy.T]
    vortex_stats=[] if omega is None else [_cycles(t,col,vorticity_floor*speed/diameter) for col in omega.T]
    periods=[r['median_period'] for r in velocity_stats if r['persistent']]
    period=float(np.median(periods)) if periods else None
    agreements=[]
    for j,(a,b) in enumerate(zip(velocity_stats,vortex_stats)):
        if a['persistent'] and b['persistent'] and period is not None:
            deviations=[abs(r['median_period']/period-1) for r in (a,b)]
            if max(deviations)<=.10:
                agreements.append(j)
    velocity_agrees=bool(len(periods)>=2 and max(abs(np.array(periods)/period-1))<=.10)
    shedding=bool(velocity_agrees and len(agreements)>=2)
    saturated=bool(shedding and all(velocity_stats[j]['saturated'] and vortex_stats[j]['saturated']
                                   for j in agreements))
    return dict(analysis_interval=[float(t[0]),float(t[-1])],diameter=float(diameter),speed=float(speed),
        sample_count=len(t),sample_gap_max=float(np.diff(t).max()),probe_x=probes.tolist(),
        velocity_floor=velocity_floor,vorticity_floor=vorticity_floor,velocity=velocity_stats,vorticity=vortex_stats,
        median_period=period,strouhal=None if period is None else float(diameter/(period*speed)),
        confirmed_probe_indices=agreements,cross_stream_periodic=velocity_agrees,passed=shedding,
        sustained_alternating_probe_signs=shedding,saturated_by_amplitude_test=saturated,
        velocity_downstream_lag=_phase_lags(t,vy,period),
        vorticity_downstream_lag=None if omega is None else _phase_lags(t,omega,period))


def centreline(ut, omega):
    """Evaluate the radial-first normalized rFFT coefficients at theta=0."""
    return tuple(z[...,0].real+2*np.sum(z[...,1:].real,axis=-1) for z in (ut,omega))


def reflection_breaking_modal(ur, ut, omega, weight):
    """Weighted physical symmetry defect from positive-frequency coefficients."""
    weight=np.asarray(weight,dtype=float)

    def square(z):
        return float(np.sum(weight*(abs(z[...,0])**2+2*np.sum(abs(z[...,1:])**2,axis=-1))))

    def odd(z):
        return float(2*np.sum(weight*np.sum(z[...,1:].imag**2,axis=-1)))

    def even(z):
        return float(np.sum(weight*(z[...,0].real**2+2*np.sum(z[...,1:].real**2,axis=-1))))

    def ratio(a,b):
        return float(np.sqrt(a/b)) if b else 0.

    return dict(velocity=ratio(odd(ur)+even(ut),square(ur)+square(ut)),
                vorticity=ratio(even(omega),square(omega)))


def spatial_cycles(time, omega, radius, theta, weight, period, *, diameter=1., wake_x=(1.5,4.5),
                   wake_abs_y=1.25):
    """Compare whole wake vorticity fields with a half-period reflection.

    The angular grid must be closed under theta -> -theta. Time interpolation
    is used only for this physical diagnostic, never as an accuracy gate.
    """
    time=np.asarray(time,dtype=float)
    omega=np.asarray(omega,dtype=float)
    radius=np.asarray(radius,dtype=float)
    theta=np.asarray(theta,dtype=float)
    if (time.ndim!=1 or len(time)<4 or np.any(np.diff(time)<=0) or period<=0
        or diameter<=0 or not np.isfinite(omega).all() or omega.shape!=(len(time),len(radius),len(theta))):
        raise ValueError('Expected positive period and time/radius/angle vorticity data')
    if time[-1]-time[0] < period or np.max(np.diff(time)) > period/8:
        raise ValueError('Spatial comparison needs a full period and at least eight snapshots per period')
    residual=np.abs(np.angle(np.exp(1j*(theta[:,None]+theta))))
    mirror=np.argmin(residual,axis=1)
    if np.max(residual[np.arange(len(theta)),mirror]) > 1e-10:
        raise ValueError('The angular grid is not closed under reflection')
    x=radius[:,None]*np.cos(theta)/diameter
    y=radius[:,None]*np.sin(theta)/diameter
    use=(x>=wake_x[0]) & (x<=wake_x[1]) & (abs(y)<=wake_abs_y)
    weight=np.broadcast_to(weight,use.shape)*use
    if np.any(weight<0) or weight.sum()==0:
        raise ValueError('Expected positive-area wake weights')

    def interp(t):
        j=int(np.searchsorted(time,t,side='right')-1)
        if j>=len(time)-1:
            return omega[-1]
        fraction=(t-time[j])/(time[j+1]-time[j])
        return omega[j]+fraction*(omega[j+1]-omega[j])

    side=np.where(abs(y)<1e-12,0.,np.sign(y))
    imbalance=np.sum(omega**2*(weight*side),axis=(1,2))/np.sum(omega**2*weight,axis=(1,2))
    rows=[]
    for i in np.flatnonzero(time<=time[-1]-period/2):
        initial=omega[i]
        half=interp(time[i]+period/2)
        scale=float(np.sqrt(np.sum(weight*initial**2)))
        if scale==0:
            continue

        def relative(error):
            return float(np.sqrt(np.sum(weight*error**2))/scale)

        row=dict(time=float(time[i]),half_reflected=relative(half+initial[:,mirror]),
                 half_unreflected=relative(half-initial))
        if time[i]+period<=time[-1]:
            row['whole_period']=relative(interp(time[i]+period)-initial)
        rows.append(row)
    if not rows or not np.isfinite(imbalance).all():
        raise ValueError('The wake is identically zero at one or more snapshots')
    median={key:float(np.median([r[key] for r in rows if key in r]))
            for key in ('half_reflected','half_unreflected','whole_period')}
    counts={key:sum(key in row for row in rows) for key in median}
    return dict(wake_x_over_d=list(wake_x),wake_abs_y_over_d=wake_abs_y,
                period=float(period),median_relative_vorticity_differences=median,comparison_counts=counts,fields=rows,
                times=time.tolist(),upper_minus_lower_enstrophy=imbalance.tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('history',type=Path,help='NPZ keys: time, transverse, vorticity, x')
    parser.add_argument('--after',type=float,default=36.)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    with np.load(args.history,allow_pickle=False) as data:
        result=analyze(data['time'],data['transverse'],data['vorticity'],x=data['x'],after=args.after)
    payload=json.dumps(result,indent=2,allow_nan=False)
    if args.output:
        args.output.write_text(payload+'\n',encoding='utf-8')
    else:
        print(payload)


if __name__=='__main__':
    main()
