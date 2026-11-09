RFM NS v6 | Full-annulus cylinder wake

Extract the archive into a new directory. Run commands from that directory
to preserve the relative work/ layout. MANIFEST.json records the byte count
and SHA-256 digest of every member. Do not overwrite the frozen evidence.
The same script can independently check archive CRCs and hashes:

  python work/rfm_v6/package_v6.py --verify RFM_NS_v6_复现实验.zip

Problem and numerical conventions

Re=100, cylinder diameter D=1, inner radius 0.5, outer radius 6 and
kinematic viscosity 0.01. The flow starts at rest. The outer normal velocity
is ramped, and an additional smooth transverse pulse is present only for
4<t<12. There is no reflection or circulation constraint. The finite circular
outer boundary prescribes normal velocity and zero vorticity. Comparisons
use identical forcing, common output times and the full spatial fields.
The modal coefficients use the normalized real Fourier convention
f(theta)=c0+2 Re(sum_{k>0} c_k exp(i k theta)).

The streamfunction/velocity solves use both real and imaginary modal
amplitudes for k>0. The angular mean v_theta is evolved by mean tangential
momentum, with zero cylinder velocity and zero outer vorticity. This
preserves the circulation degree of freedom. The Galerkin method uses a
same-space radial test basis and SPD mass/diffusion operators. Legendre
uses the same Galerkin/time/angle implementation. FD4 independently
discretizes the radial operators and advection.

Environment and reproduction

Use Python 3.13, NumPy, SciPy and the same Windows platform for comparable
timings. Report generation additionally uses ReportLab and the fonts
registered by work/report_v2/layout.py. Set BLAS threads to one before
running any numerical command. The formal run additionally pins one
logical processor and randomizes/interleaves candidate order.

  $env:OPENBLAS_NUM_THREADS='1'
  $env:OMP_NUM_THREADS='1'
  $env:MKL_NUM_THREADS='1'
  python work/rfm_v6/full_nonlinear_audit.py --timing
  python work/rfm_v6/full_fft_buffers.py
  python work/rfm_v6/full_projected_flux.py

The three optional kernel audits write local JSON output. Their timings
are pilot observations and do not substitute for the formal end-to-end
solve comparisons.

Reference files are supplied as fd_full_*.npz. From the extraction root,
regenerate the two formal reference arrays at new output locations:

  mkdir replay
  python work/rfm_v6/fd_full_reference.py --n 240 --k 120 --rate 320 --grid outer --stretch 1 --output replay/reference.npz
  python work/rfm_v6/fd_full_reference.py --n 200 --k 144 --rate 256 --grid outer --stretch 1 --output replay/predecessor.npz
  python work/rfm_v6/fd_full_reference.py --compare replay/predecessor.npz replay/reference.npz

The fd_full_band.dll binary is the native FD band-solve accelerator. The
source is included. From the workspace root a MinGW-w64 build is:

  x86_64-w64-mingw32-gcc -std=c99 -O3 -shared -nostdlib -nodefaultlibs "-Wl,--entry,0" -o work/rfm_v6/fd_full_band.dll work/rfm_v6/fd_full_band.c

A Linux shared object can be built with gcc -std=c99 -O3 -shared -fPIC;
use work/rfm_v6/fd_full_band.so. The FD class also has a portable SciPy
LAPACK fallback via linear='scipy', which is a separate timing backend.

The full_specs.json file declares the formal candidates. Run a fresh
replay from the extraction root:

  python work/rfm_v6/full_formal.py --reference work/rfm_v6/fd_full_n240_k120_rate320_outer1.npz --predecessor work/rfm_v6/fd_full_n200_k144_rate256_outer1.npz --output replay/full_formal.json

The wake figure uses the predecessor's cold-start probe record and
diagnostic-only dense snapshots from its t=54 restart:

  python work/rfm_v6/diagnostic_figure.py work/rfm_v6/fd_full_dense_from54.npz --history work/rfm_v6/fd_full_n200_k144_rate256_outer1.npz --plot replay/wake.png --metrics replay/wake.json

Accuracy and timing evidence are stored in full_formal.json, with
solution checkpoints in NPZ and wake metrics in diagnostic evidence.
Gates apply to the worst relative L2 errors for velocity and vorticity,
including near-cylinder weights, three independent score grids, all
saved output times and all required seeds. A passing row also needs
persistent shedding and reference refinement change <=25% of the
gate. Phase alignment is a physical diagnostic only and does not enter
the acceptance gate. Different annular radii and outer conditions can
change the wake frequency and elapsed cost.

Formal stepping is fixed SBDF2 with the same t=0 rest state and
no warm start for any timed configuration. If replaying the separate
kernel, phase, or energy audits, do so after the complete timed matrix
to avoid concurrent load. Use build_v6_report.py to rebuild the PDF
from the frozen comparison and supplied wake figure.
