# Second-review audit: FD4 boundary closure and mapped-operator scope

This directory contains a read-only algebraic/spatial audit, without any new PDE march, GPU run, or performance result. Frozen source files are unchanged.

Run from the repository root:

```powershell
python work/rfm_review_v4/fd_geometry/verify_fd_closure.py
```

`fd_closure_audit.json` records exact rational moments, the first omitted Taylor coefficient, four actual annular grids, a constructed FD184 wall operator check, and hashes of final cohort/reference/source inputs. `fd_closure_zh.tex` and `fd_closure_en.tex` supply a bilingual baseline subsection.

The review's displayed constant coefficients are correct only for a uniform coordinate grid with zero wall streamfunction. A nonzero wall value needs the term `-415/72 * psi(0) / h**2`. The inner-wall slope term is `-25/6 * psi_s(0) / h`. Exact polynomial moments through degree five give `D_h psi - psi_ss(0) = -h**4 * psi^(6)(0)/15 + O(h**5)`. The final annulus instead uses nonuniform logarithmic nodes `s_j=log(12)*sin(pi*j/(2*n))` and recomputes Hermite weights from the actual nodes. The outer-wall vorticity is prescribed as zero. The mean tangential-velocity block is separate. The old Cartesian cavity file is not the final annular baseline.

`geometry_extension_{zh,en}.tex` supplements the existing geometric-scope section. The spectral-equivalence inequality is an assumption, not a consequence established here for every smooth mapping. Pure radial reparameterization preserves angularly invariant implicit blocks, whereas angle-dependent metrics generally couple modes. Conditional PCG convergence requires SPD operators and a fixed SPD preconditioner. An FFT-based matrix--vector product is an implementation option, not a universal requirement.

Primary reference checked on 2026-11-10: [Netlib author-hosted PCG convergence](https://netlib.org/linalg/html_templates/node22.html), with the [book's own bibliographic entry](https://netlib.org/linalg/html_templates/Templates.html). The proposed `r21` entry is in `pcg_reference.bib`.
