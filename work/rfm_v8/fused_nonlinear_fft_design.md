# Fused angular FFT candidate

The Galerkin nonlinear stage uses three unnormalized inverse real transforms, the two products \(u_r\omega\) and \(u_\theta\omega\), and two unnormalized forward real transforms. The FD stage uses four inverse real transforms and one forward transform of \(u_r \partial_s\omega + u_\theta \partial_\theta\omega\). Its mean forcing remains the original ascending sum over the separately constructed Fourier contributions. The current raw flux is copied to the SBDF2 history before it is overwritten. No new scaling is applied because the existing projection or FD right-hand side contains \(1/N\). The full complex coefficients retain both sine and cosine components. [NVIDIA cuFFT transform conventions](https://docs.nvidia.com/cuda/cufft/index.html)

## Factorization

For radices \(r_0,\ldots,r_{s-1}\), write a scratch index as \(i=d_0+r_0d_1+\cdots\). Load source index \(p(i)=\sum_j d_j N/\prod_{\ell=0}^{j}r_\ell\). Let \(m\) be the product of previous radices. At a stage of width \(r\), each block butterfly combines the entries at \(j+tm\) for \(0\le t<r\). It first multiplies entry \(t\) by \(\exp(\sigma 2\pi i\,tj/(rm))\), then applies the length-\(r\) DFT with the same sign \(\sigma\). This is the Cooley–Tukey split by source residue modulo \(r\) of the stride-\(N/(rm)\) subtransform. By induction, after that stage each contiguous \(rm\) scratch segment holds the natural-order spectrum of its corresponding strided subsequence. The last stage covers the whole row. Positive sign gives the unnormalized inverse and negative sign the unnormalized forward transform. [Cooley and Tukey, 1965](https://web.stanford.edu/class/cme324/classics/cooley-tukey.pdf)

Two Hermitian input spectra \(A,B\) can share one complex inverse transform by forming \(A+iB\) from their separate full-spectrum expansions. Its real and imaginary outputs are their original real signals. The packed spectrum generally loses Hermitian symmetry, so a negative mode must conjugate \(A\) and \(B\) separately. For two real physical products, a complex forward transform \(Z\) of their packed sum yields \(G_k=(Z_k+\overline{Z_{-k}})/2\) and \(H_k=(Z_k-\overline{Z_{-k}})/(2i)\). The DC mode and the even-length Nyquist mode use their real endpoint convention. Pairing changes the implementation work but leaves the original five logical field transforms and one nonlinear evaluation intact.

The physical products have Fourier support up to \(2K\). A nonzero wrap into retained \(\lvert k\rvert\le K\) would require \(\lvert k+\ell N\rvert\le 2K\) for some nonzero \(\ell\). The triangle inequality gives \(\lvert k+\ell N\rvert\ge N-K>2K\) when \(N>3K\). This proves the same retained-mode de-aliasing rule for both solvers.

## Mapping and cost

One 128-thread block advances each angular row through inverse transforms, products and forward transforms. Two complex shared planes hold the inverse fields. Threads retain their two possible products in registers while the shared planes are read, then reuse the first plane for the forward transform. Threads own disjoint butterflies within each stage and synchronize before changing stages. The kernel keeps the FD mean and raw flux history paths unchanged.

| Angular length | Radices | Shared bytes per block | Block barriers | Original logical transforms |
|---|---|---:|---:|---:|
| 200 | 2, 2, 2, 5, 5 | 6,400 | 13 | 5 |
| 225 | 3, 3, 5, 5 | 7,200 | 11 | 5 |
| 243 | 3, 3, 3, 3, 3 | 7,776 | 13 | 5 |

Eliminating the separate inverse and forward physical arrays removes up to \(80N\) bytes of global-array traffic per row and nonlinear evaluation. The bound is 1.188 MB at middle Galerkin, 1.477 MB at high Galerkin, and 3.402 or 3.558 MB at FD176 or FD184. Cache hits reduce the actual device-memory savings. The current radix implementation uses approximately 32,040 real floating-point operations per N225 row and 34,020 per N243 row for the three internal complex transforms, excluding the products, packing and memory operations. FP64 throughput, block occupancy, shared access patterns and fused-kernel register allocation still matter. The new node replaces three graph-stage submissions with one. The source has yet to establish a device speedup.

## Numerical and validation boundaries

The CPU audit covers ordinary, smooth, retained-edge and imbalanced spectra, both PDE discretizations, the unnormalized NumPy FFT oracle, independent retained-mode convolution, exact flux history and exact FD mean. Pair packing is sensitive to strongly unequal channel magnitudes. Scaling the first Galerkin velocity field by \(10^{-10}\) raised that tiny output flux's worst relative error to \(2.739\times10^{-6}\) while aggregate flux error remained at FP64 rounding scale. This distinction must be tracked in device fixtures and full trajectories.

CUDA 11.8 NVRTC compiled all six fused entry points with the original compute target in standalone, Galerkin-bundled and FD-bundled sources. A device check compares both eager and captured graph output against cuFFT and NumPy. A later trajectory check needs unchanged probe and field gates through \(T=72\), matched FFT work counters, and a fused FD arm before any performance comparison. Per-method timing must include the altered module and FFT setup, plan priming and synchronization under the same isolation rules.
