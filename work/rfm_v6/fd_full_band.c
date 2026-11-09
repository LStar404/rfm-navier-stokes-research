/*
 * Solve all independent band systems with cached real LAPACK DGBTRF factors.
 * ab layout is [mode][band row][column], rhs is interleaved complex128.
 */
#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

EXPORT void fd_solve_bands(
    const double *ab, const int *pivots, double *rhs,
    int modes, int n, int lower, int upper
) {
    const int diag = lower + upper;
    const int bands = 2 * lower + upper + 1;
    for (int mode = 0; mode < modes; mode++) {
        const double *f = ab + mode * bands * n;
        const int *ipiv = pivots + mode * n;
        double *x = rhs + 2 * mode * n;
        for (int j = 0; j < n - 1; j++) {
            const int p = ipiv[j];
            if (p != j) {
                const double r = x[2*j];
                const double i = x[2*j+1];
                x[2*j] = x[2*p];
                x[2*j+1] = x[2*p+1];
                x[2*p] = r;
                x[2*p+1] = i;
            }
            const int last = j + lower < n-1 ? j + lower : n-1;
            for (int i = j+1; i <= last; i++) {
                const double l = f[(diag+i-j)*n+j];
                x[2*i] -= l*x[2*j];
                x[2*i+1] -= l*x[2*j+1];
            }
        }
        for (int j = n-1; j >= 0; j--) {
            const double inv = 1.0/f[diag*n+j];
            x[2*j] *= inv;
            x[2*j+1] *= inv;
            const int first = j-diag > 0 ? j-diag : 0;
            for (int i = first; i < j; i++) {
                const double u = f[(diag+i-j)*n+j];
                x[2*i] -= u*x[2*j];
                x[2*i+1] -= u*x[2*j+1];
            }
        }
    }
}
