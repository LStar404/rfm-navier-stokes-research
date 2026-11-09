/* Mode-contiguous application of the same precontracted Galerkin operators.
 * No allocation, changed Fourier truncation, or approximate matrix product.
 */
#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

EXPORT void galerkin_contract(
    const double *restrict linear, const double *restrict radial,
    const double *restrict angular, const double *restrict state,
    const double *restrict radial_flux, const double *restrict angular_flux,
    double *restrict mass, double *restrict flux,
    int width, int quadrature, int modes,
    int state_stride, int radial_stride, int angular_stride) {
    for (int row=0; row<width; ++row) {
        double *mr=mass + 2*row*modes;
        double *fr=flux + 2*row*modes;
        for (int k=0; k<2*modes; ++k) mr[k]=fr[k]=0.;
        for (int j=0; j<width; ++j) {
            const double *b=linear + (row*width+j)*modes;
            const double *v=state + 2*j*state_stride;
            for (int k=0; k<modes; ++k) {
                mr[2*k] += b[k]*v[2*k];
                mr[2*k+1] += b[k]*v[2*k+1];
            }
        }
        for (int j=0; j<quadrature; ++j) {
            const double *r=radial + (row*quadrature+j)*modes;
            const double *a=angular + (row*quadrature+j)*modes;
            const double *u=radial_flux + 2*j*radial_stride;
            const double *v=angular_flux + 2*j*angular_stride;
            for (int k=0; k<modes; ++k) {
                fr[2*k] += r[k]*u[2*k] + a[k]*v[2*k+1];
                fr[2*k+1] += r[k]*u[2*k+1] - a[k]*v[2*k];
            }
        }
        /* The mean velocity is real; unused rows have zero operators. */
        mr[1]=fr[1]=0.;
    }
}
