/*
 * Fused CPU FD4 radial evaluation. The first pass forms curl and
 * velocities; the second applies D1 to curl. Both use the exact narrow
 * stencils assembled by the original solver. Spectral storage is radial
 * first, interleaved real/imaginary with four padded FFT planes.
 */
#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

static void apply_row(const int *offsets, const int *columns, const double *weights,
                      const double *source, double *restrict target,
                      int row, int entries) {
    for (int i = 0; i < entries; ++i) target[i] = 0.;
    for (int z = offsets[row]; z < offsets[row+1]; ++z) {
        const double *x = source + columns[z] * entries;
        const double a = weights[z];
        for (int i = 0; i < entries; ++i) target[i] += a * x[i];
    }
}

EXPORT void fd_cpu_fields(
    const int *d1_offsets, const int *d1_columns, const double *d1_weights,
    const int *d2_offsets, const int *d2_columns, const double *d2_weights,
    const int *wall_offsets, const int *wall_columns, const double *wall_weights,
    const double *psi, const double *inverse, const double *inverse_square,
    const double *left_d1, double *omega, double *spectrum, double *mean,
    int width, int modes, int half) {
    const int entries = 2*modes;
    const int padded = 2*half;
    const int plane = width*padded;
    apply_row(wall_offsets, wall_columns, wall_weights, psi, omega, 0, entries);
    for (int i = 0; i < entries; ++i) omega[i] = -omega[i] * inverse_square[0];
    for (int z = 0; z < 2; ++z) {
        double value = 0.;
        for (int tap = 0; tap < 5; ++tap) value += left_d1[tap]*psi[tap*entries+z];
        omega[z] = value * inverse[0];
    }
    for (int row = 0; row < width; ++row) {
        const double *v = psi + (row+1)*entries;
        double *curl = omega + (row+1)*entries;
        double *ur = spectrum + row*padded;
        double *ut = ur + plane;
        double *theta = ut + 2*plane;
        const double inv = inverse[row+1], inv2 = inverse_square[row+1];
        apply_row(d1_offsets, d1_columns, d1_weights, psi, ut, row, entries);
        apply_row(d2_offsets, d2_columns, d2_weights, psi, curl, row, entries);
        curl[0] = (ut[0]+v[0])*inv;
        curl[1] = (ut[1]+v[1])*inv;
        ur[0] = 0.;
        ur[1] = 0.;
        ut[0] = v[0];
        ut[1] = v[1];
        theta[0] = 0.;
        theta[1] = 0.;
        double sum = 0.;
        for (int k = 1; k < modes; ++k) {
            const int i = 2*k;
            const double k2 = (double)k*k;
            ur[i] = (-k*v[i+1])*inv;
            ur[i+1] = (k*v[i])*inv;
            ut[i] = -ut[i]*inv;
            ut[i+1] = -ut[i+1]*inv;
            curl[i] = -(curl[i]-k2*v[i])*inv2;
            curl[i+1] = -(curl[i+1]-k2*v[i+1])*inv2;
            theta[i] = -k*curl[i+1];
            theta[i+1] = k*curl[i];
            sum += ur[i]*curl[i] + ur[i+1]*curl[i+1];
        }
        mean[row] = -2.*sum;
    }
    /* The outer vorticity row is zero throughout the solve. */
    for (int row = 0; row < width; ++row) {
        double *ds = spectrum + 2*plane + row*padded;
        apply_row(d1_offsets, d1_columns, d1_weights, omega, ds, row, entries);
    }
}

EXPORT void fd_cpu_products(double *physical, int plane) {
    double *radial = physical;
    const double *angular = physical + plane;
    const double *ds = physical + 2*plane;
    const double *theta = physical + 3*plane;
    for (int i = 0; i < plane; ++i) radial[i] = radial[i]*ds[i] + angular[i]*theta[i];
}
