/* FP64 mode-one Taylor D2 and centered D1 curl, on the existing CSR rows. */
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

static double taylor_row(const int *offsets, const int *columns, const double *weights,
                         const double *delta, const double *source, int row, int entries,
                         double span, double moment0, double moment1) {
    const double center = source[(row+1)*entries];
    const double slope = (source[(row+2)*entries] - source[row*entries]) / span;
    double value = 0.;
    for (int z = offsets[row]; z < offsets[row+1]; ++z) {
        const double residual = (source[columns[z]*entries] - center) - slope*delta[z];
        value += weights[z]*residual;
    }
    return value + (moment0*center + moment1*slope);
}

static double centered_row(const int *offsets, const int *columns, const double *weights,
                           const double *source, int row, int entries, double moment0) {
    const double center = source[(row+1)*entries];
    double value = 0.;
    for (int z = offsets[row]; z < offsets[row+1]; ++z) {
        value += weights[z]*(source[columns[z]*entries] - center);
    }
    return value + moment0*center;
}

/* Spectral storage and nonselected arithmetic follow fd_cpu_buffered.c. */
EXPORT void fd_cpu_row_moment_fields(
    const int *d1_offsets, const int *d1_columns, const double *d1_weights,
    const int *d2_offsets, const int *d2_columns, const double *d2_weights,
    const int *wall_offsets, const int *wall_columns, const double *wall_weights,
    const double *psi, const double *inverse, const double *inverse_square,
    const double *left_d1, double *omega, double *spectrum, double *mean,
    const double *d1_mu0, const double *d2_mu0, const double *d2_mu1,
    const double *d2_delta, const double *span,
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
        /* Only k=1 uses the new arithmetic. Each real component is independent. */
        for (int z = 2; z < 4; ++z) {
            curl[z] = taylor_row(d2_offsets, d2_columns, d2_weights, d2_delta,
                                  psi+z, row, entries, span[row], d2_mu0[row], d2_mu1[row]);
        }
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
    /* The prescribed outer omega remains zero. */
    for (int row = 0; row < width; ++row) {
        double *ds = spectrum + 2*plane + row*padded;
        apply_row(d1_offsets, d1_columns, d1_weights, omega, ds, row, entries);
        for (int z = 2; z < 4; ++z) {
            ds[z] = centered_row(d1_offsets, d1_columns, d1_weights, omega+z,
                                  row, entries, d1_mu0[row]);
        }
    }
}
