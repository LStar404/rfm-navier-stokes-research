// Selective row-moment corrections for the k=1 lifting mode.

extern "C" __global__ void rowcomp_curl_k1(
    const double2 *psi, const int *step, const int *offsets, const int *columns,
    const double *weights, const double *delta, const double *span,
    const double *mu0, const double *mu1, const double *inverse_square,
    double2 *omega, double2 *mass, double *mean_terms, double2 *spectrum,
    int w, int modes, int half)
{
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= w) return;
    const int radial = row+1, p = radial*modes+1;
    const double2 center = psi[p];
    const double2 left = psi[p-modes], right = psi[p+modes];
    const double2 slope = make_double2(__ddiv_rn(__dsub_rn(right.x, left.x), span[row]),
                                       __ddiv_rn(__dsub_rn(right.y, left.y), span[row]));
    double2 accum = make_double2(0., 0.);
    for (int tap = offsets[row]; tap < offsets[row+1]; ++tap) {
        const double2 value = psi[columns[tap]*modes+1];
        const double re = __dsub_rn(__dsub_rn(value.x, center.x), __dmul_rn(slope.x, delta[tap]));
        const double im = __dsub_rn(__dsub_rn(value.y, center.y), __dmul_rn(slope.y, delta[tap]));
        accum.x = __dadd_rn(accum.x, __dmul_rn(weights[tap], re));
        accum.y = __dadd_rn(accum.y, __dmul_rn(weights[tap], im));
    }
    const double restore_x = __dadd_rn(__dmul_rn(mu0[row], center.x), __dmul_rn(mu1[row], slope.x));
    const double restore_y = __dadd_rn(__dmul_rn(mu0[row], center.y), __dmul_rn(mu1[row], slope.y));
    const double d2x = __dadd_rn(accum.x, restore_x), d2y = __dadd_rn(accum.y, restore_y);
    const double2 curl = make_double2(-__dsub_rn(d2x, center.x)*inverse_square[radial],
                                      -__dsub_rn(d2y, center.y)*inverse_square[radial]);
    const int i = row*half+1;
    const double2 ur = spectrum[i];
    omega[p] = curl;
    mass[(*step % 2)*w*modes+row*modes+1] = curl;
    mean_terms[row*modes+1] = -2.*__dadd_rn(__dmul_rn(ur.x, curl.x), __dmul_rn(ur.y, curl.y));
    spectrum[3*w*half+i] = make_double2(-curl.y, curl.x);
}

extern "C" __global__ void rowcomp_gradient_k1(
    const double2 *omega, const int *offsets, const int *columns,
    const double *weights, const double *mu0, double2 *spectrum,
    int w, int modes, int half)
{
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= w) return;
    const double2 center = omega[(row+1)*modes+1];
    double2 accum = make_double2(0., 0.);
    for (int tap = offsets[row]; tap < offsets[row+1]; ++tap) {
        const double2 value = omega[columns[tap]*modes+1];
        accum.x = __dadd_rn(accum.x, __dmul_rn(weights[tap], __dsub_rn(value.x, center.x)));
        accum.y = __dadd_rn(accum.y, __dmul_rn(weights[tap], __dsub_rn(value.y, center.y)));
    }
    spectrum[2*w*half+row*half+1] = make_double2(
        __dadd_rn(accum.x, __dmul_rn(mu0[row], center.x)),
        __dadd_rn(accum.y, __dmul_rn(mu0[row], center.y)));
}
