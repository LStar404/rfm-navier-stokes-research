// CUDA kernels specific to row-equilibrated FD4 and staged radial curl.

extern "C" __global__ void pow2_scale_rhs(double2 *rhs, const double *scale, int frame) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < frame) {
        rhs[i].x *= scale[i];
        rhs[i].y *= scale[i];
    }
}

// Match the buffered CPU stencil's separate products and additions at the
// clustered boundary, where contributions can cancel at many digits.
__device__ double2 dot_row(const double2 *values, const int *columns,
                           const double *weights, int taps, int modes, int mode) {
    double2 result = {0., 0.};
    for (int j = 0; j < taps; ++j) {
        const double2 value = values[columns[j] * modes + mode];
        result.x = __dadd_rn(result.x, __dmul_rn(weights[j], value.x));
        result.y = __dadd_rn(result.y, __dmul_rn(weights[j], value.y));
    }
    return result;
}

extern "C" __global__ void pow2_extend(
    const double2 *state, const double2 *amplitudes, const int *step,
    const double *right_mean, double2 *psi, double outer_radius, int w, int modes)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int frame = w*modes;
    if (i >= frame + 2*modes) return;
    if (i < modes) {
        psi[i] = make_double2(0., 0.);
    } else if (i < frame+modes) {
        psi[i] = state[(*step % 2)*frame+i-modes];
    } else {
        const int k = i % modes;
        double2 edge = {0., 0.};
        if (k == 0) {
            const double2 *tail = state + (*step % 2)*frame+(w-4)*modes;
            for (int j = 0; j < 4; ++j) {
                edge.x = __dadd_rn(edge.x, __dmul_rn(right_mean[j], tail[j*modes].x));
                edge.y = __dadd_rn(edge.y, __dmul_rn(right_mean[j], tail[j*modes].y));
            }
        } else if (k == 1) {
            const double2 speed = amplitudes[*step];
            edge = make_double2(outer_radius*speed.x, outer_radius*speed.y);
        }
        psi[i] = edge;
    }
}

extern "C" __global__ void pow2_curl(
    const double2 *psi, const int *step, const int *columns, const double *weights,
    const int *wall_columns, const double *wall_weights, const double *left_d1,
    const double *inverse, const double *inverse_square, double2 *omega,
    double2 *mass, double *mean_terms, double2 *spectrum,
    int w, int modes, int half, int taps, int wall_taps)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int plane = w*half;
    if (i >= plane+2*modes) return;
    if (i >= plane) {
        const int edge = i-plane;
        if (edge < modes) {
            const int first_five[5] = {0, 1, 2, 3, 4};
            double2 value = edge == 0
                ? dot_row(psi, first_five, left_d1, 5, modes, 0)
                : dot_row(psi, wall_columns, wall_weights, wall_taps, modes, edge);
            const double factor = edge == 0 ? inverse[0] : -inverse_square[0];
            omega[edge] = make_double2(factor*value.x, factor*value.y);
        } else {
            omega[(w+1)*modes+edge-modes] = make_double2(0., 0.);
        }
        return;
    }
    const int row = i/half, k = i%half;
    double2 ur = {0., 0.}, ut = {0., 0.}, theta = {0., 0.};
    if (k < modes) {
        const int radial = row+1, p = radial*modes+k;
        const double2 value = psi[p];
        const double inv = inverse[radial], inv2 = inverse_square[radial];
        const int begin = row*taps;
        const double2 d1 = dot_row(psi, columns+begin, weights+begin, taps, modes, k);
        double2 curl;
        if (k == 0) {
            ut = value;
            curl = make_double2((d1.x+value.x)*inv, (d1.y+value.y)*inv);
            mass[(*step % 2)*w*modes+row*modes] = value;
            mean_terms[row*modes] = 0.;
        } else {
            const double2 d2 = dot_row(psi, columns+w*taps+begin, weights+w*taps+begin,
                                       taps, modes, k);
            const double k2 = (double)k*k;
            curl.x = -__dsub_rn(d2.x, __dmul_rn(k2, value.x))*inv2;
            curl.y = -__dsub_rn(d2.y, __dmul_rn(k2, value.y))*inv2;
            ur = make_double2((-k*value.y)*inv, (k*value.x)*inv);
            ut = make_double2(-d1.x*inv, -d1.y*inv);
            theta = make_double2(-k*curl.y, k*curl.x);
            mass[(*step % 2)*w*modes+row*modes+k] = curl;
            mean_terms[row*modes+k] = -2.*__dadd_rn(__dmul_rn(ur.x, curl.x),
                                                     __dmul_rn(ur.y, curl.y));
        }
        omega[p] = curl;
    }
    spectrum[i] = ur;
    spectrum[plane+i] = ut;
    spectrum[3*plane+i] = theta;
}

extern "C" __global__ void pow2_gradient(
    const double2 *omega, const int *columns, const double *weights,
    double2 *spectrum, int w, int modes, int half, int taps)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int plane = w*half;
    if (i >= plane) return;
    const int row = i/half, k = i%half;
    double2 ds = {0., 0.};
    if (k < modes) ds = dot_row(omega, columns+row*taps, weights+row*taps, taps, modes, k);
    spectrum[2*plane+i] = ds;
}
