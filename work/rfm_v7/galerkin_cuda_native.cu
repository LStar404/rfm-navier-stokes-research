// Exact FP64 SBDF2 Galerkin hot path. Fourier modes are normalized complex amplitudes.

extern "C" __global__ void spectrum(
    const double2 *state, const double2 *amplitudes, const int *step,
    const double *jets, const double *mean, const double *radius,
    const double2 *lift, double2 *out, int w, int k, int q, int half)
{
    const int at = blockDim.x * blockIdx.x + threadIdx.x;
    const int plane = q * half;
    if (at >= plane) return;
    const int mode = at % half;
    const int r = at / half;
    const int stride = k + 1;
    const int frame = w * stride;
    const double2 *coef = state + (*step % 3) * frame;
    double2 ur = {0., 0.}, ut = {0., 0.}, vort = {0., 0.};
    if (mode == 0) {
        for (int j = 0; j < w; ++j) {
            const double a = coef[j * stride].x;
            ut.x += mean[r * w + j] * a;
            vort.x += mean[(q + r) * w + j] * a;
        }
    } else if (mode <= k) {
        double2 psi = {0., 0.};
        for (int j = 0; j < w; ++j) {
            const double2 a = coef[j * stride + mode];
            const double f = jets[r * w + j];
            const double d = jets[(q + r) * w + j];
            const double dd = jets[(2 * q + r) * w + j];
            psi.x += f * a.x;
            psi.y += f * a.y;
            ut.x += d * a.x;
            ut.y += d * a.y;
            vort.x += dd * a.x;
            vort.y += dd * a.y;
        }
        const double factor = radius[r] * mode;
        ur.x = -factor * psi.y;
        ur.y = factor * psi.x;
        vort.x += (mode * mode) * psi.x;
        vort.y += (mode * mode) * psi.y;
        if (mode == 1) {
            const double2 speed = amplitudes[*step];
            const double2 a = lift[r];
            ur.x += speed.x * a.x - speed.y * a.y;
            ur.y += speed.x * a.y + speed.y * a.x;
            const double v = lift[q + r].x;
            const double omega = lift[2 * q + r].x;
            ut.x += speed.x * v;
            ut.y += speed.y * v;
            vort.x += speed.x * omega;
            vort.y += speed.y * omega;
        }
    }
    out[at] = ur;
    out[plane + at] = ut;
    out[2 * plane + at] = vort;
}


extern "C" __global__ void products(
    const double *physical, const double2 *spectrum, double *product,
    double2 *previous, int values, int fourier_values)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < values) {
        const double curl = physical[2 * values + i];
        product[i] = physical[i] * curl;
        product[values + i] = physical[values + i] * curl;
    }
    // The following FFT overwrites this spectrum after this kernel has retired.
    if (i < fourier_values) previous[i] = spectrum[i];
}


extern "C" __global__ void advance(
    double2 *state, const double2 *amplitudes, const int *step,
    const double2 *products, const double2 *history, const double *linear,
    const double *radial, const double *angular, const double *mass_lift,
    const double *bdf_lift, int w, int k, int q, int half)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = k + 1;
    const int frame = w * stride;
    if (i >= frame) return;
    const int mode = i % stride;
    const int row = i / stride;
    const int n = *step;
    const double2 *now = state + (n % 3) * frame;
    const double2 *before = state + ((n + 2) % 3) * frame;
    double2 *next = state + ((n + 1) % 3) * frame;
    double2 result = {0., 0.};
    for (int j = 0; j < w; ++j) {
        const int column = j * stride + mode;
        const double value = linear[(row * w + j) * stride + mode];
        result.x += value * (2. * now[column].x - .5 * before[column].x);
        result.y += value * (2. * now[column].y - .5 * before[column].y);
    }
    for (int r = 0; r < q; ++r) {
        const int p = r * half + mode;
        const int weight = (row * q + r) * stride + mode;
        const double factor = radial[weight];
        result.x += factor * (2. * products[p].x - history[p].x);
        result.y += factor * (2. * products[p].y - history[p].y);
        if (mode != 0) {
            const int a = q * half + p;
            const double factor_theta = angular[weight];
            result.x += factor_theta * (2. * products[a].y - history[a].y);
            result.y -= factor_theta * (2. * products[a].x - history[a].x);
        }
    }
    if (mode == 1) {
        const double2 current = amplitudes[n];
        const double2 past = amplitudes[n - 1];
        const double2 future = amplitudes[n + 1];
        result.x += mass_lift[row] * (2. * current.x - .5 * past.x) - bdf_lift[row] * future.x;
        result.y += mass_lift[row] * (2. * current.y - .5 * past.y) - bdf_lift[row] * future.y;
    }
    if (mode == 0) result.y = 0.;
    next[i] = result;
}


extern "C" __global__ void observe(
    const double2 *state, const double2 *amplitudes, int *step,
    const int *snapshot, const double *probe_operator, const double *probe_lift,
    double2 *checkpoints, double *probes, int frame, int stride, int increment)
{
    // A single block uses the same index before committing the next time step.
    const int n = *step + increment;
    const double2 *source = state + (n % 3) * frame;
    const int slot = snapshot[n];
    if (slot >= 0) {
        for (int i = threadIdx.x; i < frame; i += blockDim.x)
            checkpoints[slot * frame + i] = source[i];
    }
    if (n % stride == 0 && threadIdx.x < 6) {
        const int p = threadIdx.x;
        const double *weights = probe_operator + p * frame;
        double value = amplitudes[n].x * probe_lift[p];
        for (int i = 0; i < frame; ++i) value += weights[i] * source[i].x;
        probes[(n / stride) * 6 + p] = value;
    }
    __syncthreads();
    if (threadIdx.x == 0) *step = n;
}
