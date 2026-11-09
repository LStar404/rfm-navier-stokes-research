// Full-circle FP64 FD4 advection and SBDF2, reusing the CPU radial operators.

__device__ double2 stencil(const double2 *state, const int *columns, const double *weights,
                           int op, int row, int mode, int w, int modes, int width) {
    const int start = (op * w + row) * width;
    double2 v = {0., 0.};
    for (int j = 0; j < width; ++j) {
        const double2 x = state[columns[start+j] * modes + mode];
        const double a = weights[start+j];
        v.x += a * x.x;
        v.y += a * x.y;
    }
    return v;
}


extern "C" __global__ void fd_fields(
    const double2 *state, const double2 *amplitudes, const int *step,
    const int *columns, const double *weights, const double *lift,
    const double *inv_r, double2 *mass, double *mean_terms, double2 *spectrum,
    int w, int modes, int half, int stencil_width)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int plane = w * half;
    if (i >= plane) return;
    const int k = i % half;
    const int j = i / half;
    const int n = *step;
    const int frame = w * modes;
    const double2 *a = state + (n % 2) * frame;
    double2 ur = {0., 0.}, ut = {0., 0.}, ds = {0., 0.}, dtheta = {0., 0.};
    if (k == 0) {
        ut.x = a[j*modes].x;
        ds = stencil(a, columns, weights, 4, j, 0, w, modes, stencil_width);
        mass[(n % 2) * frame + j*modes] = ut;
        mean_terms[j*modes] = 0.;
    } else if (k < modes) {
        const double2 psi = a[j*modes+k];
        const double inv = inv_r[j];
        ur.x = -k * inv * psi.y;
        ur.y = k * inv * psi.x;
        const double2 d1 = stencil(a, columns, weights, 0, j, k, w, modes, stencil_width);
        ut.x = -inv * d1.x;
        ut.y = -inv * d1.y;
        double2 omega = stencil(a, columns, weights, 1, j, k, w, modes, stencil_width);
        const double k2 = (double)(k*k);
        omega.x += k2 * inv * inv * psi.x;
        omega.y += k2 * inv * inv * psi.y;
        ds = stencil(a, columns, weights, 2, j, k, w, modes, stencil_width);
        const double2 dsk = stencil(a, columns, weights, 3, j, k, w, modes, stencil_width);
        ds.x += k2 * dsk.x;
        ds.y += k2 * dsk.y;
        if (k == 1) {
            const double2 z = amplitudes[n];
            ut.x += lift[j] * z.x;
            ut.y += lift[j] * z.y;
            omega.x += lift[w+j] * z.x;
            omega.y += lift[w+j] * z.y;
            ds.x += lift[2*w+j] * z.x;
            ds.y += lift[2*w+j] * z.y;
        }
        dtheta.x = -k * omega.y;
        dtheta.y = k * omega.x;
        mass[(n % 2) * frame + j*modes+k] = omega;
        mean_terms[j*modes+k] = -2. * (ur.x * omega.x + ur.y * omega.y);
    }
    spectrum[i] = ur;
    spectrum[plane+i] = ut;
    spectrum[2*plane+i] = ds;
    spectrum[3*plane+i] = dtheta;
}


extern "C" __global__ void fd_products(
    const double *physical, const double2 *spectrum, const double *mean_terms,
    const int *step, double *product, double2 *prior, double *mean, int w,
    int modes, int ntheta, int half)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int plane = w * ntheta;
    if (i < plane) {
        product[i] = physical[i] * physical[2*plane+i] + physical[plane+i] * physical[3*plane+i];
        if (i % ntheta == 0) {
            const int row = i / ntheta;
            double result = 0.;
            for (int k = 1; k < modes; ++k) result += mean_terms[row*modes+k];
            mean[(*step % 2)*w+row] = result;
        }
    }
    if (i < w * half) prior[i] = spectrum[i];
}


extern "C" __global__ void fd_rhs(
    const double2 *amplitudes, const int *step, const double2 *mass, const double *mean,
    const double2 *flux, const double2 *prior, const double *inv_r, const double *lift_bdf,
    double2 *rhs, int w, int modes, int ntheta, int half, double dt)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int frame = w*modes;
    if (i >= frame) return;
    const int j = i / modes;
    const int k = i % modes;
    const int n = *step;
    const int now = n % 2;
    const int prev = (n+1) % 2;
    const double2 a = mass[now*frame+i], b = mass[prev*frame+i];
    double2 v = {2.*a.x-.5*b.x, 2.*a.y-.5*b.y};
    if (k == 0) {
        v.x += dt * (2.*mean[now*w+j] - mean[prev*w+j]);
        v.y = 0.;
    } else {
        const int f = j*half+k;
        const double t = -dt * inv_r[j] / ntheta;
        v.x += t * (2.*flux[f].x - prior[f].x);
        v.y += t * (2.*flux[f].y - prior[f].y);
        if (k == 1) {
            const double2 amplitude = amplitudes[n+1];
            v.x -= lift_bdf[j] * amplitude.x;
            v.y -= lift_bdf[j] * amplitude.y;
        }
    }
    rhs[i] = v;
}


extern "C" __global__ void fd_band(
    double2 *state, const int *step, const double2 *rhs, const double *factors,
    const int *pivots, int w, int modes, int lower, int upper)
{
    const int k = blockIdx.x, lane = threadIdx.x;
    const int diag = lower + upper;
    const int bands = 2*lower + upper + 1;
    const double *a = factors + k*w*bands;
    const int *pivot = pivots + k*w;
    extern __shared__ double2 x[];
    for (int j = lane; j < w; j += blockDim.x) x[j] = rhs[j*modes+k];
    __syncthreads();
    for (int j = 0; j < w-1; ++j) {
        const int p = pivot[j];
        if (p != j) {
            if (lane == 0) {
                const double2 tmp = x[j];
                x[j] = x[p];
                x[p] = tmp;
            }
            __syncthreads();
        }
        const int delta = lane+1;
        if (delta <= lower && j+delta < w) {
            const double l = a[j*bands+diag+delta];
            x[j+delta].x -= l*x[j].x;
            x[j+delta].y -= l*x[j].y;
        }
        __syncthreads();
    }
    double2 *next = state + ((*step+1) % 2)*w*modes;
    for (int j = w-1; j >= 0; --j) {
        // The one pivot value is broadcast within the warp. Its division is
        // redundant across lanes, so this step needs only one block barrier.
        const double reciprocal = 1./a[j*bands+diag];
        const double2 t = {x[j].x*reciprocal, x[j].y*reciprocal};
        if (lane == 0) next[j*modes+k] = t;
        const int delta = lane;
        if (delta > 0 && delta <= diag && j-delta >= 0) {
            const double u = a[j*bands+diag-delta];
            x[j-delta].x -= u*t.x;
            x[j-delta].y -= u*t.y;
        }
        __syncthreads();
    }
}


extern "C" __global__ void fd_dense(
    double2 *state, const int *step, const double2 *rhs, const double *inverse, int w, int modes)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int frame = w*modes;
    if (i >= frame) return;
    const int k = i % modes, j = i / modes;
    double2 result = {0., 0.};
    for (int t = 0; t < w; ++t) {
        const double factor = inverse[(j*w+t)*modes+k];
        const double2 x = rhs[t*modes+k];
        result.x += factor*x.x;
        result.y += factor*x.y;
    }
    state[((*step+1) % 2)*frame+i] = result;
}


extern "C" __global__ void fd_observe(
    const double2 *state, const double2 *amplitudes, int *step, const int *snapshot,
    const double *probe_operator, const double *probe_lift, double2 *checkpoints, double *probes,
    int w, int modes, int stride, int increment)
{
    const int n = *step + increment;
    const int frame = w*modes;
    const double2 *source = state + (n % 2)*frame;
    const int slot = snapshot[n];
    if (slot >= 0) {
        for (int i = threadIdx.x; i < frame; i += blockDim.x)
            checkpoints[slot*frame+(i % modes)*w+i/modes] = source[i];
    }
    if (n % stride == 0 && threadIdx.x < 6) {
        const int p = threadIdx.x;
        const double *weights = probe_operator + p*frame;
        double value = amplitudes[n].x * probe_lift[p];
        for (int i = 0; i < frame; ++i) value += weights[i] * source[i].x;
        probes[(n/stride)*6+p] = value;
    }
    __syncthreads();
    if (threadIdx.x == 0) *step = n;
}
