// A single variable-step SBDF2 update and absolute-time observation slots.

extern "C" __global__ void advance_bridge(
    double2 *state, const double2 *amplitudes, const int *step,
    const double2 *products, const double2 *history, const double *linear,
    const double *radial, const double *angular, const double *mass_lift,
    const double *bdf_lift, int w, int k, int q, int half,
    double current_mass, double previous_mass, double current_flux, double previous_flux)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = k+1;
    const int frame = w*stride;
    if (i >= frame) return;
    const int mode = i % stride, row = i / stride, n = *step;
    const double2 *now = state + (n % 3)*frame;
    const double2 *before = state + ((n+2) % 3)*frame;
    double2 *next = state + ((n+1) % 3)*frame;
    double2 result = {0., 0.};
    for (int j = 0; j < w; ++j) {
        const int column = j*stride+mode;
        const double factor = linear[(row*w+j)*stride+mode];
        result.x += factor * (current_mass*now[column].x + previous_mass*before[column].x);
        result.y += factor * (current_mass*now[column].y + previous_mass*before[column].y);
    }
    for (int r = 0; r < q; ++r) {
        const int p = r*half+mode;
        const int weight = (row*q+r)*stride+mode;
        const double factor = radial[weight];
        result.x += factor * (current_flux*products[p].x + previous_flux*history[p].x);
        result.y += factor * (current_flux*products[p].y + previous_flux*history[p].y);
        if (mode != 0) {
            const int a = q*half+p;
            const double factor_theta = angular[weight];
            result.x += factor_theta * (current_flux*products[a].y + previous_flux*history[a].y);
            result.y -= factor_theta * (current_flux*products[a].x + previous_flux*history[a].x);
        }
    }
    if (mode == 1) {
        const double2 current = amplitudes[n], past = amplitudes[n-1], future = amplitudes[n+1];
        result.x += mass_lift[row]*(current_mass*current.x + previous_mass*past.x) - bdf_lift[row]*future.x;
        result.y += mass_lift[row]*(current_mass*current.y + previous_mass*past.y) - bdf_lift[row]*future.y;
    }
    if (mode == 0) result.y = 0.;
    next[i] = result;
}

extern "C" __global__ void observe_absolute(
    const double2 *state, const double2 *amplitudes, int *step,
    const int *snapshot, const int *probe_slot, const double *probe_operator,
    const double *probe_lift, double2 *checkpoints, double *probes, int frame, int increment)
{
    const int n = *step+increment;
    const double2 *source = state + (n % 3)*frame;
    const int slot = snapshot[n];
    if (slot >= 0) {
        for (int i = threadIdx.x; i < frame; i += blockDim.x)
            checkpoints[slot*frame+i] = source[i];
    }
    const int sample = probe_slot[n];
    if (sample >= 0 && threadIdx.x < 6) {
        const int p = threadIdx.x;
        const double *weights = probe_operator+p*frame;
        double value = amplitudes[n].x*probe_lift[p];
        for (int i = 0; i < frame; ++i) value += weights[i]*source[i].x;
        probes[sample*6+p] = value;
    }
    __syncthreads();
    if (threadIdx.x == 0) *step = n;
}
