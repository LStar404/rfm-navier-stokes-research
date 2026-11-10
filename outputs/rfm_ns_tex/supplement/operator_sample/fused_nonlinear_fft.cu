// One FP64 block per radial row. CuFFT-compatible signs and no normalization.
namespace rfm_fused_fft {

__device__ __forceinline__ double2 add(double2 a, double2 b) {
    return make_double2(a.x+b.x,a.y+b.y);
}
__device__ __forceinline__ double2 sub(double2 a, double2 b) {
    return make_double2(a.x-b.x,a.y-b.y);
}
__device__ __forceinline__ double2 scale(double2 a, double b) {
    return make_double2(a.x*b,a.y*b);
}
__device__ __forceinline__ double2 times_i(double2 a) {
    return make_double2(-a.y,a.x);
}
__device__ __forceinline__ double2 multiply(double2 a, double2 b) {
    return make_double2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);
}

template<int Radix, bool Inverse>
__device__ __forceinline__ void butterfly(double2 *v) {
    const double sign = Inverse ? 1. : -1.;
    const double2 a = v[0], b = v[1];
    if (Radix == 2) {
        v[0] = add(a,b);
        v[1] = sub(a,b);
    } else if (Radix == 3) {
        const double2 sum = add(b,v[2]);
        const double2 even = sub(a,scale(sum,.5));
        const double2 odd = times_i(scale(sub(b,v[2]),sign*0.86602540378443865));
        v[0] = add(a,sum);
        v[1] = add(even,odd);
        v[2] = sub(even,odd);
    } else {
        const double c1 = 0.30901699437494745, c2 = -0.80901699437494734;
        const double s1 = 0.95105651629515353, s2 = 0.58778525229247325;
        const double2 p = add(b,v[4]), q = add(v[2],v[3]);
        const double2 d = sub(b,v[4]), e = sub(v[2],v[3]);
        const double2 m1 = add(add(a,scale(p,c1)),scale(q,c2));
        const double2 m2 = add(add(a,scale(p,c2)),scale(q,c1));
        const double2 j1 = times_i(scale(add(scale(d,s1),scale(e,s2)),sign));
        const double2 j2 = times_i(scale(sub(scale(d,s2),scale(e,s1)),sign));
        v[0] = add(add(a,p),q);
        v[1] = add(m1,j1);
        v[2] = add(m2,j2);
        v[3] = sub(m2,j2);
        v[4] = sub(m1,j1);
    }
}

template<int N, int Stride, int Radix, bool Inverse, int Planes>
__device__ __forceinline__ void stage(double2 *scratch, const double2 *roots) {
    const int butterflies = N/Radix;
    for (int at = threadIdx.x; at < Planes*butterflies; at += 128) {
        const int plane = at/butterflies, column = at%butterflies;
        const int j = column%Stride, group = column/Stride;
        double2 *row = scratch + plane*N + group*Radix*Stride + j;
        double2 values[5];
        values[0] = row[0];
        #pragma unroll
        for (int t = 1; t < Radix; ++t) {
            double2 twiddle = roots[t*j*(N/(Radix*Stride))];
            if (Inverse) twiddle.y = -twiddle.y;
            values[t] = multiply(row[t*Stride],twiddle);
        }
        butterfly<Radix,Inverse>(values);
        #pragma unroll
        for (int t = 0; t < Radix; ++t) row[t*Stride] = values[t];
    }
    __syncthreads();
}

template<int N, bool Inverse, int Planes>
__device__ __forceinline__ void transform(double2 *scratch, const double2 *roots) {
    if (N == 200) {
        stage<N,1,2,Inverse,Planes>(scratch,roots);
        stage<N,2,2,Inverse,Planes>(scratch,roots);
        stage<N,4,2,Inverse,Planes>(scratch,roots);
        stage<N,8,5,Inverse,Planes>(scratch,roots);
        stage<N,40,5,Inverse,Planes>(scratch,roots);
    } else if (N == 225) {
        stage<N,1,3,Inverse,Planes>(scratch,roots);
        stage<N,3,3,Inverse,Planes>(scratch,roots);
        stage<N,9,5,Inverse,Planes>(scratch,roots);
        stage<N,45,5,Inverse,Planes>(scratch,roots);
    } else {
        stage<N,1,3,Inverse,Planes>(scratch,roots);
        stage<N,3,3,Inverse,Planes>(scratch,roots);
        stage<N,9,3,Inverse,Planes>(scratch,roots);
        stage<N,27,3,Inverse,Planes>(scratch,roots);
        stage<N,81,3,Inverse,Planes>(scratch,roots);
    }
}

template<int N>
__device__ __forceinline__ double2 hermitian(const double2 *half_row, int k) {
    if (k <= N/2) {
        double2 value = half_row[k];
        if (k == 0 || (N%2 == 0 && k == N/2)) value.y = 0.;
        return value;
    }
    const double2 value = half_row[N-k];
    return make_double2(value.x,-value.y);
}

template<int N, bool FD>
__device__ __forceinline__ double2 physical_product(const double2 *scratch, int i) {
    const double2 velocity = scratch[i], derivative = scratch[N+i];
    if (FD) return make_double2(velocity.x*derivative.x+velocity.y*derivative.y,0.);
    return scale(velocity,derivative.x);
}

template<int N, bool FD>
__device__ __forceinline__ void nonlinear(
    const double2 *spectrum, double2 *flux, double2 *history, const int *permutation,
    const double2 *roots, const double *mean_terms, const int *step, double *mean,
    int rows, int modes, double2 *scratch) {
    const int radial = blockIdx.x, lane = threadIdx.x, half = N/2+1;
    const int plane = rows*half, row = radial*half;
    if (lane < half) {
        history[row+lane] = flux[row+lane];
        if (!FD) history[plane+row+lane] = flux[plane+row+lane];
    }
    for (int i = lane; i < N; i += 128) {
        const int k = permutation[i];
        scratch[i] = add(hermitian<N>(spectrum+row,k),times_i(hermitian<N>(spectrum+plane+row,k)));
        const double2 third = hermitian<N>(spectrum+2*plane+row,k);
        scratch[N+i] = FD ? add(third,times_i(hermitian<N>(spectrum+3*plane+row,k))) : third;
    }
    __syncthreads();
    transform<N,true,2>(scratch,roots);

    // Registers retain both products while the inverse buffers are still read.
    // Only after that read phase may shared storage be reused for the forward FFT.
    const double2 first = physical_product<N,FD>(scratch,permutation[lane]);
    double2 second = make_double2(0.,0.);
    if (lane+128 < N) second = physical_product<N,FD>(scratch,permutation[lane+128]);
    __syncthreads();
    scratch[lane] = first;
    if (lane+128 < N) scratch[lane+128] = second;
    __syncthreads();
    transform<N,false,1>(scratch,roots);

    if (lane < half) {
        const double2 z = scratch[lane];
        if (FD) {
            const bool real_mode = lane == 0 || (N%2 == 0 && lane == N/2);
            flux[row+lane] = make_double2(z.x,real_mode ? 0. : z.y);
        } else {
            const double2 opposite = scratch[lane == 0 ? 0 : N-lane];
            flux[row+lane] = make_double2(.5*(z.x+opposite.x),.5*(z.y-opposite.y));
            flux[plane+row+lane] = make_double2(.5*(z.y+opposite.y),.5*(opposite.x-z.x));
        }
    }
    if (FD && lane == 0) {
        double value = 0.;
        for (int mode = 1; mode < modes; ++mode) value += mean_terms[radial*modes+mode];
        mean[(*step%2)*rows+radial] = value;
    }
}

} // namespace rfm_fused_fft

#define RFM_FUSED_ENTRY(Name,N,FD) \
extern "C" __global__ void Name( \
    const double2 *spectrum, double2 *flux, double2 *history, const int *permutation, \
    const double2 *roots, const double *mean_terms, const int *step, double *mean, int rows, int modes) { \
    extern __shared__ double2 scratch[]; \
    rfm_fused_fft::nonlinear<N,FD>(spectrum,flux,history,permutation,roots,mean_terms,step,mean,rows,modes,scratch); \
}

RFM_FUSED_ENTRY(fused_gal_200,200,false)
RFM_FUSED_ENTRY(fused_gal_225,225,false)
RFM_FUSED_ENTRY(fused_gal_243,243,false)
RFM_FUSED_ENTRY(fused_fd_200,200,true)
RFM_FUSED_ENTRY(fused_fd_225,225,true)
RFM_FUSED_ENTRY(fused_fd_243,243,true)

#undef RFM_FUSED_ENTRY
