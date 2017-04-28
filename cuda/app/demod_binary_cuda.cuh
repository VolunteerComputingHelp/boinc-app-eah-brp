/***************************************************************************
 *   Copyright (C) 2010 by Oliver Bock                                     *
 *   oliver.bock[AT]aei.mpg.de                                             *
 *                                                                         *
 *   This file is part of Einstein@Home (Radio Pulsar Edition).            *
 *                                                                         *
 *   Description:                                                          *
 *                                                                         *
 *   Einstein@Home is free software: you can redistribute it and/or modify *
 *   it under the terms of the GNU General Public License as published     *
 *   by the Free Software Foundation, version 2 of the License.            *
 *                                                                         *
 *   Einstein@Home is distributed in the hope that it will be useful,      *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with Einstein@Home. If not, see <http://www.gnu.org/licenses/>. *
 *                                                                         *
 ***************************************************************************/

#ifndef DEMOD_BINARY_CUDA_CUH
#define DEMOD_BINARY_CUDA_CUH

#ifdef  __cplusplus
extern "C" {
#endif

// use constant (cached) device memory for sin/cos samples and params (computed on host)
__constant__ float constSinSamples[ERP_SINCOS_LUT_SIZE];
__constant__ float constCosSamples[ERP_SINCOS_LUT_SIZE];
__constant__ float LUT_TWO_PI;
__constant__ float LUT_TWO_PI_INV;

// use global device variable to propagate resampled time series info
__device__ int timeSeriesLength = 0;


__device__ float sinLUTLookup(float x)
{
    float xt;
    int i0;
    float d, d2;
    float ts, tc;

    // normalize value
    xt = modff(x * LUT_TWO_PI_INV, &x); // xt in (-1, 1)
    if ( xt < 0.0f ) {
        xt += 1.0f;                     // xt in [0, 1 )
    }

    // determine LUT index
    i0 = (int) __fadd_rn(__fmul_rn(xt, ERP_SINCOS_LUT_RES_F), 0.5f);
    d = d2 = __fmul_rn(LUT_TWO_PI, __fadd_rn(xt, -__fmul_rn(ERP_SINCOS_LUT_RES_F_INV, i0)));
    d2 *= 0.5f * d;

    // fetch sin/cos samples from constant memory
    ts = constSinSamples[i0];
    tc = constCosSamples[i0];

    //use taylor-expansion for sin around samples
    return __fadd_rn(__fadd_rn(ts, __fmul_rn(d, tc)), -(__fmul_rn(d2, ts)));
}


#define CUDA_RESAMP_OFFSETS_BLOCKDIM_X 128

__global__ void time_series_modulation(float *del_t, float tau, float Omega, float Psi0, float dt, float step_inv, float S0)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // compute time offset
    float t = i * dt;
    float x = __fadd_rn(__fmul_rn(Omega, t), Psi0);
    float sinX = sinLUTLookup(x);

    // compute time offsets
    del_t[i] = __fadd_rn(__fmul_rn(__fmul_rn(tau, sinX), step_inv), -S0);
}


__global__ void time_series_length_modulated(float *del_t, unsigned int nsamples_unpadded)
{
    // number of timesteps that fit into the duration = at most the amount we had before
    unsigned int n_steps = nsamples_unpadded - 1;

    // TODO: avoid global memory reads!!!
    // nearest_idx (see time_series_resampling kernel) must not exceed n_unpadded - 1, so go back as far as needed to ensure that
    while(n_steps - del_t[n_steps] >= nsamples_unpadded - 1) {
        n_steps--;
    }

    // copy length into global variable
    timeSeriesLength = n_steps;
}


#define CUDA_RESAMP_BLOCKDIM_X 384

__global__ void time_series_resampling(float *input, float *del_t, float *output, float *meanBuffer, int nsamples_unpadded, int length)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO: ensure coalesced memory access (load/store) !!!
    // only resample "existing" time samples
    if(i < length) {
        // sample i arrives at the detector at i - del_t[i], choose nearest neighbor
        int nearest_idx = (int)(i - del_t[i] + 0.5f);

        // set i-th bin in resampled time series (at the pulsar) to nearest_idx bin from de-dispersed time series
        output[i] = input[nearest_idx];
    }
    else {
        // set remaining buffercells to zero (for upcoming sum reduction)
        output[i] = 0.0f;
    }
}


#define CUDA_RESAMP_REDUCTION_BLOCKDIM_X 128

__global__ void time_series_mean_reduction(float *input, float *output)
{
    __shared__ float sharedPartialSum[CUDA_RESAMP_REDUCTION_BLOCKDIM_X];

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // coalesced load of time series data into shared memory
    sharedPartialSum[threadIdx.x] = input[i];

    // wait for load to finish
    __syncthreads();

    // compute sum of current block (in log2(blocksize) iterations)
    for(unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        // sum two strided values per thread
        if(threadIdx.x < stride) {
            sharedPartialSum[threadIdx.x] += sharedPartialSum[threadIdx.x + stride];
        }

        // wait for (partial) block summing iteration to finish
        __syncthreads();
    }

    // store sum of current block in global memory (single thread)
    if(threadIdx.x == 0) {
        output[blockIdx.x] = sharedPartialSum[0];
    }
}


#define CUDA_RESAMP_PADDING_BLOCKDIM_X 512

__global__ void time_series_padding(float *output, float mean, int offset)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // can't be avoided as time series varies in length (incl. non-multiple-of-32 values)
    if(i >= offset) {
        // coalesced store of resampled time series padding data to global memory
        output[i] = mean;
    }
}


#define CUDA_FFT_BLOCKDIM_X 256

__global__ void fft_powerspectrum(cufftComplex *fft_data, float *ps_data, float norm_factor)
{
    __shared__ cufftComplex sharedFFTData[CUDA_FFT_BLOCKDIM_X];
   
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float nf;
    // coalesced load of FFT data into shared memory
    sharedFFTData[threadIdx.x] = fft_data[i];

    // wait for load to finish
    __syncthreads();
    
    // computer power spectrum
    nf = (i==0) ? 0.0f : norm_factor;	
    ps_data[i] = __fmul_rn(nf, __fadd_rn(__fmul_rn(sharedFFTData[threadIdx.x].x, sharedFFTData[threadIdx.x].x), __fmul_rn(sharedFFTData[threadIdx.x].y, sharedFFTData[threadIdx.x].y)));
}

#ifdef  __cplusplus
}
#endif

#endif
