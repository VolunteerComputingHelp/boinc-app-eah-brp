/***************************************************************************
 *   Copyright (C) 2010 by Oliver Bock                                     *
 *   oliver.bock[AT]aei.mpg.de                                             *
 *                                                                         *
 *   This file is part of Einstein@Home (Radio Pulsar Edition).            *
 *                                                                         *
 *   Description:                                                          *
 *   Demodulates dedispersed time series using a bank of orbital           *
 *   parameters. After this step, an FFT of the resampled time series is   *
 *   searched for pulsed, periodic signals by harmonic summing.            *
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

#include "demod_binary_cuda.h"

#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include "../../demod_binary.h"
#include "../../erp_utilities.h"
#include <cuda.h>
#include "cuda_utilities.h"

// TODO: do we wanna keep those global (or use proper C++, or pass them around)?
CUdevice cuDevice = NULL;                 // CUDA device pointer
CUcontext cuContext = NULL;               // CUDA context we work in
CUmodule cuModuleMain;                    // Device module, kernel and symbol handles
CUfunction kernelTimeSeriesModulation;
CUfunction kernelTimeSeriesLengthModulated;
CUfunction kernelTimeSeriesResampling;
CUfunction kernelTimeSeriesMeanReduction;
CUfunction kernelTimeSeriesPadding;
CUfunction kernelPowerSpectrum;
CUdeviceptr timeSeriesLengthDeviceBuffer;
CUresult cuResult = CUDA_SUCCESS;         // CUDA return results (driver API)
cufftResult_t cufResult = CUFFT_SUCCESS;  // CUFFT return results
int cudDriverVersion = 0;                 // Version of the installed CUDA driver (not runtime)
unsigned int cudApiVersion = 0;           // Version of the installed CUDA driver API
cudaDeviceProp cudDeviceProperties;       // CUDA device properties

CUdeviceptr originalTimeSeriesDeviceBuffer;    // Original time series device buffer
CUdeviceptr modTimeOffsetsDeviceBuffer;        // Modulated time offsets device buffer
CUdeviceptr timeSeriesMeanDeviceBuffer;        // Sum-reduction buffer for time series mean value
CUdeviceptr sinLUTDeviceBuffer;                // sin lookup table device buffer
CUdeviceptr cosLUTDeviceBuffer;                // cos lookup table device buffer

cufftHandle cufPlan;                      // FFT plan handle

#include "demod_binary_cuda.cuh"

// macro for fft padding (based on powerspectrum kernel's blocksize, defined in demod_binary_cuda.cuh)
#define PADDED_FFT_SIZE(fftsize) CUDA_FFT_BLOCKDIM_X * (unsigned int) ceil((float)fftsize / (float)CUDA_FFT_BLOCKDIM_X)


int initialize_cuda(int cudDeviceIdGiven, int *cudDeviceIdPtr)
{
    int i;
    int res;
    // has the CUDA device been set already (doing this a second time results in an error)?
    static bool cudaDeviceNotSet = true;
    static int cudDeviceId = *cudDeviceIdPtr;

    if (cudaDeviceNotSet) {
        // initialize driver API
        cuResult = cuInit(0);
        if(cuResult != CUDA_SUCCESS) {
            logMessage(error, true, "Couldn't initialize CUDA driver API (error: %i)!\n", cuResult);
            return(RADPUL_CUDA_DRIVER_INIT);
        }

#ifdef BOINCIFIED
        // if controlled by a BOINC client, check device_num in init_data.xml
        // terminate if we couldn't find one there.
        //
        if (!cudDeviceIdGiven && !running_standalone()) {

            // we already checked the command line for the device number, so pass empty command line here
            res = boinc_get_cuda_device_id(0,NULL, & cudDeviceId);
            if (res) {
                logMessage(error, true, "No suitable CUDA device available!\n");
                return(RADPUL_CUDA_DEVICE_FIND);
            } else {
                cudDeviceIdGiven = 1;
            }
        }
#endif

        // if no device was explicitly specified so far, find best suitable CUDA device
        if (!cudDeviceIdGiven) {

            // find appropriate CUDA device ourself;
            logMessage(debug, true, "No (valid) device ID passed via command line. Determining suitable device... \n");
            cudDeviceId = findBestFreeDevice(0,0,0,0);
            if(cudDeviceId < 0) {
                logMessage(error, true, "No suitable CUDA device available!\n");
                return(RADPUL_CUDA_DEVICE_FIND);
            }
        }

        // update caller's device ID value
        *cudDeviceIdPtr = cudDeviceId;

        // Get handle for requested device
        cuResult = cuDeviceGet(&cuDevice, cudDeviceId);
        if(cuResult != CUDA_SUCCESS) {
            logMessage(error, true, "Couldn't acquire CUDA device #%i (error: %i)!\n", cudDeviceId, cuResult);
            return(RADPUL_CUDA_DEVICE_SET);
        }

        cudaDeviceNotSet = false;
    }

    // acquire device (set thread scheduling to yield/block during GPU execution: increases latency but reduces CPU usage -> BOINC!)
    cuResult = cuCtxCreate(&cuContext, CU_CTX_BLOCKING_SYNC, cudDeviceId);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Failed to enable CUDA thread yielding for device #%i (error: %i)! Sorry, will try to occupy one CPU core...\n",
                   cudDeviceId, cuResult);

        // retry with auto scheduling
        cuResult = cuCtxCreate(&cuContext, CU_CTX_SCHED_AUTO, cudDeviceId);
        if(cuResult != CUDA_SUCCESS) {
            logMessage(error, true, "Couldn't acquire CUDA context of device #%i (error: %i)!\n", cudDeviceId, cuResult);
            return(RADPUL_CUDA_DEVICE_SET);
        }
    }

    // show some details if possible
    int compcapMajor = 0;
    int compcapMinor = 0;
    int multiProcessorCount = 0;
    int coreCount = 0;
    int clockRate = 0;
    int flopsPerClockTick = 0;
    char deviceName[256] = {0};

    logMessage(info, true, "CUDA global memory status (initial GPU state, including context):\n");
    printDeviceGlobalMemStatus(info, true);

    // number of multi processors
    cuResult = cuDeviceGetAttribute(&multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuDevice);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(warn, true, "Couldn't retrieve multiprocessor count property of device #%i (error: %i)! Trying next one...\n", cudDeviceId, cuResult);
    }

    // clock rate
    cuResult = cuDeviceGetAttribute(&clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, cuDevice);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(warn, true, "Couldn't retrieve clock rate property of device #%i (error: %i)! Trying next one...\n", cudDeviceId, cuResult);
    }

    // compute capability
    cuResult = cuDeviceComputeCapability (&compcapMajor, &compcapMinor, cuDevice);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(warn, true, "Couldn't retrieve compute capability of device #%i (error: %i)! Trying next one...\n", cudDeviceId, cuResult);
    }

    // assign proper number of cores
    if(compcapMajor == 1) {
        coreCount = multiProcessorCount * 8;
        flopsPerClockTick = 3;
    }
    else if(compcapMajor == 2 && compcapMinor == 0) {
        coreCount = multiProcessorCount * 32;
        flopsPerClockTick = 2;
    }
    else if(compcapMajor == 2 && compcapMinor >= 0) {
        coreCount = multiProcessorCount * 48;
        flopsPerClockTick = 2;
    }
    else if(compcapMajor == 3) {
        coreCount = multiProcessorCount * 192;
        flopsPerClockTick = 2;
    }
    else if(compcapMajor == 5) {
        coreCount = multiProcessorCount * 128;
        flopsPerClockTick = 2;
    }


    // name
    cuResult = cuDeviceGetName(deviceName, 256, cuDevice);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(debug, true, "Couldn't retrieve name of device #%i (error: %i)!\n", cudDeviceId, cuResult);
        strcpy(deviceName, "UNKNOWN");
    }

    // check if our device is a "real" device
    if(compcapMajor == 9999 || compcapMinor == 9999) {
        logMessage(error, true, "Error acquiring \"real\" CUDA device!\n");
        logMessage(error, false, "The acquired device is a \"%s\"\n", deviceName);
        return(RADPUL_CUDA_EMULATION_MODE);
    }
    else {
        logMessage(info, true, "Using CUDA device #%i \"%s\" (%i CUDA cores / %.2f GFLOPS)\n",
                   cudDeviceId, deviceName, coreCount, coreCount * clockRate * flopsPerClockTick * 1e-6);
    }

    // determine CUDA driver version
    cuResult = cuDriverGetVersion(&cudDriverVersion);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(warn, true, "Couldn't retrieve CUDA driver version (error: %i)!\n", cuResult);
    }
    else {
        logMessage(info, true, "Version of installed CUDA driver: %i\n", cudDriverVersion);
    }

    // determine CUDA driver API version
    cuResult = cuCtxGetApiVersion(NULL, &cudApiVersion);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(warn, true, "Couldn't retrieve CUDA driver API version (error: %i)!\n", cuResult);
    }
    else {
        logMessage(info, true, "Version of CUDA driver API used: %u\n", cudApiVersion);
    }

    // load device modules / kernels
    char modulePath[1024] = {0};
    i = resolveFilename("db.dev", modulePath, 1023);
    if(i) {
        logMessage(error, true, "Couldn't retrieve main CUDA device module path (error: %i)!\n", i);
        return(RADPUL_CUDA_LOAD_MODULE);
    }
    cuResult = cuModuleLoad(&cuModuleMain, modulePath);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Couldn't load main CUDA device module (error: %i)!\n", cuResult);
        return(RADPUL_CUDA_LOAD_MODULE);
    }

    cuResult = cuModuleGetFunction(&kernelTimeSeriesModulation, cuModuleMain, "time_series_modulation");
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Couldn't get CUDA TSM kernel handle (error: %i)!\n", cuResult);
        return(RADPUL_CUDA_LOOKUP_KERNEL);
    }

    cuResult = cuModuleGetFunction(&kernelTimeSeriesLengthModulated, cuModuleMain, "time_series_length_modulated");
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Couldn't get CUDA TSLM kernel handle (error: %i)!\n", cuResult);
        return(RADPUL_CUDA_LOOKUP_KERNEL);
    }

    cuResult = cuModuleGetFunction(&kernelTimeSeriesResampling, cuModuleMain, "time_series_resampling");
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Couldn't get CUDA TSR kernel handle (error: %i)!\n", cuResult);
        return(RADPUL_CUDA_LOOKUP_KERNEL);
    }

    cuResult = cuModuleGetFunction(&kernelTimeSeriesMeanReduction, cuModuleMain, "time_series_mean_reduction");
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Couldn't get CUDA TSMR kernel handle (error: %i)!\n", cuResult);
        return(RADPUL_CUDA_LOOKUP_KERNEL);
    }

    cuResult = cuModuleGetFunction(&kernelTimeSeriesPadding, cuModuleMain, "time_series_padding");
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Couldn't get CUDA TSP kernel handle (error: %i)!\n", cuResult);
        return(RADPUL_CUDA_LOOKUP_KERNEL);
    }

    cuResult = cuModuleGetFunction(&kernelPowerSpectrum, cuModuleMain, "fft_powerspectrum");
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Couldn't get CUDA PS kernel handle (error: %i)!\n", cuResult);
        return(RADPUL_CUDA_LOOKUP_KERNEL);
    }

    return 0;
}


int set_up_resampling(DIfloatPtr input_dip, DIfloatPtr *output_dip, const RESAMP_PARAMS *const params, float *sinLUTsamples, float *cosLUTsamples)
{
    float * input = input_dip.host_ptr; // original time series is on host

    CUdeviceptr resampledTimeSeriesDeviceBuffer;   // Resampled time series device buffer (also used for FFT in-/output)

    // sanity check
    if(params->nsamples_unpadded % CUDA_RESAMP_REDUCTION_BLOCKDIM_X != 0) {
        logMessage(error, true, "The time series length %i isn't an integer multiple of the CUDA block size %i!\n", params->nsamples_unpadded, CUDA_RESAMP_REDUCTION_BLOCKDIM_X);
        return(RADPUL_EVAL);
    }

    // allocate device memory for original time series
    cuResult = cuMemAlloc(&originalTimeSeriesDeviceBuffer, sizeof(float) * params->nsamples_unpadded);
    if(cuResult != CUDA_SUCCESS)
    {
        logMessage(error, true, "Error allocating original time series device memory: %i bytes (error: %i)\n", sizeof(float) * params->nsamples_unpadded, cuResult);
        return(RADPUL_CUDA_MEM_ALLOC_DEVICE);
    }
    logMessage(debug, true, "Allocated original time series device memory: %i bytes\n", sizeof(float) * params->nsamples_unpadded);

    // allocate device memory for modulated time offsets
    cuResult = cuMemAlloc(&modTimeOffsetsDeviceBuffer, sizeof(float) * params->nsamples_unpadded);
    if(cuResult != CUDA_SUCCESS)
    {
        logMessage(error, true, "Error allocating modulated time offsets device memory: %i bytes (error: %i)\n", sizeof(float) * params->nsamples_unpadded, cuResult);
        return(RADPUL_CUDA_MEM_ALLOC_DEVICE);
    }
    logMessage(debug, true, "Allocated modulated time offsets device memory: %i bytes\n", sizeof(float) * params->nsamples_unpadded);

    // increase FFT buffer length such that it matches the powerspectrum kernel's blocklength (no further control flow required in kernel)
    const unsigned int fft_size_padded = PADDED_FFT_SIZE(params->fft_size);

    // allocate device memory for resampled time series (we use cufftComplex*fft_size_padded here as we reuse the buffer later on for the FFT)
    cuResult = cuMemAlloc(&resampledTimeSeriesDeviceBuffer, sizeof(cufftComplex) * fft_size_padded);
    if(cuResult != CUDA_SUCCESS)
    {
        logMessage(error, true, "Error allocating modulated time series device memory: %i bytes (error: %d)\n", sizeof(cufftComplex) * fft_size_padded, cuResult);
        return(RADPUL_CUDA_MEM_ALLOC_DEVICE);
    }
    logMessage(debug, true, "Allocated modulated time series device memory: %i bytes\n", sizeof(cufftComplex) * fft_size_padded);

    // allocate device memory for time series mean sum reduction (we need two separate buffers, each half used alternately)
    cuResult = cuMemAlloc(&timeSeriesMeanDeviceBuffer, sizeof(float) * (params->nsamples_unpadded/CUDA_RESAMP_REDUCTION_BLOCKDIM_X*2));
    if(cuResult != CUDA_SUCCESS)
    {
        logMessage(error, true, "Error allocating modulated time series mean reduction device memory: %i bytes (error: %i)\n", sizeof(float) * (params->nsamples_unpadded/CUDA_RESAMP_REDUCTION_BLOCKDIM_X*2), cuResult);
        return(RADPUL_CUDA_MEM_ALLOC_DEVICE);
    }
    logMessage(debug, true, "Allocated time series mean reduction device memory: %i bytes\n", sizeof(float) * (params->nsamples_unpadded/CUDA_RESAMP_REDUCTION_BLOCKDIM_X*2));

    // transfer original time series data to device
    cuResult = cuMemcpyHtoD(originalTimeSeriesDeviceBuffer, input, sizeof(float) * params->nsamples_unpadded);
    if(cuResult != CUDA_SUCCESS)
    {
        logMessage(error, true, "Error during CUDA host->device original time series data transfer (error: %i)\n", cuResult);
        return(RADPUL_CUDA_MEM_COPY_HOST_DEVICE);
    }
    logMessage(debug, true, "CUDA host->device original time series data transfer successful...\n");

    // lookup constant memory symbols
    CUdeviceptr cudSymbol;

    // transfer sin lookup table data to device
    cuResult = cuModuleGetGlobal(&cudSymbol, NULL, cuModuleMain, "constSinSamples");
    if(cuResult != CUDA_SUCCESS) {
            logMessage(error, true, "Couldn't get CUDA CSS symbol handle (error: %i)!\n", cuResult);
            return(RADPUL_CUDA_LOOKUP_SYMBOL);
    }
    cuResult = cuMemcpyHtoD(cudSymbol, sinLUTsamples, ERP_SINCOS_LUT_SIZE * sizeof(float));
    if(cuResult != CUDA_SUCCESS)
    {
        logMessage(error, true, "Error during CUDA host->device sin lookup table data transfer (error: %i)\n", cuResult);
        return(RADPUL_CUDA_MEM_COPY_HOST_DEVICE);
    }
    logMessage(debug, true, "CUDA host->device sin lookup table data transfer successful...\n");

    // transfer cos lookup table data to device
    cuResult = cuModuleGetGlobal(&cudSymbol, NULL, cuModuleMain, "constCosSamples");
    if(cuResult != CUDA_SUCCESS) {
            logMessage(error, true, "Couldn't get CUDA CCS symbol handle (error: %i)!\n", cuResult);
            return(RADPUL_CUDA_LOOKUP_SYMBOL);
    }
    cuResult = cuMemcpyHtoD(cudSymbol, cosLUTsamples, ERP_SINCOS_LUT_SIZE * sizeof(float));
    if(cuResult != CUDA_SUCCESS)
    {
        logMessage(error, true, "Error during CUDA host->device cos lookup table data transfer (error: %d)\n", cuResult);
        return(RADPUL_CUDA_MEM_COPY_HOST_DEVICE);
    }
    logMessage(debug, true, "CUDA host->device cos lookup table data transfer successful...\n");

    // transfer lookup table parameters to device
    float lutParam = ERP_TWO_PI;
    cuResult = cuModuleGetGlobal(&cudSymbol, NULL, cuModuleMain, "LUT_TWO_PI");
    if(cuResult != CUDA_SUCCESS) {
            logMessage(error, true, "Couldn't get CUDA LTP symbol handle (error: %i)!\n", cuResult);
            return(RADPUL_CUDA_LOOKUP_SYMBOL);
    }
    cuResult = cuMemcpyHtoD(cudSymbol, &lutParam, sizeof(float));
    if(cuResult != CUDA_SUCCESS)
    {
        logMessage(error, true, "Error during CUDA host->device lookup table parameter transfer (error: %d)\n", cuResult);
        return(RADPUL_CUDA_MEM_COPY_HOST_DEVICE);
    }
    logMessage(debug, true, "CUDA host->device lookup table parameter transfer successful...\n");

    lutParam = ERP_TWO_PI_INV;
    cuResult = cuModuleGetGlobal(&cudSymbol, NULL, cuModuleMain, "LUT_TWO_PI_INV");
    if(cuResult != CUDA_SUCCESS) {
            logMessage(error, true, "Couldn't get CUDA LTPI symbol handle (error: %i)!\n", cuResult);
            return(RADPUL_CUDA_LOOKUP_SYMBOL);
    }
    cuResult = cuMemcpyHtoD(cudSymbol, &lutParam, sizeof(float));
    if(cuResult != CUDA_SUCCESS)
    {
        logMessage(error, true, "Error during CUDA host->device lookup table parameter transfer (error: %d)\n", cuResult);
        return(RADPUL_CUDA_MEM_COPY_HOST_DEVICE);
    }
    logMessage(debug, true, "CUDA host->device lookup table parameter transfer successful...\n");

    // retrieve global symbol used later on
    cuResult = cuModuleGetGlobal(&timeSeriesLengthDeviceBuffer, NULL, cuModuleMain, "timeSeriesLength");
    if(cuResult != CUDA_SUCCESS) {
            logMessage(error, true, "Couldn't get CUDA TSL symbol handle (error: %i)!\n", cuResult);
            return(RADPUL_CUDA_LOOKUP_SYMBOL);
    }

    // return allocated device pointer to caller

    output_dip->device_ptr=resampledTimeSeriesDeviceBuffer;

    return 0;
}


int run_resampling(DIfloatPtr input_dip, DIfloatPtr output_dip, const RESAMP_PARAMS *const params)
{
    // unused (doesn't prevent nvcc warnings, oh well)
    float * input = NULL;

    CUdeviceptr resampledTimeSeriesDeviceBuffer = output_dip.device_ptr;

    // kernel parameter offset counter (used per kernel launch, reset to 0 accordingly!)
    int kernelParamOffset = 0;

    // output variables
    int n_steps = 0;
    float mean = 0.0f;

    // compute time offsets

    dim3 dimBlockResampOffsets(CUDA_RESAMP_OFFSETS_BLOCKDIM_X);
    dim3 dimGridResampOffsets(params->nsamples_unpadded / dimBlockResampOffsets.x);

    logMessage(debug, true, "Executing time series modulation CUDA kernel (%u threads each in %u blocks)...\n", dimBlockResampOffsets.x, dimGridResampOffsets.x);

    // prepare parameters
    KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(modTimeOffsetsDeviceBuffer));
    cuResult = cuParamSetv(kernelTimeSeriesModulation, kernelParamOffset, &modTimeOffsetsDeviceBuffer, sizeof(modTimeOffsetsDeviceBuffer));
    kernelParamOffset += sizeof(modTimeOffsetsDeviceBuffer);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error during CUDA TSM kernel parameter setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(params->tau));
    cuResult = cuParamSetf(kernelTimeSeriesModulation, kernelParamOffset, params->tau);
    kernelParamOffset += sizeof(params->tau);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error during CUDA TSM kernel parameter setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(params->Omega));
    cuResult = cuParamSetf(kernelTimeSeriesModulation, kernelParamOffset, params->Omega);
    kernelParamOffset += sizeof(params->Omega);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error during CUDA TSM kernel parameter setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(params->Psi0));
    cuResult = cuParamSetf(kernelTimeSeriesModulation, kernelParamOffset, params->Psi0);
    kernelParamOffset += sizeof(params->Psi0);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error during CUDA TSM kernel parameter setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(params->dt));
    cuResult = cuParamSetf(kernelTimeSeriesModulation, kernelParamOffset, params->dt);
    kernelParamOffset += sizeof(params->dt);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error during CUDA TSM kernel parameter setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(params->step_inv));
    cuResult = cuParamSetf(kernelTimeSeriesModulation, kernelParamOffset, params->step_inv);
    kernelParamOffset += sizeof(params->step_inv);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error during CUDA TSM kernel parameter setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(params->S0));
    cuResult = cuParamSetf(kernelTimeSeriesModulation, kernelParamOffset, params->S0);
    kernelParamOffset += sizeof(params->S0);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error during CUDA TSM kernel parameter setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    cuResult = cuParamSetSize(kernelTimeSeriesModulation, kernelParamOffset);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error finalizing CUDA TSM kernel parameter setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    // prepare block size (CUDA_RESAMP_OFFSETS_BLOCKDIM_X threads per block (1D))
    cuResult = cuFuncSetBlockShape(kernelTimeSeriesModulation, dimBlockResampOffsets.x, dimBlockResampOffsets.y, dimBlockResampOffsets.z);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error during CUDA TSM kernel block setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    // launch kernel grid (n_unpadded/CUDA_RESAMP_OFFSETS_BLOCKDIM_X blocks in 1D grid)
    cuResult = cuLaunchGrid(kernelTimeSeriesModulation, dimGridResampOffsets.x, dimGridResampOffsets.y);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error launching CUDA TSM kernel (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_INVOKE);
    }

    // determine modulated time series length (done in kernel because global sync required and to avoid bus transfer of del_t as well as waste of host memory)

    dim3 dimBlockResampLength(1); // single thread per block (1D)
    dim3 dimGridResampLength(1);  // single block in grid (1D)

    logMessage(debug, true, "Executing modulated time series length CUDA kernel (%u threads each in %u blocks)...\n", dimBlockResampLength.x, dimGridResampLength.x);

    // prepare parameters
    kernelParamOffset = 0;
    KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(modTimeOffsetsDeviceBuffer));
    cuResult = cuParamSetv(kernelTimeSeriesLengthModulated, kernelParamOffset, &modTimeOffsetsDeviceBuffer, sizeof(modTimeOffsetsDeviceBuffer));
    kernelParamOffset += sizeof(modTimeOffsetsDeviceBuffer);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error during CUDA TSLM kernel parameter setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(params->nsamples_unpadded));
    cuResult = cuParamSeti(kernelTimeSeriesLengthModulated, kernelParamOffset, params->nsamples_unpadded);
    kernelParamOffset += sizeof(params->nsamples_unpadded);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error during CUDA TSLM kernel parameter setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    cuResult = cuParamSetSize(kernelTimeSeriesLengthModulated, kernelParamOffset);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error finalizing CUDA TSLM kernel parameter setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    // prepare block size (one single thread)
     cuResult = cuFuncSetBlockShape(kernelTimeSeriesLengthModulated, dimBlockResampLength.x, dimBlockResampLength.y, dimBlockResampLength.z);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error during CUDA TSLM kernel block setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    // launch kernel grid (one single block)
    cuResult = cuLaunchGrid(kernelTimeSeriesLengthModulated, dimGridResampLength.x, dimGridResampLength.y);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error launching CUDA TSLM kernel (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_INVOKE);
    }

    // return computed time series length
    cuResult = cuMemcpyDtoH(&n_steps, timeSeriesLengthDeviceBuffer, sizeof(int));
    if(cuResult != CUDA_SUCCESS)
    {
        logMessage(error, true, "Error during CUDA device->host time series length transfer (error: %d)\n", cuResult);
        return(RADPUL_CUDA_MEM_COPY_DEVICE_HOST);
    }
    logMessage(debug, true, "CUDA device->host time series length (%d) transfer successful...\n", n_steps);

    // compute resampled time series (unpadded)

    dim3 dimBlockResamp(CUDA_RESAMP_BLOCKDIM_X);             // CUDA_RESAMP_BLOCKDIM_X threads per block (1D)
    dim3 dimGridResamp(params->nsamples / dimBlockResamp.x); // nsamples/CUDA_RESAMP_BLOCKDIM_X blocks in grid (1D)

    logMessage(debug, true, "Executing time series resampling CUDA kernel (%u threads each in %u blocks)...\n", dimBlockResamp.x, dimGridResamp.x);

    // prepare parameters
    kernelParamOffset = 0;
    KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(originalTimeSeriesDeviceBuffer));
    cuResult = cuParamSetv(kernelTimeSeriesResampling, kernelParamOffset, &originalTimeSeriesDeviceBuffer, sizeof(originalTimeSeriesDeviceBuffer));
    kernelParamOffset += sizeof(originalTimeSeriesDeviceBuffer);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error during CUDA TSR kernel parameter setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(modTimeOffsetsDeviceBuffer));
    cuResult = cuParamSetv(kernelTimeSeriesResampling, kernelParamOffset, &modTimeOffsetsDeviceBuffer, sizeof(modTimeOffsetsDeviceBuffer));
    kernelParamOffset += sizeof(modTimeOffsetsDeviceBuffer);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error during CUDA TSR kernel parameter setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(resampledTimeSeriesDeviceBuffer));
    cuResult = cuParamSetv(kernelTimeSeriesResampling, kernelParamOffset, &resampledTimeSeriesDeviceBuffer, sizeof(resampledTimeSeriesDeviceBuffer));
    kernelParamOffset += sizeof(resampledTimeSeriesDeviceBuffer);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error during CUDA TSR kernel parameter setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(timeSeriesMeanDeviceBuffer));
    cuResult = cuParamSetv(kernelTimeSeriesResampling, kernelParamOffset, &timeSeriesMeanDeviceBuffer, sizeof(timeSeriesMeanDeviceBuffer));
    kernelParamOffset += sizeof(timeSeriesMeanDeviceBuffer);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error during CUDA TSR kernel parameter setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(params->nsamples_unpadded));
    cuResult = cuParamSeti(kernelTimeSeriesResampling, kernelParamOffset, params->nsamples_unpadded);
    kernelParamOffset += sizeof(params->nsamples_unpadded);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error during CUDA TSR kernel parameter setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(n_steps));
    cuResult = cuParamSeti(kernelTimeSeriesResampling, kernelParamOffset, n_steps);
    kernelParamOffset += sizeof(n_steps);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error during CUDA TSR kernel parameter setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    cuResult = cuParamSetSize(kernelTimeSeriesResampling, kernelParamOffset);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error finalizing CUDA TSR kernel parameter setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    // prepare block size (CUDA_RESAMP_BLOCKDIM_X threads per block (1D))
     cuResult = cuFuncSetBlockShape(kernelTimeSeriesResampling, dimBlockResamp.x, dimBlockResamp.y, dimBlockResamp.z);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error during CUDA TSR kernel block setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    // launch kernel grid (params->nsamples_unpadded / CUDA_RESAMP_BLOCKDIM_X blocks in 1D grid)
    cuResult = cuLaunchGrid(kernelTimeSeriesResampling, dimGridResamp.x, dimGridResamp.y);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error launching CUDA TSR kernel (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_INVOKE);
    }

    // compute time series mean value

    // TODO: n_steps (effective length) threads would by sufficient
    int threadsPerBlockX = CUDA_RESAMP_REDUCTION_BLOCKDIM_X;  // CUDA_RESAMP_REDUCTION_BLOCKDIM_X threads per block (1D)

    // sum reduction loop control variables
    CUdeviceptr currentInputBuffer = NULL;
    CUdeviceptr currentOutputBuffer = NULL;
    int requiredBlocks = params->nsamples_unpadded / CUDA_RESAMP_REDUCTION_BLOCKDIM_X;
    int secondHalfOffset = requiredBlocks;
    bool useSecondHalfForOutput = false;
    int i = 1;

    do {
        // use kernel decomposition to do sum reduction (facilitates global memory sync, kernel invocations are cheap)
        logMessage(debug, true, "Executing time series mean reduction CUDA kernel (iteration %i using %i blocks of %u threads)...\n", i, requiredBlocks, threadsPerBlockX);

        // only the first iteration uses the resampled time series as input (obviously)
        if(i==1) {
            currentInputBuffer = resampledTimeSeriesDeviceBuffer;
        }
        else {
            // otherwise: alternate between first and second half of mean device buffer for input (inversely to output buffer)
            currentInputBuffer = (CUdeviceptr) ((float*)timeSeriesMeanDeviceBuffer + (useSecondHalfForOutput ? 0 : secondHalfOffset));
        }

        // alternate between first and second half of mean device buffer for output (inversely to input buffer)
        currentOutputBuffer = (CUdeviceptr) ((float*)timeSeriesMeanDeviceBuffer + (useSecondHalfForOutput ? secondHalfOffset : 0));

        // prepare parameters
        kernelParamOffset = 0;
        KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(currentInputBuffer));
        cuResult = cuParamSetv(kernelTimeSeriesMeanReduction, kernelParamOffset, &currentInputBuffer, sizeof(currentInputBuffer));
        kernelParamOffset += sizeof(currentInputBuffer);
        if(cuResult != CUDA_SUCCESS) {
            logMessage(error, true, "Error during CUDA TSMR-%i kernel parameter setup (error: %d)\n", i, cuResult);
            return(RADPUL_CUDA_KERNEL_PREPARE);
        }

        KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(currentOutputBuffer));
        cuResult = cuParamSetv(kernelTimeSeriesMeanReduction, kernelParamOffset, &currentOutputBuffer, sizeof(currentOutputBuffer));
        kernelParamOffset += sizeof(currentOutputBuffer);
        if(cuResult != CUDA_SUCCESS) {
            logMessage(error, true, "Error during CUDA TSMR-%i kernel parameter setup (error: %d)\n", i, cuResult);
            return(RADPUL_CUDA_KERNEL_PREPARE);
        }

        cuResult = cuParamSetSize(kernelTimeSeriesMeanReduction, kernelParamOffset);
        if(cuResult != CUDA_SUCCESS) {
            logMessage(error, true, "Error finalizing CUDA TSMR-%i kernel parameter setup (error: %d)\n", i, cuResult);
            return(RADPUL_CUDA_KERNEL_PREPARE);
        }

        // prepare block size (CUDA_RESAMP_REDUCTION_BLOCKDIM_X)
        cuResult = cuFuncSetBlockShape(kernelTimeSeriesMeanReduction, threadsPerBlockX, 1, 1);
        if(cuResult != CUDA_SUCCESS) {
            logMessage(error, true, "Error during CUDA TSMR-%i kernel block setup (error: %d)\n", i, cuResult);
            return(RADPUL_CUDA_KERNEL_PREPARE);
        }

        // launch kernel grid (iteratively reduced number of threads)
        cuResult = cuLaunchGrid(kernelTimeSeriesMeanReduction, requiredBlocks, 1);
        if(cuResult != CUDA_SUCCESS) {
            logMessage(error, true, "Error launching CUDA TSMR-%i kernel (error: %d)\n", i, cuResult);
            return(RADPUL_CUDA_KERNEL_INVOKE);
        }

        // required blocks for next iteration
        if (requiredBlocks >= CUDA_RESAMP_REDUCTION_BLOCKDIM_X) {
            // we still can fill full blocks
            requiredBlocks /= CUDA_RESAMP_REDUCTION_BLOCKDIM_X;
        }
        else {
            if (requiredBlocks == 1) {
                // this was the final summing by the last block
                break;
            }
            else {
                // we're now within the last block (with fewer than blocksize elements), so sum pairs with one thread each
                threadsPerBlockX = requiredBlocks;
                requiredBlocks = 1;
            }
        }

        // flip output buffer specifier
        useSecondHalfForOutput = useSecondHalfForOutput ? false : true;

        // update progress counter
        i++;
    }
    while(requiredBlocks > 0);

    // return computed time series mean (first element of output buffer)
    cuResult = cuMemcpyDtoH(&mean, currentOutputBuffer, sizeof(float));
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error during CUDA device->host time series mean transfer (error: %i)\n", cuResult);
        return(RADPUL_CUDA_MEM_COPY_DEVICE_HOST);
    }
    logMessage(debug, true, "CUDA device->host time series sum (%f) transfer successful...\n", mean);

    // compute actual mean
    mean /= n_steps;

    logMessage(debug, true, "Actual time series mean is: %e\n", mean);

    // apply mean padding to time series

    // TODO: params->nsamples-n_steps threads would be sufficient
    dim3 dimBlockResampPadding(CUDA_RESAMP_PADDING_BLOCKDIM_X);            // CUDA_RESAMP_PADDING_BLOCKDIM_X threads per block (1D)
    dim3 dimGridResampPadding(params->nsamples / dimBlockResampPadding.x); // nsamples/CUDA_RESAMP_BLOCKDIM_X blocks in grid (1D)

    logMessage(debug, true, "Executing time series padding CUDA kernel (%u threads each in %u blocks)...\n", dimBlockResampPadding.x, dimGridResampPadding.x);

    // prepare parameters
    kernelParamOffset = 0;
    KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(resampledTimeSeriesDeviceBuffer));
    cuResult = cuParamSetv(kernelTimeSeriesPadding, kernelParamOffset, &resampledTimeSeriesDeviceBuffer, sizeof(resampledTimeSeriesDeviceBuffer));
    kernelParamOffset += sizeof(resampledTimeSeriesDeviceBuffer);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error during CUDA TSR kernel parameter setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(mean));
    cuResult = cuParamSetf(kernelTimeSeriesPadding, kernelParamOffset, mean);
    kernelParamOffset += sizeof(mean);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error during CUDA TSP kernel parameter setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(n_steps));
    cuResult = cuParamSeti(kernelTimeSeriesPadding, kernelParamOffset, n_steps);
    kernelParamOffset += sizeof(n_steps);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error during CUDA TSP kernel parameter setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    cuResult = cuParamSetSize(kernelTimeSeriesPadding, kernelParamOffset);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error finalizing CUDA TSP kernel parameter setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    // prepare block size (CUDA_RESAMP_PADDING_BLOCKDIM_X)
     cuResult = cuFuncSetBlockShape(kernelTimeSeriesPadding, dimBlockResampPadding.x, dimBlockResampPadding.y, dimBlockResampPadding.z);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error during CUDA TSP kernel block setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    // launch kernel grid (params->nsamples / CUDA_RESAMP_PADDING_BLOCKDIM_X blocks in 1D grid)
    cuResult = cuLaunchGrid(kernelTimeSeriesPadding, dimGridResampPadding.x, dimGridResampPadding.y);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error launching CUDA TSP kernel (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_INVOKE);
    }

    return 0;
}


int tear_down_resampling(DIfloatPtr output)
{
    CUdeviceptr resampledTimeSeriesDeviceBuffer = output.device_ptr;

    cuResult = cuMemFree(resampledTimeSeriesDeviceBuffer);
    if(cuResult != CUDA_SUCCESS)
    {
        logMessage(error, true, "Error deallocating resampled time series device memory (error: %i)\n", cuResult);
        cuMemFree(timeSeriesMeanDeviceBuffer);
        return(RADPUL_CUDA_MEM_FREE_DEVICE);
    }

    cuResult = cuMemFree(originalTimeSeriesDeviceBuffer);
    if(cuResult != CUDA_SUCCESS)
    {
        logMessage(error, true, "Error deallocating original time series device memory (error: %i)\n", cuResult);
        cuMemFree(modTimeOffsetsDeviceBuffer);
        cuMemFree(timeSeriesMeanDeviceBuffer);
        return(RADPUL_CUDA_MEM_FREE_DEVICE);
    }

    cuResult = cuMemFree(modTimeOffsetsDeviceBuffer);
    if(cuResult != CUDA_SUCCESS)
    {
        logMessage(error, true, "Error deallocating modulated time offsets device memory (error: %i)\n", cuResult);
        cuMemFree(timeSeriesMeanDeviceBuffer);
        return(RADPUL_CUDA_MEM_FREE_DEVICE);
    }


    cuResult = cuMemFree(timeSeriesMeanDeviceBuffer);
    if(cuResult != CUDA_SUCCESS)
    {
        logMessage(error, true, "Error deallocating time series mean reduction device memory (error: %i)\n", cuResult);
        return(RADPUL_CUDA_MEM_FREE_DEVICE);
    }

    return 0;
}


int set_up_fft(DIfloatPtr input_dip, DIfloatPtr *output_dip, uint32_t nsamples, unsigned int fft_size)
{

    // unused (doesn't prevent nvcc warnings, oh well)
    float * input = NULL;
    CUdeviceptr * output =&(output_dip->device_ptr); // powerspectrum on device memory

    // increase powerspectrum buffer length such that it matches the powerspectrum kernel's blocklength (no further control flow required in kernel)
    const unsigned int fft_size_padded = PADDED_FFT_SIZE(fft_size);

    logMessage(debug, true, "Padding output size of FFT with %u samples from %u to %u...\n", nsamples, fft_size,
fft_size_padded);

    // create fft plan
    cufResult = cufftPlan1d(&cufPlan, nsamples, CUFFT_R2C, 1);
    if(cufResult != CUFFT_SUCCESS)
    {
        logMessage(error, true, "Error creating CUDA FFT plan (error code: %i)\n", cufResult);
        return(RADPUL_CUDA_FFT_PLAN);
    }
    logMessage(debug, true, "Created CUFFT plan...\n");

    // ensure FFTW compatibility
    cufResult = cufftSetCompatibilityMode(cufPlan, CUFFT_COMPATIBILITY_FFTW_ALL);
    if(cufResult != CUFFT_SUCCESS)
    {
            logMessage(error, true, "Error setting CUDA FFTW compatibility (error code: %i)\n", cufResult);
            return(RADPUL_CUDA_FFT_PLAN);
    }


    // allocate device memory for power spectrum
    cuResult = cuMemAlloc(output, sizeof(float) * fft_size_padded);
    if(cuResult != CUDA_SUCCESS)
    {
        logMessage(error, true, "Error allocating power spectrum device memory: %i bytes (error: %i)\n", sizeof(float) * fft_size_padded, cuResult);
        return(RADPUL_CUDA_MEM_ALLOC_DEVICE);
    }
    logMessage(debug, true, "Allocated power spectrum device memory: %i bytes\n", sizeof(float) * fft_size_padded);

    return 0;
}


int run_fft(DIfloatPtr input, DIfloatPtr output, uint32_t nsamples, unsigned int fft_size, float norm_factor)
{
    CUdeviceptr psDeviceBuffer=output.device_ptr;
    CUdeviceptr resampledTimeSeriesDeviceBuffer=input.device_ptr;

    // increase number of powerspectrum threads such that all blocks/threads are used completely (no further control flow required in kernel)
    static const unsigned int fft_size_padded = PADDED_FFT_SIZE(fft_size);

    // execute FFT
    cufResult = cufftExecR2C(cufPlan, (cufftReal*)resampledTimeSeriesDeviceBuffer, (cufftComplex*)resampledTimeSeriesDeviceBuffer);
    if(cufResult != CUFFT_SUCCESS)
    {
        logMessage(error, true, "Error executing CUDA FFT plan (error code: %i)\n", cufResult);
        cufftDestroy(cufPlan);
        return(RADPUL_CUDA_FFT_EXEC);
    }
    logMessage(debug, true, "CUDA FFT execution successful...\n");

    // compute powerspectrum using CUDA kernel
    dim3 dimBlockFFT(CUDA_FFT_BLOCKDIM_X);            // CUDA_BLOCKDIM_X threads per block (1D)
    dim3 dimGridFFT(fft_size_padded / dimBlockFFT.x); // (fft_size_padded/CUDA_FFT_BLOCKDIM_X) blocks in grid (1D)

    logMessage(debug, true, "Executing power spectrum CUDA kernel (%u threads each in %u blocks)...\n", dimBlockFFT.x, dimGridFFT.x);

    // prepare parameters
    int kernelParamOffset = 0;

    KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(resampledTimeSeriesDeviceBuffer));
    cuResult = cuParamSetv(kernelPowerSpectrum, kernelParamOffset, &resampledTimeSeriesDeviceBuffer, sizeof(resampledTimeSeriesDeviceBuffer));
    kernelParamOffset += sizeof(resampledTimeSeriesDeviceBuffer);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error during CUDA PS kernel parameter setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(psDeviceBuffer));
    cuResult = cuParamSetv(kernelPowerSpectrum, kernelParamOffset, &psDeviceBuffer, sizeof(psDeviceBuffer));
    kernelParamOffset += sizeof(psDeviceBuffer);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error during CUDA PS kernel parameter setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(norm_factor));
    cuResult = cuParamSetf(kernelPowerSpectrum, kernelParamOffset, norm_factor);
    kernelParamOffset += sizeof(norm_factor);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error during CUDA PS kernel parameter setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    cuResult = cuParamSetSize(kernelPowerSpectrum, kernelParamOffset);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error finalizing CUDA PS kernel parameter setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    // prepare block size (CUDA_FFT_BLOCKDIM_X)
     cuResult = cuFuncSetBlockShape(kernelPowerSpectrum, dimBlockFFT.x, dimBlockFFT.y, dimBlockFFT.z);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error during CUDA PS kernel block setup (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_PREPARE);
    }

    // launch kernel grid
    cuResult = cuLaunchGrid(kernelPowerSpectrum, dimGridFFT.x, dimGridFFT.y);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error launching CUDA PS kernel (error: %d)\n", cuResult);
        return(RADPUL_CUDA_KERNEL_INVOKE);
    }

    return 0;
}


int tear_down_fft(DIfloatPtr output_dip)
{
    CUdeviceptr psDeviceBuffer = output_dip.device_ptr;

    cuResult = cuMemFree(psDeviceBuffer);
    if(cuResult != CUDA_SUCCESS)
    {
        logMessage(error, true, "Error deallocating power spectrum device memory (error: %i)\n", cuResult);
        cufftDestroy(cufPlan);
        return(RADPUL_CUDA_MEM_FREE_DEVICE);
    }

    cufResult = cufftDestroy(cufPlan);
    if(cufResult != CUFFT_SUCCESS)
    {
        logMessage(error, true, "Error destroying CUDA FFT plan (error code: %i)\n", cufResult);
        return(RADPUL_CUDA_FFT_DESTROY);
    }

    return 0;
}


int shutdown_cuda()
{
    // destroy context
    cuResult = cuCtxDestroy(cuContext);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(warn, true, "Couldn't destroy CUDA context (error: %i)!\n", cuResult);
    }

    logMessage(debug, true, "CUDA shutdown successful...\n");

    return 0;
}
