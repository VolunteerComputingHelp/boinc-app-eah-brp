/***************************************************************************
 *   Copyright (C) 2011,2012 by                                            *
 *   Oliver Bock                oliver.bock[AT]aei.mpg.de                  *
 *   Heinz-Bernd Eggenstein     heinz-bernd.eggenstein[AT]aei.mpg.de       *
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

#include "demod_binary_ocl.h"
#include "demod_binary_ocl_kernels.h"

#include <cstring>
#include <cmath>
#include <cstdlib>

#include <clFFT.h>

#include "ocl_utilities.h"
#include "../../demod_binary.h"
#include "../../erp_utilities.h"
#include "../../hs_common.h"

#define OCL_MAX_STRING 256
#define VENDOR_AMD     1
#define VENDOR_NVIDIA  2

// macro for fft padding (based on powerspectrum kernel's workgroup size, defined in demod_binary_ocl_kernels.h)
#define PADDED_FFT_SIZE(fftsize) workGroupSizePS[0] * (unsigned int) ceil((float)fftsize / (float)workGroupSizePS[0])

// globals
size_t workGroupSizeTSMR[] = {OCL_RESAMP_REDUCTION_WGSIZE_X};
size_t workGroupSizePS[] = {OCL_PS_WGSIZE_X};
size_t workGroupSizePS_R3_R2C[]={OCL_PS_R3_R2C_WGSIZE_X};
size_t workGroupSizeHS[] = {OCL_HS_WGSIZE_X, 1};
size_t workGroupSizeHSG[] = {OCL_HS_WGSIZE_X / 2, 1}; // work group size MUST be half than that of the first kernel

cl_context oclContext;
cl_command_queue oclQueue;
cl_program oclProgramDemodBinary;
cl_kernel oclKernelTimeSeriesModulation;
cl_kernel oclKernelTimeSeriesLengthModulated;
cl_kernel oclKernelTimeSeriesResampling;
cl_kernel oclKernelTimeSeriesMeanReduction;
cl_kernel oclKernelTimeSeriesPadding;
cl_kernel oclKernelTimeSeriesPaddingTranspose;
cl_kernel oclKernelFillFloatBuffer;
cl_kernel oclKernelFillIntBuffer;
cl_kernel oclKernelPowerSpectrum;
cl_kernel oclKernelPowerSpectrum_radix3_r2c;
cl_kernel oclKernelHarmonicSumming;
cl_kernel oclKernelHarmonicSummingGaps;

cl_mem originalTimeSeriesDeviceBuffer;
cl_mem sinLUTDeviceBuffer;
cl_mem cosLUTDeviceBuffer;
cl_mem modTimeOffsetsDeviceBuffer;
cl_mem timeSeriesLengthDeviceBuffer;
cl_mem timeSeriesMeanDeviceBuffer;
cl_mem fftComplexDeviceBuffer;
cl_mem fftTwiddleFactorDeviceBuffer;
cl_mem fftTwiddleFactor_r2cDeviceBuffer;
clFFT_Plan clFftPlan;
cl_mem harmomicSumminghLUTDeviceBuffer;
cl_mem harmomicSummingkLUTDeviceBuffer;
cl_mem harmomicSummingThresholdsDeviceBuffer;
cl_mem sumspecDev[5] ;      // an array of device memory pointers (NOT an array on the device!)

float * powerspectrumHostBuffer;
float butterfly_twiddle_radix3;

// local prototypes
cl_int createProgramKernels(const cl_device_id oclDeviceId, const cl_uint oclPlatformVendor);
cl_int releaseProgramKernels();


int initialize_ocl(int oclDeviceIdGiven, int *oclDeviceIdPtr, cl_platform_id boincPlatformId, cl_device_id boincDeviceId)
{
    cl_int oclResult;
    cl_uint oclNumPlatforms = 0;
    cl_platform_id oclPlatform = NULL;
    cl_device_id oclDeviceId = NULL;
    cl_uint oclPlatformVendor = NULL;

    // did BOINC provide the platform and device IDs to be used?
    if(boincPlatformId && boincDeviceId) {
        oclPlatform = boincPlatformId;
        oclDeviceId = boincDeviceId;

        // display platform vendor
        char platformVendor[OCL_MAX_STRING] = "UNKNOWN";
        oclResult = clGetPlatformInfo(oclPlatform, CL_PLATFORM_VENDOR, sizeof(platformVendor), platformVendor, NULL);
        if(CL_SUCCESS != oclResult) {
            logMessage(warn, true, "Couldn't retrieve OpenCL platform vendor (error: %i)!\n", oclResult);
        }
        else {
            logMessage(info, true, "Using OpenCL platform provided by: %s\n", platformVendor);
        }
    }
    else {
        // get number of available platforms
        oclResult = clGetPlatformIDs(0, NULL, &oclNumPlatforms);
        if(CL_SUCCESS != oclResult) {
            logMessage(error, true, "Couldn't retrieve number of OpenCL platforms (error: %i)!\n", oclResult);
            return(RADPUL_OCL_PLATFORM_FIND);
        }

        // retrieve available platforms
        if (0 < oclNumPlatforms) {
            cl_platform_id* platforms = new cl_platform_id[oclNumPlatforms];

            oclResult = clGetPlatformIDs(oclNumPlatforms, platforms, NULL);
            if(CL_SUCCESS != oclResult) {
                logMessage(error, true, "Couldn't retrieve list of OpenCL platforms (error: %i)!\n", oclResult);
                return(RADPUL_OCL_PLATFORM_FIND);
            }

            // iterate over platforms found and check their profiles
            for(cl_uint i = 0; i < oclNumPlatforms; ++i) {
                char profile[32] = {0};

                oclResult = clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, sizeof(profile), profile, NULL);
                if(CL_SUCCESS != oclResult) {
                    logMessage(warn, true, "Couldn't retrieve OpenCL platform %u/%u details (error: %i)! Trying next one...\n", i, oclNumPlatforms, oclResult);
                    continue;
                }

                // use first FULL_PROFILE platform found
                if (!strcmp(profile, "FULL_PROFILE")) {
                    oclPlatform = platforms[i];
                    break;
                }
                else {
                    logMessage(debug, true, "OpenCL platform %u/%u doesn't support FULL_PROFILE (profile: %s)! Trying next one...\n", i, oclNumPlatforms, profile);
                }
            }

            // clean up
            delete[] platforms;
        }

        // did we find any suitable platform?
        if(NULL == oclPlatform) {
            logMessage(error, true, "Couldn't find any suitable OpenCL platform!\n");
            return(RADPUL_OCL_PLATFORM_UNAVAILABLE);
        }

        // display platform vendor
        char platformVendor[OCL_MAX_STRING] = "UNKNOWN";
        oclResult = clGetPlatformInfo(oclPlatform, CL_PLATFORM_VENDOR, sizeof(platformVendor), platformVendor, NULL);
        if(CL_SUCCESS != oclResult) {
            logMessage(warn, true, "Couldn't retrieve OpenCL platform vendor (error: %i)!\n", oclResult);
        }
        else {
            logMessage(info, true, "Using OpenCL platform provided by: %s\n", platformVendor);
        }

        if(oclDeviceIdGiven == 0) {
            // find appropriate OpenCL device;
            logMessage(debug, true, "No (valid) OpenCL device ID passed via command line. Determining suitable GPU device... \n");
            oclDeviceId = findBestFreeDevice(oclPlatform, 1, 1, 0, CL_DEVICE_TYPE_GPU);
            if(NULL == oclDeviceId) {
                logMessage(error, true, "Couldn't find any suitable OpenCL GPU device!\n");
                return(RADPUL_OCL_DEVICE_FIND);
            }
        }
        else {
            cl_device_id oclDevices[OCL_MAX_DEVICES] = {NULL};
            cl_uint oclDeviceCount = 0;

            // retrieve all available GPU devices for given platform
            oclResult = clGetDeviceIDs(oclPlatform, CL_DEVICE_TYPE_GPU, OCL_MAX_DEVICES, oclDevices, &oclDeviceCount);
            if(CL_SUCCESS != oclResult) {
                logMessage(error, true, "Couldn't retrieve list of OpenCL GPU devices (error: %i)!\n", oclResult);
                return(NULL);
            }

            // select device based on ordinal provided (via command line)
            if(*oclDeviceIdPtr >= 0 && oclDeviceCount > *oclDeviceIdPtr) {
                logMessage(debug, true, "Selected OpenCL device #%i as requested via command line...\n", *oclDeviceIdPtr);
                oclDeviceId = oclDevices[*oclDeviceIdPtr];
            }
        }
        if(strstr(platformVendor, "Advanced Micro Devices")) {
            oclPlatformVendor = VENDOR_AMD;
        }
        else if(strstr(platformVendor, "NVIDIA")) {
            oclPlatformVendor = VENDOR_NVIDIA;
        }
    }

    // sanity check
    if(!oclDeviceId) {
        logMessage(error, true, "No suitable OpenCL device available for use!\n");
        return(RADPUL_OCL_DEVICE_FIND);
    }

    // get device vendor
    char deviceVendor[OCL_MAX_STRING] = "UNKNOWN";
    oclResult = clGetDeviceInfo(oclDeviceId, CL_DEVICE_VENDOR, sizeof(deviceVendor), deviceVendor, NULL);
    if(CL_SUCCESS != oclResult) {
        logMessage(warn, true, "Couldn't retrieve OpenCL device vendor (error: %i)!\n", oclResult);
    }

    // get device name
    char deviceProduct[OCL_MAX_STRING] = "UNKNOWN";
    oclResult = clGetDeviceInfo(oclDeviceId, CL_DEVICE_NAME, sizeof(deviceProduct), deviceProduct, NULL);
    if(CL_SUCCESS != oclResult) {
        logMessage(warn, true, "Couldn't retrieve OpenCL device name (error: %i)!\n", oclResult);
    }

    // check whether we can compile our sources
    cl_bool compilerAvailable;
    oclResult = clGetDeviceInfo(oclDeviceId, CL_DEVICE_COMPILER_AVAILABLE, sizeof(cl_bool), &compilerAvailable, NULL);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Couldn't check compiler availability of OpenCL device (error: %i)\n", oclResult);
        return(RADPUL_OCL_COMPILER_UNAVAILABLE);
    }
    if(!compilerAvailable) {
        logMessage(error, true, "OpenCL device \"%s\" by %s doesn't support code compilation!\n", deviceProduct, deviceVendor);
        return(RADPUL_OCL_COMPILER_UNAVAILABLE);
    }

    // check device availability
    cl_bool deviceAvailable;
    oclResult = clGetDeviceInfo(oclDeviceId, CL_DEVICE_AVAILABLE, sizeof(cl_bool), &deviceAvailable, NULL);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Couldn't check availability of OpenCL device (error: %i)\n", oclResult);
        return(RADPUL_OCL_DEVICE_UNAVAILABLE);
    }
    if(deviceAvailable) {
        logMessage(info, true, "Using OpenCL device \"%s\" by: %s\n", deviceProduct, deviceVendor);
    }
    else {
        logMessage(error, true, "OpenCL device \"%s\" by %s isn't available for computation!\n", deviceProduct, deviceVendor);
        return(RADPUL_OCL_DEVICE_UNAVAILABLE);
    }

    // create OpenCL context
    cl_context_properties ctxProps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties) oclPlatform, 0};
    oclContext = clCreateContext(ctxProps, 1, &oclDeviceId, &contextErrorCallback, NULL, &oclResult);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Couldn't create OpenCL context (error: %i)!\n", oclResult);
        if(CL_OUT_OF_HOST_MEMORY == oclResult || CL_OUT_OF_RESOURCES == oclResult) {
            return(RADPUL_OCL_MEM_ALLOC_DEVICE);
        }
        return(RADPUL_OCL_CONTEXT_CREATE);
    }

    // create OpenCL command queue
    oclQueue = clCreateCommandQueue(oclContext, oclDeviceId, 0, &oclResult);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Couldn't create OpenCL command queue (error: %i)!\n", oclResult);
        shutdown_ocl();
        if(CL_OUT_OF_HOST_MEMORY == oclResult || CL_OUT_OF_RESOURCES == oclResult) {
            return(RADPUL_OCL_MEM_ALLOC_DEVICE);
        }
        return(RADPUL_OCL_CMDQUEUE_CREATE);
    }

    // load source, build it and create kernel objects
    oclResult = createProgramKernels(oclDeviceId, oclPlatformVendor);
    if(CL_SUCCESS != oclResult) {
        shutdown_ocl();
        return oclResult;
    }

    // check work group size constraints for each kernel with explicit work group size (limit size if necessary)
    bool needRebuild = false;
    size_t workGroupSizeLimit = 0;

    checkKernelWorkGroupInfo(&oclDeviceId, &oclKernelTimeSeriesMeanReduction, workGroupSizeTSMR[0], &workGroupSizeLimit);
    if(workGroupSizeLimit) {
        needRebuild = true;
        workGroupSizeTSMR[0] = workGroupSizeLimit;
    }

    checkKernelWorkGroupInfo(&oclDeviceId, &oclKernelPowerSpectrum, workGroupSizePS[0], &workGroupSizeLimit);
    if(workGroupSizeLimit) {
        needRebuild = true;
        workGroupSizePS[0] = workGroupSizeLimit;
    }

    checkKernelWorkGroupInfo(&oclDeviceId, &oclKernelPowerSpectrum_radix3_r2c, workGroupSizePS_R3_R2C[0], &workGroupSizeLimit);
    if(workGroupSizeLimit) {
        needRebuild = true;
        workGroupSizePS_R3_R2C[0] = workGroupSizeLimit;
    }



    checkKernelWorkGroupInfo(&oclDeviceId, &oclKernelHarmonicSumming, workGroupSizeHS[0], &workGroupSizeLimit);
    if(workGroupSizeLimit) {
        needRebuild = true;
        workGroupSizeHS[0] = workGroupSizeLimit;

        // ensure that workgroup size is an integer power of 2
        double dummy;
        if(0.0 != modf(log2(workGroupSizeLimit), &dummy)) {
            workGroupSizeHS[0] = lrint(pow(2, floor(log2(workGroupSizeLimit))));
            logMessage(warn, true, "Work group size for \"kernelHarmonicSumming\" must be an integer power of 2! Reducing from %d to %d...\n", workGroupSizeLimit, workGroupSizeHS[0]);
        }

        workGroupSizeHSG[0] = workGroupSizeHS[0] / 2; // work group size MUST be half than that of the first kernel
    }

    // load source with modified compile options, build it again and recreate kernel objects
    if(needRebuild) {

        oclResult = releaseProgramKernels();
        if(CL_SUCCESS != oclResult) {
            shutdown_ocl();
            return oclResult;
        }

        oclResult = createProgramKernels(oclDeviceId, oclPlatformVendor);
        if(CL_SUCCESS != oclResult) {
            shutdown_ocl();
            return oclResult;
        }
    }

    // let the runtime know that we don't need the compiler anymore
    clUnloadCompiler();

    return(0);
}


int set_up_resampling(DIfloatPtr input_dip, DIfloatPtr *output_dip, const RESAMP_PARAMS *const params, float *sinLUTsamples, float *cosLUTsamples)
{
    cl_int oclResult;
    float * input = input_dip.host_ptr;

    cl_mem resampledTimeSeriesDeviceBuffer;

    // sanity checks
    if(params->nsamples_unpadded % workGroupSizeTSMR[0] != 0) {
        logMessage(error, true, "The time series length %i isn't an integer multiple of the OpenCL work group size %i!\n", params->nsamples_unpadded, workGroupSizeTSMR[0]);
        return(RADPUL_EVAL);
    }


    // supported padded time series lengths are
    // 2^n
    // 3*2^n

    int len=params->nsamples;
    int complexLen=len/2;

    int subFFTlen=1;
    while( len % 2 == 0) {
      len >>= 1;
      subFFTlen <<= 1;
    }

    if(len !=1 && len != 3 ) {
        logMessage(error, true, "The padded time series length %i isn't supported. Must be k*2^n, k in {1,3}!\n", params->nsamples);
        return(RADPUL_EVAL);
    }

    // allocate device memory for original time series
    originalTimeSeriesDeviceBuffer = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, sizeof(float) * params->nsamples_unpadded, NULL, &oclResult);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error allocating original time series device memory: %i bytes (error: %i)\n", sizeof(float) * params->nsamples_unpadded, oclResult);
        return(RADPUL_OCL_MEM_ALLOC_DEVICE);
    }
    logMessage(debug, true, "Allocated original time series (%u samples, unpadded) device memory: %i bytes\n", params->nsamples_unpadded, sizeof(float) * params->nsamples_unpadded);

    // allocate device memory for sin/cos lookup table
    sinLUTDeviceBuffer = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, ERP_SINCOS_LUT_SIZE * sizeof(float), NULL, &oclResult);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error allocating sin lookup table device memory: %i bytes (error: %i)\n", ERP_SINCOS_LUT_SIZE * sizeof(float), oclResult);
        return(RADPUL_OCL_MEM_ALLOC_DEVICE);
    }
    logMessage(debug, true, "Allocated sin lookup table device memory: %i bytes\n", ERP_SINCOS_LUT_SIZE * sizeof(float));

    cosLUTDeviceBuffer = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, ERP_SINCOS_LUT_SIZE * sizeof(float), NULL, &oclResult);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error allocating cos lookup table device memory: %i bytes (error: %i)\n", ERP_SINCOS_LUT_SIZE * sizeof(float), oclResult);
        return(RADPUL_OCL_MEM_ALLOC_DEVICE);
    }
    logMessage(debug, true, "Allocated cos lookup table device memory: %i bytes\n", ERP_SINCOS_LUT_SIZE * sizeof(float));

    // allocate device memory for modulated time offsets
    modTimeOffsetsDeviceBuffer = clCreateBuffer(oclContext, CL_MEM_READ_WRITE, sizeof(float) * params->nsamples_unpadded, NULL, &oclResult);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error allocating modulated time offsets device memory: %i bytes (error: %i)\n", sizeof(float) * params->nsamples_unpadded, oclResult);
        return(RADPUL_OCL_MEM_ALLOC_DEVICE);
    }
    logMessage(debug, true, "Allocated modulated time offsets device memory: %i bytes\n", sizeof(float) * params->nsamples_unpadded);

    // allocate device memory for modulated time series length
    timeSeriesLengthDeviceBuffer = clCreateBuffer(oclContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &oclResult);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error allocating modulated time series length device memory: %i bytes (error: %i)\n", sizeof(int), oclResult);
        return(RADPUL_OCL_MEM_ALLOC_DEVICE);
    }
    logMessage(debug, true, "Allocated modulated time series length device memory: %i bytes\n", sizeof(int));

    // allocate device memory for resampled time series (we need twice the amount of samples as buffer because of the fake C2C FFT input (split-complex)
    resampledTimeSeriesDeviceBuffer = clCreateBuffer(oclContext, CL_MEM_READ_WRITE, 2 * sizeof(float) * params->nsamples, NULL, &oclResult);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error allocating modulated time series device memory: %i bytes (error: %d)\n", 2 * sizeof(float) * params->nsamples, oclResult);
        return(RADPUL_OCL_MEM_ALLOC_DEVICE);
    }
    logMessage(debug, true, "Allocated modulated time series device memory: %i bytes\n", 2 * sizeof(float) * params->nsamples);


    // allocate device memory for time series mean sum reduction (we need two separate buffers, each half used alternately)
    timeSeriesMeanDeviceBuffer = clCreateBuffer(oclContext, CL_MEM_READ_WRITE, sizeof(float) * (params->nsamples_unpadded/workGroupSizeTSMR[0] * 2), NULL, &oclResult);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error allocating modulated time series mean reduction device memory: %i bytes (error: %i)\n", sizeof(float) * (params->nsamples_unpadded/workGroupSizeTSMR[0]*2), oclResult);
        return(RADPUL_OCL_MEM_ALLOC_DEVICE);
    }
    logMessage(debug, true, "Allocated time series mean reduction device memory: %i bytes\n", sizeof(float) * (params->nsamples_unpadded/OCL_RESAMP_REDUCTION_WGSIZE_X*2));

    //allocate device memory for FFT input buffer (padded resampled time series)
    fftComplexDeviceBuffer = clCreateBuffer(oclContext, CL_MEM_READ_WRITE,sizeof(float)*2*complexLen ,NULL, &oclResult);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during FFT in/out buffer acquisition: %i bytes (error: %d)\n", sizeof(float)*2*complexLen, oclResult);
        return(RADPUL_OCL_MEM_ALLOC_DEVICE);
    }
    logMessage(debug, true, "Acquired complex FFT in/out buffer: %i bytes\n", sizeof(float)*2*complexLen);




    // transfer original time series data to device
    oclResult = clEnqueueWriteBuffer(oclQueue, originalTimeSeriesDeviceBuffer, CL_FALSE, 0, sizeof(float) * params->nsamples_unpadded, input, 0 , NULL, NULL);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL host->device original time series data transfer (error: %i)\n", oclResult);
        return(RADPUL_OCL_MEM_COPY_HOST_DEVICE);
    }
    logMessage(debug, true, "OpenCL host->device original time series data transfer enqueued...\n");




    // transfer sin lookup table data to device
    oclResult = clEnqueueWriteBuffer(oclQueue, sinLUTDeviceBuffer, CL_FALSE, 0, ERP_SINCOS_LUT_SIZE * sizeof(float), sinLUTsamples, 0 , NULL, NULL);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL host->device sin lookup table data transfer (error: %i)\n", oclResult);
        return(RADPUL_OCL_MEM_COPY_HOST_DEVICE);
    }
    logMessage(debug, true, "OpenCL host->device sin lookup table data transfer enqueued...\n");

    // transfer cos lookup table data to device
    oclResult = clEnqueueWriteBuffer(oclQueue, cosLUTDeviceBuffer, CL_FALSE, 0, ERP_SINCOS_LUT_SIZE * sizeof(float), cosLUTsamples, 0 , NULL, NULL);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL host->device cos lookup table data transfer (error: %d)\n", oclResult);
        return(RADPUL_OCL_MEM_COPY_HOST_DEVICE);
    }
    logMessage(debug, true, "OpenCL host->device cos lookup table data transfer enqueued...\n");

    oclResult = clFinish(oclQueue);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL buffer synchronization: resampling (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_INVOKE);
    }
    logMessage(debug, true, "OpenCL resampling buffer initialization successful...\n");


    output_dip->device_ptr=resampledTimeSeriesDeviceBuffer;

    return(0);
}


int run_resampling(DIfloatPtr input_dip, DIfloatPtr output_dip, const RESAMP_PARAMS *const params)
{
    // unused (doesn't prevent nvcc warnings, oh well)
    float * input = NULL;

    cl_mem resampledTimeSeriesDeviceBuffer = output_dip.device_ptr;

    // output variables
    unsigned int n_steps = 0;
    float mean = 0.0f;

    cl_int oclResult;

    unsigned int subFFTlen=1;
    unsigned int fftradix=1;

    // decompose padded time series length into k*2^n
    // set_up part took care to check we actually support value of k

    unsigned int len=params->nsamples;
    while(len % 2 == 0) {
      subFFTlen <<= 1;
      len >>=1;
    }
    fftradix=len;


    // compute time offsets

    const size_t workItemAmountTSM[] = {params->nsamples_unpadded};
    logMessage(debug, true, "Executing time series modulation OpenCL kernel %lu times...\n", workItemAmountTSM[0]);

    // prepare parameters
    oclResult = clSetKernelArg(oclKernelTimeSeriesModulation, 0, sizeof(sinLUTDeviceBuffer), &sinLUTDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: TSM (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelTimeSeriesModulation, 1, sizeof(cosLUTDeviceBuffer), &cosLUTDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: TSM (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelTimeSeriesModulation, 2, sizeof(params->tau), &params->tau);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: TSM (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelTimeSeriesModulation, 3, sizeof(params->Omega), &params->Omega);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: TSM (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelTimeSeriesModulation, 4, sizeof(params->Psi0), &params->Psi0);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: TSM (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelTimeSeriesModulation, 5, sizeof(params->dt), &params->dt);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: TSM (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelTimeSeriesModulation, 6, sizeof(params->step_inv), &params->step_inv);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: TSM (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelTimeSeriesModulation, 7, sizeof(params->S0), &params->S0);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: TSM (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelTimeSeriesModulation, 8, sizeof(modTimeOffsetsDeviceBuffer), &modTimeOffsetsDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: TSM (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    // invoke kernel grid
    oclResult = clEnqueueNDRangeKernel(oclQueue, oclKernelTimeSeriesModulation, 1, NULL, workItemAmountTSM, NULL, 0, NULL, NULL);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel setup: TSM (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_INVOKE);
    }

    // wait for all enqueued commands to finish
    oclResult = clFinish(oclQueue);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL synchronization: TSM (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_INVOKE);
    }
    logMessage(debug, true, "OpenCL TSM kernel execution successful...\n");

    // determine modulated time series length (done in kernel because global sync required and to avoid bus transfer of del_t as well as waste of host memory)

    logMessage(debug, true, "Executing modulated time series length OpenCL kernel (single work item)...\n");

    // prepare parameters
    oclResult = clSetKernelArg(oclKernelTimeSeriesLengthModulated, 0, sizeof(params->nsamples_unpadded), &params->nsamples_unpadded);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: TSLM (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelTimeSeriesLengthModulated, 1, sizeof(modTimeOffsetsDeviceBuffer), &modTimeOffsetsDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: TSLM (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelTimeSeriesLengthModulated, 2, sizeof(timeSeriesLengthDeviceBuffer), &timeSeriesLengthDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: TSLM (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    // launch kernel (one single work item)
    oclResult = clEnqueueTask(oclQueue, oclKernelTimeSeriesLengthModulated, 0, NULL, NULL);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel setup: TSLM (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_INVOKE);
    }

    // return computed time series length
    oclResult = clEnqueueReadBuffer(oclQueue, timeSeriesLengthDeviceBuffer, CL_FALSE, 0, sizeof(n_steps), &n_steps, 0, NULL, NULL);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL device->host time series length transfer (error: %d)\n", oclResult);
        return(RADPUL_OCL_MEM_COPY_DEVICE_HOST);
    }
    logMessage(debug, true, "OpenCL device->host time series length transfer successful...\n");

    // wait for all enqueued commands to finish
    oclResult = clFinish(oclQueue);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL synchronization: TSLM (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_INVOKE);
    }
    logMessage(debug, true, "OpenCL TSLM kernel execution successful...\n");
    logMessage(debug, true, "Modulated time series length: %u\n", n_steps);

    // compute resampled time series (unpadded)

    const size_t workItemAmountTSR[] = {2 * params->nsamples};
    logMessage(debug, true, "Executing time series resampling OpenCL kernel %lu times...\n", workItemAmountTSR[0]);

    // prepare parameters
    oclResult = clSetKernelArg(oclKernelTimeSeriesResampling, 0, sizeof(originalTimeSeriesDeviceBuffer), &originalTimeSeriesDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: TSR (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelTimeSeriesResampling, 1, sizeof(modTimeOffsetsDeviceBuffer), &modTimeOffsetsDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: TSR (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelTimeSeriesResampling, 2, sizeof(n_steps), &n_steps);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: TSR (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelTimeSeriesResampling, 3, sizeof(resampledTimeSeriesDeviceBuffer), &resampledTimeSeriesDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: TSR (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    // launch kernel grid
    oclResult = clEnqueueNDRangeKernel(oclQueue, oclKernelTimeSeriesResampling, 1, NULL, workItemAmountTSR, NULL, 0, NULL, NULL);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel setup: TSR (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_INVOKE);
    }

    // wait for all enqueued commands to finish
    oclResult = clFinish(oclQueue);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL synchronization: TSR (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_INVOKE);
    }
    logMessage(debug, true, "OpenCL TSR kernel execution successful...\n");

    // compute time series mean value

    // sum reduction loop control variables
    cl_mem currentInputBuffer = NULL;

    // TODO: n_steps (effective length) threads would by sufficient
    // don't mess with global value, use a local copy since we'll modify its value
    size_t currentWorkGroupSizeTSMR[] = {workGroupSizeTSMR[0]};
    int requiredBlocks = params->nsamples_unpadded / currentWorkGroupSizeTSMR[0];
    cl_mem currentOutputBuffer = NULL;
    int secondHalfOffset = requiredBlocks;
    bool useSecondHalfForOutput = false;
    int i = 1;
    unsigned int offsetInput,offsetOutput;

    do {
        // use kernel decomposition to do sum reduction (facilitates global memory sync, kernel invocations are cheap)

        const size_t workItemAmountTSMR[] = {requiredBlocks * currentWorkGroupSizeTSMR[0]};
        logMessage(debug, true, "Executing time series mean reduction OpenCL kernel (iteration %i using %i work groups of %u work items)...\n", i, requiredBlocks, currentWorkGroupSizeTSMR[0]);

        // only the first iteration uses the resampled time series as input (obviously)
        if(1 == i) {
            currentInputBuffer = resampledTimeSeriesDeviceBuffer;
            offsetInput=0;
        }
        else {
            // otherwise: alternate between first and second half of mean device buffer for input (inversely to output buffer)
            offsetInput = useSecondHalfForOutput ? 0 : secondHalfOffset;
            currentInputBuffer = timeSeriesMeanDeviceBuffer;
        }

        // alternate between first and second half of mean device buffer for output (inversely to input buffer)

        offsetOutput = useSecondHalfForOutput ? secondHalfOffset : 0;
        currentOutputBuffer = timeSeriesMeanDeviceBuffer;

        // prepare parameters

        oclResult = clSetKernelArg(oclKernelTimeSeriesMeanReduction, 0, sizeof(currentInputBuffer), &currentInputBuffer);
        if(CL_SUCCESS != oclResult) {
            logMessage(error, true, "Error during OpenCL kernel parameter setup: TSMR-%i (error: %d)\n", i, oclResult);
            return(RADPUL_OCL_KERNEL_SETUP);
        }


        oclResult = clSetKernelArg(oclKernelTimeSeriesMeanReduction, 1, sizeof(currentOutputBuffer), &currentOutputBuffer);
        if(CL_SUCCESS != oclResult) {
            logMessage(error, true, "Error during OpenCL kernel parameter setup: TSMR-%i (error: %d)\n", i, oclResult);
            return(RADPUL_OCL_KERNEL_SETUP);
        }

       oclResult = clSetKernelArg(oclKernelTimeSeriesMeanReduction, 2, sizeof(offsetInput), &offsetInput);
        if(CL_SUCCESS != oclResult) {
            logMessage(error, true, "Error during OpenCL kernel parameter setup: TSMR-%i (error: %d)\n", i, oclResult);
            return(RADPUL_OCL_KERNEL_SETUP);
        }
       oclResult = clSetKernelArg(oclKernelTimeSeriesMeanReduction, 3, sizeof(offsetOutput), &offsetOutput);
        if(CL_SUCCESS != oclResult) {
            logMessage(error, true, "Error during OpenCL kernel parameter setup: TSMR-%i (error: %d)\n", i, oclResult);
            return(RADPUL_OCL_KERNEL_SETUP);
        }



        // launch kernel grid (iteratively reduced number of threads)
        oclResult = clEnqueueNDRangeKernel(oclQueue, oclKernelTimeSeriesMeanReduction, 1, NULL, workItemAmountTSMR, currentWorkGroupSizeTSMR, 0, NULL, NULL);
        if(CL_SUCCESS != oclResult) {
            logMessage(error, true, "Error during OpenCL kernel setup: TSMR-%i (error: %d)\n", i, oclResult);
            return(RADPUL_OCL_KERNEL_INVOKE);
        }

        // wait for all enqueued commands to finish
        oclResult = clFinish(oclQueue);
        if(CL_SUCCESS != oclResult) {
            logMessage(error, true, "Error during OpenCL synchronization: TSMR-%i (error: %d)\n", i, oclResult);
            return(RADPUL_OCL_KERNEL_INVOKE);
        }
        logMessage(debug, true, "OpenCL TSMR-%i kernel execution successful...\n", i);

        // required blocks for next iteration
        if (requiredBlocks >= currentWorkGroupSizeTSMR[0]) {
            // we still can fill full blocks
            requiredBlocks /= currentWorkGroupSizeTSMR[0];
        }
        else {
            if (requiredBlocks == 1) {
                // this was the final summing by the last block
                break;
            }
            else {
                // we're now within the last block (with fewer than blocksize elements), so sum pairs with one thread each
                currentWorkGroupSizeTSMR[0] = requiredBlocks;
                requiredBlocks = 1;
            }
        }

        // flip output buffer specifier
        useSecondHalfForOutput = useSecondHalfForOutput ? false : true;

        // update progress counter
        i++;
    }
    while(requiredBlocks > 0);

    // return computed time series sum (first element of output buffer)


    oclResult = clEnqueueReadBuffer(oclQueue, currentOutputBuffer, CL_TRUE,offsetOutput*sizeof(float) , sizeof(float), &mean, 0, NULL, NULL);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL device->host time series sum transfer (error: %d)\n", oclResult);
        return(RADPUL_OCL_MEM_COPY_DEVICE_HOST);
    }


    logMessage(debug, true, "OpenCL device->host time series sum (%e) transfer successful...\n", mean);


    // compute actual mean
    mean /= n_steps;

    logMessage(debug, true, "Actual time series mean is: %e\n", mean);

    // apply mean padding to time series
    // if the padded length is N = k*2^n , k not a p.o.two, then
    //    also reorder so that the input array is set up for k many batched FFTs of len 2^n each
    //    in this case the output is produced in the second half of the input array and the first half is
    //    zeroed out in prep. for FFT step


    if(fftradix == 1) {
      const size_t workItemAmountTSMP[] = {params->nsamples};
    logMessage(debug, true, "Executing time series padding OpenCL kernel %lu times...\n", workItemAmountTSMP[0]);

    // prepare parameters
    oclResult = clSetKernelArg(oclKernelTimeSeriesPadding, 0, sizeof(mean), &mean);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: TSMP (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelTimeSeriesPadding, 1, sizeof(n_steps), &n_steps);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: TSMP (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelTimeSeriesPadding, 2, sizeof(resampledTimeSeriesDeviceBuffer), &resampledTimeSeriesDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: TSMP (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelTimeSeriesPadding, 3, sizeof(fftComplexDeviceBuffer), &fftComplexDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: TSMP (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    // launch kernel grid
    oclResult = clEnqueueNDRangeKernel(oclQueue, oclKernelTimeSeriesPadding, 1, NULL, workItemAmountTSMP, NULL, 0, NULL, NULL);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel setup: TSMP (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_INVOKE);
    }

    } else {
      // fftradix != 1

      const size_t workItemAmountTSMP[] = {subFFTlen/2};

      logMessage(debug, true, "Executing time series padding & transpose OpenCL kernel %lu times...\n", workItemAmountTSMP[0]);


      // prepare parameters
      oclResult = clSetKernelArg(oclKernelTimeSeriesPaddingTranspose, 0, sizeof(mean), &mean);
      if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: TSMP_T0 (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
      }

      oclResult = clSetKernelArg(oclKernelTimeSeriesPaddingTranspose, 1, sizeof(n_steps), &n_steps);
      if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: TSMP_T1 (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
      }


      oclResult = clSetKernelArg(oclKernelTimeSeriesPaddingTranspose, 2, sizeof(fftradix), &fftradix);
      if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: TSMP_t2 (error: %d)\n", oclResult);
	    return(RADPUL_OCL_KERNEL_SETUP);
      }

      oclResult = clSetKernelArg(oclKernelTimeSeriesPaddingTranspose, 3, sizeof(subFFTlen), &subFFTlen);
      if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: TSMP_T3 (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
      }



      oclResult = clSetKernelArg(oclKernelTimeSeriesPaddingTranspose, 4, sizeof(resampledTimeSeriesDeviceBuffer), &resampledTimeSeriesDeviceBuffer);
      if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: TSMP_T5 (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
      }

      oclResult = clSetKernelArg(oclKernelTimeSeriesPaddingTranspose, 5, sizeof(fftComplexDeviceBuffer), &fftComplexDeviceBuffer);
      if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: TSMP_T5 (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
      }

      // launch kernel grid
      oclResult = clEnqueueNDRangeKernel(oclQueue, oclKernelTimeSeriesPaddingTranspose, 1, NULL, workItemAmountTSMP, NULL, 0, NULL, NULL);
      if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel setup: TSMP_T (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_INVOKE);
      }

    }

    // wait for all enqueued commands to finish
    oclResult = clFinish(oclQueue);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL synchronization: TSMP (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_INVOKE);
    }
    logMessage(debug, true, "OpenCL TSMP kernel execution successful...\n");



    return(0);
}


int tear_down_resampling(DIfloatPtr output_dip)
{
    cl_int oclResult;

    cl_mem resampledTimeSeriesDeviceBuffer = output_dip.device_ptr;

    oclResult = clReleaseMemObject(originalTimeSeriesDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(warn, true, "Couldn't release OpenCL memory object: OTS (error: %i)\n", oclResult);
    }

    oclResult = clReleaseMemObject(sinLUTDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(warn, true, "Couldn't release OpenCL memory object: SLT (error: %i)\n", oclResult);
    }

    oclResult = clReleaseMemObject(cosLUTDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(warn, true, "Couldn't release OpenCL memory object: CLT (error: %i)\n", oclResult);
    }

    oclResult = clReleaseMemObject(modTimeOffsetsDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(warn, true, "Couldn't release OpenCL memory object: MTO (error: %i)\n", oclResult);
    }

    oclResult = clReleaseMemObject(timeSeriesLengthDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(warn, true, "Couldn't release OpenCL memory object: TSL (error: %i)\n", oclResult);
    }

    oclResult = clReleaseMemObject(timeSeriesMeanDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(warn, true, "Couldn't release OpenCL memory object: TSM (error: %i)\n", oclResult);
    }

    // release FFT in/out buffer
    oclResult = clReleaseMemObject(fftComplexDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(warn, true, "Couldn't release OpenCL memory object: FFT (error: %i)\n", oclResult);
    }


    // release resampling buffer object
    oclResult = clReleaseMemObject(resampledTimeSeriesDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(warn, true, "Couldn't release OpenCL memory object: RTS (error: %i)\n", oclResult);
    }


    return(0);
}


int set_up_fft(DIfloatPtr input_dip, DIfloatPtr *output_dip, uint32_t nsamples, unsigned int fft_size)
{
    // unused
    float * input = NULL;

    cl_mem powerSpectrumDeviceBuffer;

    cl_int oclResult;

    int len=nsamples;
    int complexLen=len/2;

    int subFFTlen=1;
    while( len % 2 == 0) {
      len >>= 1;
      subFFTlen<<= 1;
    }
    int fftradix=len;


    // TODO check
    // increase powerspectrum buffer length such that it matches the powerspectrum kernel's blocklength (no further control flow required in kernel)
    const unsigned int fft_size_padded = PADDED_FFT_SIZE(fft_size);

    logMessage(debug, true, "Padding output size of FFT with %u samples from %u to %u...\n", nsamples, fft_size, fft_size_padded);

    // prepare FFT
    // set length to 2^n when nsamples = k*2^n
    // we are doing a "packed" C2C transform , so we additionally half the length
    clFFT_Dim3 sampleSpace = {subFFTlen/2, 1, 1};

    // create FFT plan
    clFftPlan = clFFT_CreatePlanAdv(oclContext, sampleSpace, clFFT_1D, clFFT_InterleavedComplexFormat, clFFT_TaylorLUT, &oclResult);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error creating OpenCL FFT plan (error: %i)\n", oclResult);
        if(CL_MEM_OBJECT_ALLOCATION_FAILURE == oclResult || CL_OUT_OF_RESOURCES == oclResult) {
            return(RADPUL_OCL_MEM_ALLOC_DEVICE);
        }
        return(RADPUL_OCL_FFT_PLAN);
    }
    logMessage(debug, true, "Created OpenCL FFT plan for %d samples...\n", sampleSpace.x * sampleSpace.y * sampleSpace.z);


    // allocate device memory for power spectrum
    powerSpectrumDeviceBuffer = clCreateBuffer(oclContext, CL_MEM_READ_WRITE, sizeof(float) * fft_size_padded, NULL, &oclResult);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error allocating power spectrum device memory: %i bytes (error: %i)\n", sizeof(float) * fft_size_padded, oclResult);
        return(RADPUL_OCL_MEM_ALLOC_DEVICE);
    }
    logMessage(debug, true, "Allocated power spectrum device memory: %i bytes\n", sizeof(float) * fft_size_padded);

    // we are using interleaved format to pack a length N Real to Complex transform into a length N/2 Complex 2 Complex transform
    // The "untangling" to get the right results for the Real 2 Complex transform is done in the kernel that also performs the
    // computation of the power and the radix 3 butterflies in case of N=3*2^n


    // pre-compute twiddle factors for radix-k butterfly
    // TODO: two alternatives: optimized per k or generic. let's start with k=3
    // TODO: it's somewhat ugly that we pass the plan as global var, if we could make it generic across
    // TODO: FFT impls we could pass it via function argument. then also the twiddle factors might be stored
    // TODO: in it. Until then, let's keep the twiddle factors as global vars as well :-(

    // allocate device memory for (complex) twiddle factors.

    fftTwiddleFactorDeviceBuffer = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, 2* sizeof(float) * nsamples/2, NULL, &oclResult);
    if(CL_SUCCESS != oclResult) {
      logMessage(error, true, "Error allocating twiddle factor device memory: %i bytes (error: %i)\n", sizeof(float) * 2*nsamples/2, oclResult);
      return(RADPUL_OCL_MEM_ALLOC_DEVICE);
    }
    logMessage(debug, true, "Allocated twiddle factor device memory: %i bytes\n", sizeof(float) * 2 * nsamples/2);

    fftTwiddleFactor_r2cDeviceBuffer = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, 2* sizeof(float) * nsamples/2, NULL, &oclResult);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error allocating twiddle factor (2) device memory: %i bytes (error: %i)\n", sizeof(float) * 2*nsamples/2, oclResult);
        return(RADPUL_OCL_MEM_ALLOC_DEVICE);
    }
    logMessage(debug, true, "Allocated twiddle factor (2) device memory: %i bytes\n", sizeof(float) * 2 * nsamples/2);

    float * twiddle_f = (float*)calloc(2*nsamples/2,sizeof(float));
    if(NULL == twiddle_f) {
      logMessage(error, true, "Couldn't allocate %d bytes of memory for twiddle factors!\n", sizeof(float) * 2 * nsamples/2);
      return(RADPUL_OCL_MEM_ALLOC_HOST);
    }
    logMessage(debug, true, "Allocated temp host memory for twiddle factors: %i bytes\n", sizeof(float)* 2 * nsamples/2);

    unsigned int n,k,r;
    double TWO_PI=atan(1)*8.0;
    double N_recip= 1.0 / ((double) nsamples *0.5);

    // computes complex twiddle factors e^(-2pi*i*k*r/N)
    // where k and r can be thought of as indices over a
    // matrix of size radix x 2^m
    // note: if nsamples = 2^m, all twiddle_factors equal 1.0 + 0i
    // The packed C2C FFT effectively halves the transform length

    for(n=0,k=0,r=0; n < nsamples/2; n++) {
      double angle= TWO_PI*((double)k * (double)r * N_recip);

      // some special twiddling for radix=3, saves two multiplications per
      // butterfly step
      double extra_twiddle=1.0;
      if(fftradix == 3) {
	  if( (r==1) || (r==2) ) {
	    extra_twiddle=0.5;
	  }
      }

      // real and imaginary parts, resp.
      twiddle_f[n] = cos(angle) * extra_twiddle;
      twiddle_f[n+nsamples/2] = -sin(angle) * extra_twiddle;
      k++;
      if(k==subFFTlen/2) {
	    k=0;
	    r++;
      }
    }

    // special twiddle factor for radix 3 butterfly;
    butterfly_twiddle_radix3 = sqrt(3);

    // copy twiddle factors from host to device. Blocking because we want to reuse twiddle_f

    oclResult = clEnqueueWriteBuffer(oclQueue, fftTwiddleFactorDeviceBuffer, CL_TRUE, 0, 2 * nsamples /2 * sizeof(float), twiddle_f, 0 , NULL, NULL);
    if(CL_SUCCESS != oclResult) {
      logMessage(error, true, "Error during OpenCL host->device fft twiddle factors data transfer (error: %i)\n", oclResult);
      return(RADPUL_OCL_MEM_COPY_HOST_DEVICE);
    }
    logMessage(debug, true, "OpenCL host->device fft twiddle factors data transfer ...\n");


    // Precompute second set of twiddle factors for untangling the "packed" C2C transform result to the initial R2C transform result
    N_recip= 1.0 / ((double) nsamples);

    for(n=0; n < nsamples/2; n++) {
        double angle= TWO_PI*(double)n * N_recip;


        // real and imaginary parts, resp.
        twiddle_f[n] = cos(angle);
        twiddle_f[n+nsamples/2] = -sin(angle);
    }

    oclResult = clEnqueueWriteBuffer(oclQueue, fftTwiddleFactor_r2cDeviceBuffer, CL_FALSE, 0, 2 * nsamples/2 * sizeof(float), twiddle_f, 0 , NULL, NULL);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL host->device fft twiddle factors (2) data transfer (error: %i)\n", oclResult);
        return(RADPUL_OCL_MEM_COPY_HOST_DEVICE);
    }
    logMessage(debug, true, "OpenCL host->device fft twiddle (2) factors data transfer ...\n");



    oclResult = clFinish(oclQueue);
    if(CL_SUCCESS != oclResult) {
      logMessage(error, true, "Error during OpenCL buffer synchronization: twiddle factors (2) (error: %d)\n", oclResult);
      return(RADPUL_OCL_KERNEL_INVOKE);
    }
    logMessage(debug, true, "OpenCL twiddle factors (2) buffer initialization successful...\n");



    // free host memory for twiddle factors

    free(twiddle_f);


    output_dip->device_ptr=powerSpectrumDeviceBuffer;

    return(0);
}


int run_fft(DIfloatPtr input_dip, DIfloatPtr output_dip, uint32_t nsamples, unsigned int fft_size, float norm_factor)
{
    cl_int oclResult;
    cl_mem powerSpectrumDeviceBuffer=output_dip.device_ptr;


    // input is passed implicitly in the FFT plan object. Logically, it is the resampled Time
    // series device Buffer, added here for clarity
    cl_mem resampledTimeSeriesDeviceBuffer = input_dip.device_ptr;

    // increase number of powerspectrum threads such that all blocks/threads are used completely (no further control flow required in kernel)
    static const unsigned int fft_size_padded = PADDED_FFT_SIZE(fft_size);


    unsigned int len=nsamples;
    unsigned int subFFTlen=1;
    while( len % 2 == 0) {
      len >>= 1;
      subFFTlen<<= 1;
    }
    int fftradix=len;
    unsigned int packedSubFFTlen=subFFTlen/2;


    // set batch size to radix
    // execute FFT (note: this is in-place!)

    oclResult = clFFT_ExecuteInterleaved(oclQueue, clFftPlan, fftradix, clFFT_Forward, fftComplexDeviceBuffer, fftComplexDeviceBuffer,  0, NULL, NULL);

    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL FFT setup (error: %d)\n", oclResult);
        if(CL_MEM_OBJECT_ALLOCATION_FAILURE == oclResult || CL_OUT_OF_RESOURCES == oclResult) {
            return(RADPUL_OCL_MEM_ALLOC_DEVICE);
        }
        return(RADPUL_OCL_FFT_EXEC);
    }

    oclResult = clFinish(oclQueue);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL synchronization: FFT (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_INVOKE);
    }
    logMessage(debug, true, "OpenCL FFT execution successful...\n");


    // compute powerspectrum

    // TODO: split into a separate source code file perhaps???, this is too much

    if(fftradix == 1) {

    const size_t workItemAmountPS[] = {packedSubFFTlen/2+workGroupSizePS[0]}; //half the length of the complex(!) DFT + 1 . Each thread computes 2 elements of the PS

    logMessage(debug, true, "Executing powerspectrum OpenCL kernel (%u work items each in %u work groups)...\n", workGroupSizePS[0], workItemAmountPS[0]/workGroupSizePS[0]);

    // prepare parameters
    oclResult = clSetKernelArg(oclKernelPowerSpectrum, 0, sizeof(fftComplexDeviceBuffer), &fftComplexDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: PS (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelPowerSpectrum, 1, sizeof(fftTwiddleFactor_r2cDeviceBuffer), &fftTwiddleFactor_r2cDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: PS_R3 (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }


    oclResult = clSetKernelArg(oclKernelPowerSpectrum, 2, sizeof(norm_factor), &norm_factor);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: PS (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelPowerSpectrum, 3, sizeof(packedSubFFTlen), &packedSubFFTlen);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: PS_R3 (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelPowerSpectrum, 4, sizeof(powerSpectrumDeviceBuffer), &powerSpectrumDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: PS (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    // launch kernel grid
    oclResult = clEnqueueNDRangeKernel(oclQueue, oclKernelPowerSpectrum, 1, NULL, workItemAmountPS, workGroupSizePS, 0, NULL, NULL);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel setup: PS (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_INVOKE);
    }
    }

    if(fftradix==3) {
      // kernel computes six (2* radixfft) results per thread,
      const size_t workItemAmountPS_R3_R2C[] = {packedSubFFTlen/2+workGroupSizePS_R3_R2C[0]};

      logMessage(debug, true, "Executing powerspectrum (radix 3) OpenCL kernel (%u work items each in %u work groups)...\n", workGroupSizePS_R3_R2C[0], workItemAmountPS_R3_R2C[0]/workGroupSizePS_R3_R2C[0]);

      // prepare parameters
      oclResult = clSetKernelArg(oclKernelPowerSpectrum_radix3_r2c, 0, sizeof(fftComplexDeviceBuffer), &fftComplexDeviceBuffer);
      if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: PS_R3 (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
      }

      oclResult = clSetKernelArg(oclKernelPowerSpectrum_radix3_r2c, 1, sizeof(fftTwiddleFactorDeviceBuffer), &fftTwiddleFactorDeviceBuffer);
      if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: PS_R3 (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
      }

      oclResult = clSetKernelArg(oclKernelPowerSpectrum_radix3_r2c, 2, sizeof(fftTwiddleFactor_r2cDeviceBuffer), &fftTwiddleFactor_r2cDeviceBuffer);
      if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: PS_R3 (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
      }
      oclResult = clSetKernelArg(oclKernelPowerSpectrum_radix3_r2c, 3, sizeof(packedSubFFTlen), &packedSubFFTlen);
      if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: PS_R3 (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
      }

      oclResult = clSetKernelArg(oclKernelPowerSpectrum_radix3_r2c, 4, sizeof(fft_size_padded), &fft_size_padded);
      if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: PS_R3 (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
      }

      oclResult = clSetKernelArg(oclKernelPowerSpectrum_radix3_r2c, 5, sizeof(butterfly_twiddle_radix3), &butterfly_twiddle_radix3);
      if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: PS_R3 (error: %d)\n", oclResult);
	return(RADPUL_OCL_KERNEL_SETUP);
      }

      oclResult = clSetKernelArg(oclKernelPowerSpectrum_radix3_r2c, 6, sizeof(norm_factor), &norm_factor);
      if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: PS_R3 (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
      }

      oclResult = clSetKernelArg(oclKernelPowerSpectrum_radix3_r2c, 7, sizeof(powerSpectrumDeviceBuffer), &powerSpectrumDeviceBuffer);
      if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: PS_R3 (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
      }

      // launch kernel grid
      oclResult = clEnqueueNDRangeKernel(oclQueue, oclKernelPowerSpectrum_radix3_r2c, 1, NULL, workItemAmountPS_R3_R2C, workGroupSizePS_R3_R2C, 0, NULL, NULL);
      if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel setup: PS_R3 (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_INVOKE);
      }


    }



    // wait for all enqueued commands to finish
    oclResult = clFinish(oclQueue);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL synchronization: PS (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_INVOKE);
    }
    logMessage(debug, true, "OpenCL PS kernel execution successful...\n");


    return(0);
}


int tear_down_fft(DIfloatPtr output_dip)
{
    cl_int oclResult;
    cl_mem powerSpectrumDeviceBuffer = output_dip.device_ptr;

    // release device buffer (power spectrum)
    oclResult = clReleaseMemObject(powerSpectrumDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
      logMessage(warn, true, "Couldn't release OpenCL memory object: FFTPS (error: %i)\n", oclResult);
    }

    // release FFT resources
    clFFT_DestroyPlan(clFftPlan);


    // release twiddle factors buffer objects
    oclResult = clReleaseMemObject(fftTwiddleFactorDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
      logMessage(warn, true, "Couldn't release OpenCL memory object: FFTTF (error: %i)\n", oclResult);
    }
    oclResult = clReleaseMemObject(fftTwiddleFactor_r2cDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(warn, true, "Couldn't release OpenCL memory object: FFTTF (2)(error: %i)\n", oclResult);
    }


    return(0);
}

int set_up_harmonic_summing(float ** sumspec, int32_t** dirty, unsigned int * nr_pages_ptr, unsigned int fundamental_idx_hi,unsigned int harmonic_idx_hi)
{
    cl_int oclResult;
    unsigned int nr_pages;
    int i;

    // allocate memory for the harmonic summed spectra
    // in GPU version, this includes the 1st harmonics

    for( i = 1; i < 5; i++) {
        sumspec[i] = (float *) calloc(fundamental_idx_hi, sizeof(float));
        if(NULL == sumspec[i]) {
            logMessage(error, true, "Error allocating summed spectra memory: %d bytes\n", fundamental_idx_hi * sizeof(float));
            return(RADPUL_EMEM);
        }
    }


    for(i = 1; i < 5; ++i) {
        sumspecDev[i] = clCreateBuffer(oclContext, CL_MEM_WRITE_ONLY, sizeof(float) * fundamental_idx_hi, NULL, &oclResult);
        if(CL_SUCCESS != oclResult) {
            logMessage(error, true, "Error allocating harmonic summing spectra device memory: %i bytes (error: %i)\n", sizeof(float) * fundamental_idx_hi, oclResult);
            return(RADPUL_OCL_MEM_ALLOC_DEVICE);
        }
    }


    // allocate memory for "dirty" page tables and set them to zero initially
    // (no dirty pages)
    nr_pages=(fundamental_idx_hi >> LOG_PS_PAGE_SIZE)+1;
    *nr_pages_ptr = nr_pages;
    for( i = 0; i < 5 ; i++) {
      dirty[i] = (int32_t *) calloc(nr_pages, sizeof(int32_t));
      if(dirty[i] == NULL)
      {
        logMessage(error, true, "Couldn't allocate %d bytes of memory for sumspec page flags at bottom level.\n", fundamental_idx_hi*sizeof(float ));
        return(RADPUL_EMEM);
      }
    }


    // allocate constant device buffers
    // TODO: try using texture memory (image objects) for improved caching
    harmomicSumminghLUTDeviceBuffer = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, sizeof(int32_t) * 16, NULL, &oclResult);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error allocating harmonic summing h-LUT device memory: %i bytes (error: %i)\n", sizeof(int32_t) * 16, oclResult);
        return(RADPUL_OCL_MEM_ALLOC_DEVICE);
    }
    logMessage(debug, true, "Allocated harmonic summing h-LUT device memory: %i bytes\n", sizeof(int32_t) * 16);

    harmomicSummingkLUTDeviceBuffer = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, sizeof(int32_t) * 16, NULL, &oclResult);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error allocating harmonic summing k-LUT device memory: %i bytes (error: %i)\n", sizeof(int32_t) * 16, oclResult);
        return(RADPUL_OCL_MEM_ALLOC_DEVICE);
    }
    logMessage(debug, true, "Allocated harmonic summing k-LUT device memory: %i bytes\n", sizeof(int32_t) * 16);

    harmomicSummingThresholdsDeviceBuffer = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, sizeof(float) * 5, NULL, &oclResult);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error allocating harmonic summing thresholds device memory: %i bytes (error: %i)\n", sizeof(float) * 5, oclResult);
        return(RADPUL_OCL_MEM_ALLOC_DEVICE);
    }
    logMessage(debug, true, "Allocated harmonic summing thresholds device memory: %i bytes\n", sizeof(float) * 5);

    // transfer lookup tables
    oclResult = clEnqueueWriteBuffer(oclQueue, harmomicSumminghLUTDeviceBuffer, CL_FALSE, 0, sizeof(int32_t) * 16, h_lut, 0, NULL, NULL);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL host->device HS h-LUT data transfer (error: %i)\n", oclResult);
        return(RADPUL_OCL_MEM_COPY_HOST_DEVICE);
    }

    oclResult = clEnqueueWriteBuffer(oclQueue, harmomicSummingkLUTDeviceBuffer, CL_FALSE, 0, sizeof(int32_t) * 16, k_lut, 0, NULL, NULL);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL host->device HS k-LUT data transfer (error: %i)\n", oclResult);
        return(RADPUL_OCL_MEM_COPY_HOST_DEVICE);
    }


    powerspectrumHostBuffer = (float *) calloc(harmonic_idx_hi, sizeof(float));
    if(NULL == powerspectrumHostBuffer) {
        logMessage(error, true, "Error allocating powerspectrum memory: %d bytes\n", harmonic_idx_hi * sizeof(float));
        return(RADPUL_EMEM);
    }

    return(0);
}

int run_harmonic_summing(float ** sumspec, int32_t ** dirty, unsigned int nr_pages, DIfloatPtr  powerspectrum_dip, unsigned int window_2,unsigned int fundamental_idx_hi,unsigned int harmonic_idx_hi, float * thresholds)
{
    cl_int oclResult;
    cl_mem powerSpectrumDeviceBuffer= powerspectrum_dip.device_ptr;

    float * powerspectrum = powerspectrumHostBuffer;
    unsigned int l1, l2, i,j, k;
    int nr_pages_total = nr_pages * 5;

    // borders for kernel computation in a 16 index grid
    // for simplicity we always start at the left border of the spectrum,
    // the kernel itself will take care of the window_2 offset */
    l1= 0;
    // the number of main kernel blocks of width 16 that is needed to fully cover
    // the spectrum up to index harmonic_idx_hi -1 (inclusive) */
    l2= ((harmonic_idx_hi -1 + 8) >> 4) +1 ;

    cl_mem dirtyDeviceBuffer;
    int32_t * dirtyTmp;


    // add powerspectrum as first spectra element

    sumspec[0] = powerspectrum;
    sumspecDev[0] = powerSpectrumDeviceBuffer;

    // prepare to initialize sumspec device memory below
    const size_t workItemAmountFFB[] = {((fundamental_idx_hi >> 10) +1) << 10};
    const size_t workItemAmountFIB[] = {((nr_pages_total  >> 10) +1) << 10};

    // allocate sumspec arrays on device
    for(i = 1; i < 5; ++i) {

        // initialize sumspec device memory
        logMessage(debug, true, "Executing buffer fill OpenCL kernel %lu times...\n", workItemAmountFFB[0]);

        // prepare parameters
        oclResult = clSetKernelArg(oclKernelFillFloatBuffer, 0, sizeof(fundamental_idx_hi), &fundamental_idx_hi);
        if(CL_SUCCESS != oclResult) {
            logMessage(error, true, "Error during OpenCL kernel parameter setup: HSFFB (error: %d)\n", oclResult);
            return(RADPUL_OCL_KERNEL_SETUP);
        }

        float value = 0.0f;
        oclResult = clSetKernelArg(oclKernelFillFloatBuffer, 1, sizeof(float), &value);
        if(CL_SUCCESS != oclResult) {
            logMessage(error, true, "Error during OpenCL kernel parameter setup: HSFFB (error: %d)\n", oclResult);
            return(RADPUL_OCL_KERNEL_SETUP);
        }

        oclResult = clSetKernelArg(oclKernelFillFloatBuffer, 2, sizeof(sumspecDev[i]), &sumspecDev[i]);
        if(CL_SUCCESS != oclResult) {
            logMessage(error, true, "Error during OpenCL kernel parameter setup: HSFFB (error: %d)\n", oclResult);
            return(RADPUL_OCL_KERNEL_SETUP);
        }

        // launch kernel grid
        oclResult = clEnqueueNDRangeKernel(oclQueue, oclKernelFillFloatBuffer, 1, NULL, workItemAmountFFB, NULL, 0, NULL, NULL);
        if(CL_SUCCESS != oclResult) {
            logMessage(error, true, "Error during OpenCL kernel setup: HSFFB (error: %d)\n", oclResult);
            return(RADPUL_OCL_KERNEL_INVOKE);
        }
    }


    // allocate dirty pages array on device and fill with 0s
    dirtyDeviceBuffer = clCreateBuffer(oclContext, CL_MEM_WRITE_ONLY, sizeof(int32_t) * nr_pages_total, NULL, &oclResult);
    if(oclResult != CL_SUCCESS) {
        logMessage(error, true, "Couldn't allocate %d bytes of OCL HS summing memory (error: %i)!\n", sizeof(int32_t) *  nr_pages_total,oclResult);
        return(RADPUL_OCL_MEM_ALLOC_DEVICE);
    }


    // prepare parameters
    oclResult = clSetKernelArg(oclKernelFillIntBuffer, 0, sizeof(nr_pages_total), &nr_pages_total);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: HSFIB (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    int32_t fillValue = 0.0f;
    oclResult = clSetKernelArg(oclKernelFillIntBuffer, 1, sizeof(int32_t), &fillValue);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: HSFIB (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelFillIntBuffer, 2, sizeof(dirtyDeviceBuffer), &dirtyDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: HSFIB (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    // launch kernel grid
    oclResult = clEnqueueNDRangeKernel(oclQueue, oclKernelFillIntBuffer, 1, NULL, workItemAmountFIB, NULL, 0, NULL, NULL);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel setup: HSFIB (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_INVOKE);
    }


    // wait for all enqueued commands to finish
    oclResult = clFinish(oclQueue);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL synchronization: HSFFB,HSFIB (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_INVOKE);
    }
    logMessage(debug, true, "OpenCL FFB,FIB kernel execution successful...\n");


    // copy thresholds to device
    oclResult = clEnqueueWriteBuffer(oclQueue, harmomicSummingThresholdsDeviceBuffer, CL_FALSE, 0, sizeof(float) * 5, thresholds, 0 , NULL, NULL);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL host->device HS thresholds data transfer (error: %i)\n", oclResult);
        return(RADPUL_OCL_MEM_COPY_HOST_DEVICE);
    }

    // Execute kernel to perform harmonic summing (with some gaps where sumspec target values would overlap

    // somehow this seems to work better than y=16 , x = (l2-l1)/16
    // anyway we have to use a 2 dim because index in each dim is limited.
    // x = 16 work groups is rather arbitrarily, there is no algorithmic reason for the value 16
    int count=(l2 - l1) / workGroupSizeHS[0];
    if((l2-l1) % workGroupSizeHS[0] !=0 ) {
      /* add one block if not perfectly aligned */
      count++;
    }

  
    const size_t workItemAmountHS[] = {16 * workGroupSizeHS[0], count};

    // first kernel in first stream
    // TODO: check effect on performance of having concurrent streams (create second command queue)
    logMessage(debug, true, "Executing harmonic summing OpenCL kernel (%u work items each in %u work groups)...\n", workGroupSizeHS[0], (workItemAmountHS[0] * workItemAmountHS[1]) / workGroupSizeHS[0]);

    // prepare parameters
    for(i = 1; i < 5; ++i) {
        oclResult = clSetKernelArg(oclKernelHarmonicSumming, i - 1, sizeof(sumspecDev[i]), &sumspecDev[i]);
        if(CL_SUCCESS != oclResult) {
            logMessage(error, true, "Error during OpenCL kernel parameter setup: HS (error: %d)\n", oclResult);
            return(RADPUL_OCL_KERNEL_SETUP);
        }
    }

    oclResult = clSetKernelArg(oclKernelHarmonicSumming, 4, sizeof(powerSpectrumDeviceBuffer), &powerSpectrumDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: HS (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelHarmonicSumming, 5, sizeof(dirtyDeviceBuffer), &dirtyDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: HS (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }


    oclResult = clSetKernelArg(oclKernelHarmonicSumming, 6, sizeof(harmomicSumminghLUTDeviceBuffer), &harmomicSumminghLUTDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: HS (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelHarmonicSumming, 7, sizeof(harmomicSummingkLUTDeviceBuffer), &harmomicSummingkLUTDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: HS (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelHarmonicSumming, 8, sizeof(harmomicSummingThresholdsDeviceBuffer), &harmomicSummingThresholdsDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: HS (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelHarmonicSumming, 9, sizeof(window_2), &window_2);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: HS (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelHarmonicSumming, 10, sizeof(fundamental_idx_hi), &fundamental_idx_hi);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: HS (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelHarmonicSumming, 11, sizeof(harmonic_idx_hi), &harmonic_idx_hi);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: HS (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    // launch kernel grid
    oclResult = clEnqueueNDRangeKernel(oclQueue, oclKernelHarmonicSumming, 2, NULL, workItemAmountHS, workGroupSizeHS, 0, NULL, NULL);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel setup: HS (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_INVOKE);
    }

    // execute second kernel, this time to fill the gaps

    l1=0;
    l2=(((harmonic_idx_hi -1 + 12) >> 4 ) +1) ;
    count=(l2 - l1) / (2 * workGroupSizeHSG[0]);
    if((l2 - l1) %  (2 * workGroupSizeHSG[0]) != 0) {
      count++;
    }
  
    const size_t workItemAmountHSG[] = {16 * workGroupSizeHSG[0], count};

    // prepare parameters
    for(i = 1; i < 5; ++i) {
        oclResult = clSetKernelArg(oclKernelHarmonicSummingGaps, i - 1, sizeof(sumspecDev[i]), &sumspecDev[i]);
        if(CL_SUCCESS != oclResult) {
            logMessage(error, true, "Error during OpenCL kernel parameter setup: HSG (error: %d)\n", oclResult);
            return(RADPUL_OCL_KERNEL_SETUP);
        }
    }

    logMessage(debug, true, "Executing harmonic summing gaps OpenCL kernel (%u work items each in %u work groups)...\n", workGroupSizeHSG[0], (workItemAmountHSG[0] * workItemAmountHSG[1]) / workGroupSizeHSG[0]);

    oclResult = clSetKernelArg(oclKernelHarmonicSummingGaps, 4, sizeof(powerSpectrumDeviceBuffer), &powerSpectrumDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: HSG (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelHarmonicSummingGaps, 5, sizeof(dirtyDeviceBuffer), &dirtyDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: HS (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelHarmonicSummingGaps, 6, sizeof(harmomicSumminghLUTDeviceBuffer), &harmomicSumminghLUTDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: HSG (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelHarmonicSummingGaps, 7, sizeof(harmomicSummingkLUTDeviceBuffer), &harmomicSummingkLUTDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: HSG (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelHarmonicSummingGaps, 8, sizeof(harmomicSummingThresholdsDeviceBuffer), &harmomicSummingThresholdsDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: HSG (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelHarmonicSummingGaps, 9, sizeof(window_2), &window_2);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: HSG (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelHarmonicSummingGaps, 10, sizeof(fundamental_idx_hi), &fundamental_idx_hi);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: HSG (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    oclResult = clSetKernelArg(oclKernelHarmonicSummingGaps, 11, sizeof(harmonic_idx_hi), &harmonic_idx_hi);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel parameter setup: HSG (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_SETUP);
    }

    // launch kernel grid
    oclResult = clEnqueueNDRangeKernel(oclQueue, oclKernelHarmonicSummingGaps, 2, NULL, workItemAmountHSG, workGroupSizeHSG, 0, NULL, NULL);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL kernel setup: HSG (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_INVOKE);
    }


    // wait for all enqueued commands to finish
    oclResult = clFinish(oclQueue);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL synchronization: HS/HSG (error: %d)\n", oclResult);
        return(RADPUL_OCL_KERNEL_INVOKE);
    }
    logMessage(debug, true, "OpenCL HS/HSG kernel execution successful...\n");




   // copy back dirty page flags to memory

   dirtyTmp = (int32_t*) malloc (nr_pages_total * sizeof(int32_t));
   if(dirtyTmp == NULL) {
      logMessage(error, true, "Couldn't allocate %d bytes of memory for temp mem (HS).\n", nr_pages_total * sizeof(int32_t) );
      return(RADPUL_EMEM);
   };
   oclResult = clEnqueueReadBuffer(oclQueue, dirtyDeviceBuffer, CL_TRUE, 0,sizeof(int32_t) * nr_pages_total,dirtyTmp, 0, NULL, NULL);

   if(oclResult != CL_SUCCESS) {
        logMessage(error, true, "Couldn't release OpenCL memory object: HS (dirty) (error: %i)\n", oclResult);
        return(RADPUL_OCL_MEM_FREE_DEVICE);
   }


   int dirty_idx_min[5] = {0,0,0,0,0};
   int dirty_idx_max[5] = {0,0,0,0,0};
   int d,d_min,d_max;
       
 
   k=0;
   for(i=0 ; i < 5 ; i++) {
      d_min=nr_pages;
      d_max=-1;
      // find the first dirty page
      for(j=0; j < nr_pages ; j++) {
 	  d=dirty[i][j] = dirtyTmp[k++];
          if(d!=0) {
 	      d_min = j ; 
              d_max = j;
              j++; 
              break;
         }
      }
      // go thru the rest and record the last dirty page we find
      for(    ; j < nr_pages ; j++) {
          d=dirty[i][j] = dirtyTmp[k++];
          if(d!=0) {
              d_max = j ;                          
          }
     }

     dirty_idx_min[i]=d_min;
     dirty_idx_max[i]=d_max;
 
   } 		



    /* copy back the results from the OCL kernel.
     * make sure to copy only those cells from sumspec
     * (including the "1st harmonics" powerspectrum itself)
     * that have a chance to include a candidate that makes it to the toplist
     *
     */


    for(i = 0; i < 5; i++) {

        /* no need to copy anything if there is no potential candidate at all */
        if (dirty_idx_max[i]!=-1) {
            size_t seg_length = (dirty_idx_max[i]-dirty_idx_min[i] +1)  << LOG_PS_PAGE_SIZE;
            size_t seg_offset =  dirty_idx_min[i] << LOG_PS_PAGE_SIZE;
            // clip the segment to be copied at the max length of the array
            size_t seg_length_limit = fundamental_idx_hi - seg_offset;
            if (seg_length > seg_length_limit) {
                seg_length = seg_length_limit;
            }
            
            oclResult = clEnqueueReadBuffer(oclQueue, sumspecDev[i], CL_TRUE, sizeof(float) * seg_offset, sizeof(float) * seg_length, sumspec[i]+seg_offset , 0, NULL, NULL);
            if(CL_SUCCESS != oclResult) {
                logMessage(error, true, "Error during OpenCL device->host powerspectrum transfer (error: %d)\n", oclResult);
                return(RADPUL_OCL_MEM_COPY_DEVICE_HOST);
            }
        }
    }


    free(dirtyTmp);
    oclResult = clReleaseMemObject(dirtyDeviceBuffer);
    if(oclResult != CL_SUCCESS) {
        logMessage(error, true, "Error freeing OCL HS device memory (error: %d)\n", oclResult);
        return(RADPUL_OCL_MEM_FREE_DEVICE);
    }


    return(0);
}


int tear_down_harmonic_summing(float **sumspec, int32_t** dirty)
{
    cl_int oclResult;
    int i;

    // clean up. (0th element is powerspectrum, freed separately (see below) )
    for(i = 1; i < 5; i++) {
        free(sumspec[i]);
    }

    for(i = 1; i < 5; i++) {
        oclResult = clReleaseMemObject(sumspecDev[i]);
        if(CL_SUCCESS != oclResult) {
            logMessage(error, true, "Couldn't release OpenCL memory object: HS (error: %i)\n", oclResult);
            return(RADPUL_OCL_MEM_FREE_DEVICE);
        }
    }


    for(i = 0; i < 5; i++) {
      free(dirty[i]);
    }



    // free device buffers
    oclResult = clReleaseMemObject(harmomicSumminghLUTDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(warn, true, "Couldn't release OpenCL memory object: HSHL (error: %i)\n", oclResult);
    }

    oclResult = clReleaseMemObject(harmomicSummingkLUTDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(warn, true, "Couldn't release OpenCL memory object: HSKL (error: %i)\n", oclResult);
    }

    oclResult = clReleaseMemObject(harmomicSummingThresholdsDeviceBuffer);
    if(CL_SUCCESS != oclResult) {
        logMessage(warn, true, "Couldn't release OpenCL memory object: HST (error: %i)\n", oclResult);
    }

    return(0);
}


int shutdown_ocl()
{
    cl_int oclResult;

    oclResult = releaseProgramKernels();

    if(NULL != oclQueue) {
        oclResult = clReleaseCommandQueue(oclQueue);
        if(CL_SUCCESS != oclResult) {
            logMessage(warn, true, "Couldn't release OpenCL command queue (error: %i)!\n", oclResult);
        }
    }


    if(NULL != oclContext) {
        oclResult = clReleaseContext(oclContext);
        if(CL_SUCCESS != oclResult) {
            logMessage(warn, true, "Couldn't release OpenCL context (error: %i)!\n", oclResult);
        }
    }

    logMessage(info, true, "OpenCL shutdown complete!\n");

    return(CL_SUCCESS);
}


cl_int createProgramKernels(const cl_device_id oclDeviceId, const cl_uint oclPlatformVendor)
{
    cl_int oclResult;

    // prepare compile options
    int maxLength = 2 * strlen(defaultCompileOptions);
    char *compileOptions = (char*) calloc(maxLength, sizeof(char));
    if(!compileOptions) {
        logMessage(error, true, "Couldn't prepare OpenCL compile options!\n");
        return(RADPUL_OCL_PROGRAM_CREATE);
    }

    // add conditional vendor-specific options
    const char *vendorOptions = "";
    if(VENDOR_NVIDIA == oclPlatformVendor) {
        vendorOptions = "-cl-nv-verbose";
    }

    // compile final set of options
    int result = snprintf(compileOptions, maxLength, defaultCompileOptions, vendorOptions, workGroupSizeTSMR[0], workGroupSizePS[0],workGroupSizePS_R3_R2C[0],workGroupSizeHS[0], lrint(log2(workGroupSizeHS[0])), LOG_PS_PAGE_SIZE);
    if(result < 0 || result >= maxLength) {
        logMessage(error, true, "Couldn't parameterize OpenCL compile options!\n");
        return(RADPUL_OCL_PROGRAM_CREATE);
    }
    else {
        logMessage(debug, true, "Using the following parameterized OpenCL compile options:\n");
        logMessage(debug, false, "%s\n", compileOptions);
    }

    // load source and build it
    const char *programSources[] = {deviceSinLUTLookup,
                                    kernelTimeSeriesModulation,
                                    kernelTimeSeriesLengthModulated,
                                    kernelTimeSeriesResampling,
                                    kernelTimeSeriesMeanReduction,
                                    kernelTimeSeriesPadding,
                                    kernelTimeSeriesPaddingTranspose,
                                    kernelPowerSpectrum,
                                    kernelPowerSpectrum_radix3_r2c,
                                    kernelHarmonicSumming,
                                    kernelHarmonicSummingGaps,
                                    kernelFillFloatBuffer,
                                    kernelFillIntBuffer};

    oclProgramDemodBinary = clCreateProgramWithSource(oclContext, 13, programSources, NULL, &oclResult);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Couldn't create OpenCL program (error: %i)!\n", oclResult);
        if(CL_OUT_OF_HOST_MEMORY == oclResult || CL_OUT_OF_RESOURCES == oclResult) {
            return(RADPUL_OCL_MEM_ALLOC_DEVICE);
        }
        return(RADPUL_OCL_PROGRAM_CREATE);
    }

    oclResult = clBuildProgram(oclProgramDemodBinary, NULL, NULL, compileOptions, NULL, NULL);

    // free resources
    free(compileOptions);

#ifndef NDEBUG
    // store build log when debugging
    size_t buildLogLength = 0;
    clGetProgramBuildInfo(oclProgramDemodBinary, oclDeviceId, CL_PROGRAM_BUILD_LOG, NULL, NULL, &buildLogLength);
    if(1 < buildLogLength) {
        char *buildLog = (char*) calloc(buildLogLength, sizeof(char));
        if(buildLog) {
            clGetProgramBuildInfo(oclProgramDemodBinary, oclDeviceId, CL_PROGRAM_BUILD_LOG, buildLogLength * sizeof(char), buildLog, NULL);
            logMessage(debug, true, "OpenCL build log:\n\n%s\n\n", buildLog);
            free(buildLog);
        }
        else {
            logMessage(warn, true, "Couldn't allocate OpenCL build log buffer!\n");
        }
    }
#endif

    // finally, check build outcome
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Couldn't build OpenCL program (error: %i)!\n", oclResult);
        if(CL_OUT_OF_HOST_MEMORY == oclResult || CL_OUT_OF_RESOURCES == oclResult) {
            return(RADPUL_OCL_MEM_ALLOC_DEVICE);
        }
        return(RADPUL_OCL_PROGRAM_BUILD);
    }

    // prepare kernels
    oclKernelTimeSeriesModulation = clCreateKernel(oclProgramDemodBinary, "kernelTimeSeriesModulation", &oclResult);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Couldn't create OpenCL kernel: TSM (error: %i)!\n", oclResult);
        if(CL_OUT_OF_HOST_MEMORY == oclResult || CL_OUT_OF_RESOURCES == oclResult) {
            return(RADPUL_OCL_MEM_ALLOC_DEVICE);
        }
        return(RADPUL_OCL_KERNEL_CREATE);
    }

    oclKernelTimeSeriesLengthModulated = clCreateKernel(oclProgramDemodBinary, "kernelTimeSeriesLengthModulated", &oclResult);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Couldn't create OpenCL kernel: TSLM (error: %i)!\n", oclResult);
        if(CL_OUT_OF_HOST_MEMORY == oclResult || CL_OUT_OF_RESOURCES == oclResult) {
            return(RADPUL_OCL_MEM_ALLOC_DEVICE);
        }
        return(RADPUL_OCL_KERNEL_CREATE);
    }

    oclKernelTimeSeriesResampling = clCreateKernel(oclProgramDemodBinary, "kernelTimeSeriesResampling", &oclResult);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Couldn't create OpenCL kernel: TSR (error: %i)!\n", oclResult);
        if(CL_OUT_OF_HOST_MEMORY == oclResult || CL_OUT_OF_RESOURCES == oclResult) {
            return(RADPUL_OCL_MEM_ALLOC_DEVICE);
        }
        return(RADPUL_OCL_KERNEL_CREATE);
    }

    oclKernelTimeSeriesMeanReduction = clCreateKernel(oclProgramDemodBinary, "kernelTimeSeriesMeanReduction", &oclResult);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Couldn't create OpenCL kernel: TSMR (error: %i)!\n", oclResult);
        if(CL_OUT_OF_HOST_MEMORY == oclResult || CL_OUT_OF_RESOURCES == oclResult) {
            return(RADPUL_OCL_MEM_ALLOC_DEVICE);
        }
        return(RADPUL_OCL_KERNEL_CREATE);
    }

    oclKernelTimeSeriesPadding = clCreateKernel(oclProgramDemodBinary, "kernelTimeSeriesPadding", &oclResult);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Couldn't create OpenCL kernel: TSMP (error: %i)!\n", oclResult);
        if(CL_OUT_OF_HOST_MEMORY == oclResult || CL_OUT_OF_RESOURCES == oclResult) {
            return(RADPUL_OCL_MEM_ALLOC_DEVICE);
        }
        return(RADPUL_OCL_KERNEL_CREATE);
    }

    oclKernelTimeSeriesPaddingTranspose = clCreateKernel(oclProgramDemodBinary, "kernelTimeSeriesPaddingTranspose", &oclResult);
    if(CL_SUCCESS != oclResult) {
      logMessage(error, true, "Couldn't create OpenCL kernel: TSMP_T (error: %i)!\n", oclResult);
      if(CL_OUT_OF_HOST_MEMORY == oclResult || CL_OUT_OF_RESOURCES == oclResult) {
          return(RADPUL_OCL_MEM_ALLOC_DEVICE);
      }
      return(RADPUL_OCL_KERNEL_CREATE);
    }



    oclKernelPowerSpectrum = clCreateKernel(oclProgramDemodBinary, "kernelPowerSpectrum", &oclResult);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Couldn't create OpenCL kernel: PS (error: %i)!\n", oclResult);
        if(CL_OUT_OF_HOST_MEMORY == oclResult || CL_OUT_OF_RESOURCES == oclResult) {
            return(RADPUL_OCL_MEM_ALLOC_DEVICE);
        }
        return(RADPUL_OCL_KERNEL_CREATE);
    }

    oclKernelPowerSpectrum_radix3_r2c = clCreateKernel(oclProgramDemodBinary, "kernelPowerSpectrum_radix3_r2c", &oclResult);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Couldn't create OpenCL kernel: PS_R3_R2C (error: %i)!\n", oclResult);
        if(CL_OUT_OF_HOST_MEMORY == oclResult || CL_OUT_OF_RESOURCES == oclResult) {
            return(RADPUL_OCL_MEM_ALLOC_DEVICE);
        }
        return(RADPUL_OCL_KERNEL_CREATE);
    }

    oclKernelHarmonicSumming = clCreateKernel(oclProgramDemodBinary, "kernelHarmonicSumming", &oclResult);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Couldn't create OpenCL kernel: HS (error: %i)!\n", oclResult);
        if(CL_OUT_OF_HOST_MEMORY == oclResult || CL_OUT_OF_RESOURCES == oclResult) {
            return(RADPUL_OCL_MEM_ALLOC_DEVICE);
        }
        return(RADPUL_OCL_KERNEL_CREATE);
    }

    oclKernelHarmonicSummingGaps = clCreateKernel(oclProgramDemodBinary, "kernelHarmonicSummingGaps", &oclResult);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Couldn't create OpenCL kernel: HSG (error: %i)!\n", oclResult);
        if(CL_OUT_OF_HOST_MEMORY == oclResult || CL_OUT_OF_RESOURCES == oclResult) {
            return(RADPUL_OCL_MEM_ALLOC_DEVICE);
        }
        return(RADPUL_OCL_KERNEL_CREATE);
    }

    oclKernelFillFloatBuffer = clCreateKernel(oclProgramDemodBinary, "kernelFillFloatBuffer", &oclResult);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Couldn't create OpenCL kernel: FFB (error: %i)!\n", oclResult);
        if(CL_OUT_OF_HOST_MEMORY == oclResult || CL_OUT_OF_RESOURCES == oclResult) {
            return(RADPUL_OCL_MEM_ALLOC_DEVICE);
        }
        return(RADPUL_OCL_KERNEL_CREATE);
    }


    oclKernelFillIntBuffer = clCreateKernel(oclProgramDemodBinary, "kernelFillIntBuffer", &oclResult);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Couldn't create OpenCL kernel: FIB (error: %i)!\n", oclResult);
        if(CL_OUT_OF_HOST_MEMORY == oclResult || CL_OUT_OF_RESOURCES == oclResult) {
            return(RADPUL_OCL_MEM_ALLOC_DEVICE);
        }
        return(RADPUL_OCL_KERNEL_CREATE);
    }


    return CL_SUCCESS;
}


cl_int releaseProgramKernels()
{
    cl_int oclResult;
    cl_int oclLastError = CL_SUCCESS;

    if(NULL != oclKernelTimeSeriesModulation) {
        oclResult = clReleaseKernel(oclKernelTimeSeriesModulation);
        if(CL_SUCCESS != oclResult) {
            logMessage(warn, true, "Couldn't release OpenCL kernel TSM (error: %i)!\n", oclResult);
            oclLastError = oclResult;
        }
    }

    if(NULL != oclKernelTimeSeriesLengthModulated) {
        oclResult = clReleaseKernel(oclKernelTimeSeriesLengthModulated);
        if(CL_SUCCESS != oclResult) {
            logMessage(warn, true, "Couldn't release OpenCL kernel TSLM (error: %i)!\n", oclResult);
            oclLastError = oclResult;
        }
    }

    if(NULL != oclKernelTimeSeriesResampling) {
        oclResult = clReleaseKernel(oclKernelTimeSeriesResampling);
        if(CL_SUCCESS != oclResult) {
            logMessage(warn, true, "Couldn't release OpenCL kernel TSR (error: %i)!\n", oclResult);
            oclLastError = oclResult;
        }
    }

    if(NULL != oclKernelTimeSeriesMeanReduction) {
        oclResult = clReleaseKernel(oclKernelTimeSeriesMeanReduction);
        if(CL_SUCCESS != oclResult) {
            logMessage(warn, true, "Couldn't release OpenCL kernel TSMR (error: %i)!\n", oclResult);
            oclLastError = oclResult;
        }
    }

    if(NULL != oclKernelTimeSeriesPadding) {
        oclResult = clReleaseKernel(oclKernelTimeSeriesPadding);
        if(CL_SUCCESS != oclResult) {
            logMessage(warn, true, "Couldn't release OpenCL kernel TSP (error: %i)!\n", oclResult);
            oclLastError = oclResult;
        }
    }

    if(NULL != oclKernelPowerSpectrum) {
            oclResult = clReleaseKernel(oclKernelPowerSpectrum);
            if(CL_SUCCESS != oclResult) {
                logMessage(warn, true, "Couldn't release OpenCL kernel PS (error: %i)!\n", oclResult);
                oclLastError = oclResult;
            }
    }

    if(NULL != oclKernelHarmonicSumming) {
           oclResult = clReleaseKernel(oclKernelHarmonicSumming);
           if(CL_SUCCESS != oclResult) {
               logMessage(warn, true, "Couldn't release OpenCL kernel HS (error: %i)!\n", oclResult);
               oclLastError = oclResult;
           }
    }

    if(NULL != oclKernelHarmonicSummingGaps) {
           oclResult = clReleaseKernel(oclKernelHarmonicSummingGaps);
           if(CL_SUCCESS != oclResult) {
               logMessage(warn, true, "Couldn't release OpenCL kernel HSG (error: %i)!\n", oclResult);
               oclLastError = oclResult;
           }
    }

    if(NULL != oclKernelFillFloatBuffer) {
            oclResult = clReleaseKernel(oclKernelFillFloatBuffer);
            if(CL_SUCCESS != oclResult) {
                logMessage(warn, true, "Couldn't release OpenCL kernel FFB (error: %i)!\n", oclResult);
                oclLastError = oclResult;
            }
    }


    if(NULL != oclKernelFillIntBuffer) {
            oclResult = clReleaseKernel(oclKernelFillIntBuffer);
            if(CL_SUCCESS != oclResult) {
                logMessage(warn, true, "Couldn't release OpenCL kernel FIB (error: %i)!\n", oclResult);
                oclLastError = oclResult;
            }
    }


    if(NULL != oclProgramDemodBinary) {
        oclResult = clReleaseProgram(oclProgramDemodBinary);
        if(CL_SUCCESS != oclResult) {
            logMessage(warn, true, "Couldn't release OpenCL program (error: %i)!\n", oclResult);
            oclLastError = oclResult;
        }
    }

    return oclLastError != CL_SUCCESS ? oclLastError : CL_SUCCESS;
}
