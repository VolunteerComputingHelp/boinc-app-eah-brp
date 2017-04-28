/***************************************************************************
 *   Copyright (C) 2009 by Oliver Bock                                     *
 *   oliver.bock[AT]aei.mpg.de                                             *
 *                                                                         *
 *   This file is part of Einstein@Home (Radio Pulsar Edition).            *
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


#include "cuda_utilities.h"

#include <cstdio>
#include <stddef.h>
#include <string.h>
#include <fenv.h>
#include <stdio.h>

#include "../../demod_binary.h"

#ifndef BOINCIFIED
int running_standalone(void) { return 1; };
#else
#include "boinc_api.h"

int running_standalone(void) { return boinc_is_standalone(); };

// helper function that really belongs into boinc API rather than app code

// This version is compatible with older clients.
// Usage:
// Pass the argc and argv received from the BOINC client
// device Id found in init_data.xml takes precedence over command line
//
// returns
// - 0 if success
// - ERR_FOPEN if init_data.xml missing
// - ERR_XML_PARSE if can't parse init_data.xml
// - ERR_NOT_FOUND if unable to get gpu device_num
//

int boinc_get_cuda_device_id(
    int argc, const char** argv, int * deviceId)

{
    int retval,i;
    APP_INIT_DATA aid;
    int gpu_device_num = -1;

    retval = boinc_parse_init_data_file();
    if (retval) {
        fprintf(stderr, "Error opening or parsing %s for GPU device ID\n", INIT_DATA_FILE);
        return retval;
    }
    boinc_get_init_data(aid);

    gpu_device_num = aid.gpu_device_num;
    if (gpu_device_num < 0) {
            // older versions of init_data.xml don't have gpu_device_num field
            for (i=0; i<argc-1; i++) {
                if ((!strcmp(argv[i], "--device")) || (!strcmp(argv[i], "-device"))) {
                    gpu_device_num = atoi(argv[i+1]);
                    break;
                }
            }
    }

    if (gpu_device_num < 0) {
        fprintf(stderr, "GPU device # not found in %s or command line\n", INIT_DATA_FILE);
        return ERR_NOT_FOUND;
    }
    *deviceId = gpu_device_num;
    return retval;
}
#endif


  /**
    TODO: add more error checks!
    TODO: add "percent GPU occupied" as factor to maxGFlops (if possible)
        -> alternatively run all tasks in exlusive mode and check that!
        -> use cudaDriverGetVersion to determine the CUDA runtime version (>=2.2 only)
   */

int findBestFreeDevice(const int minMajorRevision,
                       const int minMinorRevision,
                       const unsigned int minGlobalMemoryBytes,
                       const int needExclusive)
{
    CUdevice cudCurrentDevicePtr = NULL;
    CUresult cuResult = CUDA_SUCCESS;

    // find best GPU (with maximum GFLOPS)
    int cudDeviceCount = 0;
    float cudGFlops = 0;
    int cudMaxGFlops = 0;
    int cudMaxGFlopsDevice = -1;

    logMessage(debug, true, "Analyzing available CUDA devices...\n");
    cuResult = cuDeviceGetCount(&cudDeviceCount);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(warn, false, "Number of available CUDA device couldn't be determined!\n");
    }
    if(cudDeviceCount == 0) {
        // return "no device available" (-1)
        return cudMaxGFlopsDevice;
    }

    for(int cudCurrentDevice = 0; cudCurrentDevice < cudDeviceCount; ++cudCurrentDevice) {
        // get current device
        cuResult = cuDeviceGet(&cudCurrentDevicePtr, cudCurrentDevice);
        if(cuResult != CUDA_SUCCESS) {
            logMessage(warn, true, "Couldn't acquire CUDA device #%i (error: %i)! Trying next one...\n", cudCurrentDevice, cuResult);
            continue;
        }

        // TODO: refactor this stuff (see findBestFreeDevice())

        // get device properties
        int compcapMajor = 0;
        int compcapMinor = 0;
        int multiProcessorCount = 0;
        int coreCount = 0;
        int clockRate = 0;
        int flopsPerClockTick = 0;
        char deviceName[256] = {0};
        size_t totalGlobalMem = 0;
        int computeMode = 0;

        // number of multi processors
        cuResult = cuDeviceGetAttribute(&multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cudCurrentDevicePtr);
        if(cuResult != CUDA_SUCCESS) {
            logMessage(warn, true, "Couldn't retrieve multi processor count property of device #%i (error: %i)! Trying next one...\n", cudCurrentDevice, cuResult);
            continue;
        }

        // clock rate
        cuResult = cuDeviceGetAttribute(&clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, cudCurrentDevicePtr);
        if(cuResult != CUDA_SUCCESS) {
            logMessage(warn, true, "Couldn't retrieve clock rate property of device #%i (error: %i)! Trying next one...\n", cudCurrentDevice, cuResult);
            continue;
        }

        // compute capability
        cuResult = cuDeviceComputeCapability(&compcapMajor, &compcapMinor, cudCurrentDevicePtr);
        if(cuResult != CUDA_SUCCESS) {
            logMessage(warn, true, "Couldn't retrieve compute capability of device #%i (error: %i)! Trying next one...\n", cudCurrentDevice, cuResult);
            continue;
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
        cuResult = cuDeviceGetName(deviceName, 256, cudCurrentDevicePtr);
        if(cuResult != CUDA_SUCCESS) {
            logMessage(warn, true, "Couldn't retrieve name of device #%i (error: %i)!\n", cudCurrentDevice, cuResult);
            strcpy(deviceName, "UNKNOWN");
        }

        // compute peak FLOPS
        cudGFlops = coreCount * clockRate * flopsPerClockTick * 1e-6;
        logMessage(debug, false, "Device #%i (%s): %i CUDA cores / %.2f GFLOPS\n", cudCurrentDevice, deviceName, coreCount, cudGFlops);

        // global memory
        cuResult = cuDeviceTotalMem(&totalGlobalMem, cudCurrentDevicePtr);
        if(cuResult != CUDA_SUCCESS) {
            logMessage(warn, true, "Couldn't retrieve total global memory property of device #%i (error: %i)!\n", cudCurrentDevice, cuResult);
        }

        // compute mode
        cuResult = cuDeviceGetAttribute(&computeMode,  CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, cudCurrentDevicePtr);
        if(cuResult != CUDA_SUCCESS) {
            logMessage(warn, true, "Couldn't retrieve compute mode property of device #%i (error: %i)!\n", cudCurrentDevice, cuResult);
        }

        if(cudGFlops > cudMaxGFlops) {
            // check device against requirements
            if(computeMode ==  CU_COMPUTEMODE_PROHIBITED) {
                logMessage(debug, false, "Device #%i (%s) currently unavailable (usage prohibited)!\n", cudCurrentDevice, deviceName);
                continue;
            }
            if(!(compcapMajor >= minMajorRevision && compcapMinor >= minMinorRevision)) {
                logMessage(debug, false, "Device #%i (%s) doesn't meet requested capabilities! Found: %i.%i\n",
                           cudCurrentDevice, deviceName, compcapMajor, compcapMinor);
                continue;
            }
            if(totalGlobalMem < minGlobalMemoryBytes) {
                logMessage(debug, false, "Device #%i (%s) doesn't meet global memory requirements: %i\n",
                           cudCurrentDevice, deviceName, totalGlobalMem);
                continue;
            }
            if(needExclusive != 0 && computeMode == CU_COMPUTEMODE_EXCLUSIVE) {
                logMessage(debug, false, "Device #%i (%s) not available in exclusive mode!\n", cudCurrentDevice, deviceName);
                continue;
            }
            // finally set device as current best option
            cudMaxGFlops = cudGFlops;
            cudMaxGFlopsDevice = cudCurrentDevice;
        }
    }

    return cudMaxGFlopsDevice;
}


void printDeviceGlobalMemStatus(const ERP_LOGLEVEL logLevel, const bool followUp)
{
    CUresult cuResult = CUDA_SUCCESS;
    size_t cudGlobalMemFree = 0;
    size_t cudGlobalMemTotal = 0;

    // get mem info
    cuResult = cuMemGetInfo(&cudGlobalMemFree, &cudGlobalMemTotal);
    if(cuResult == CUDA_SUCCESS) {
        cudGlobalMemFree = (size_t) (cudGlobalMemFree / 1048000.0f + 0.5f);
        cudGlobalMemTotal = (size_t) (cudGlobalMemTotal / 1048000.0f + 0.5f);
        static const size_t cudGlobalMemUsedByOthers = cudGlobalMemTotal - cudGlobalMemFree;
        size_t cudGlobalMemUsedTotal = cudGlobalMemTotal - cudGlobalMemFree;
        size_t cudGlobalMemUsedByUs = cudGlobalMemUsedTotal - cudGlobalMemUsedByOthers;
        logMessage(logLevel, !followUp, "Used in total: %u MB (%u MB free / %u MB total) -> Used by this application (assuming a single GPU task): %u MB\n", cudGlobalMemUsedTotal, cudGlobalMemFree, cudGlobalMemTotal, cudGlobalMemUsedByUs);
    }
    else {
        logMessage(warn, true, "CUDA global memory status couldn't be determined (error: %i)!\n", cuResult);
    }
}


void printHostRoundingMode(const ERP_LOGLEVEL logLevel, const bool showLevel)
{
    switch(fegetround()) {
        case FE_TONEAREST:
            logMessage(logLevel, showLevel, "Current host floating point rounding mode: FE_TONEAREST\n");
            break;
        case FE_TOWARDZERO:
            logMessage(logLevel, showLevel, "Current host floating point rounding mode: FE_TOWARDZERO\n");
            break;
        case FE_DOWNWARD:
            logMessage(logLevel, showLevel, "Current host floating point rounding mode: FE_DOWNWARD\n");
            break;
        case FE_UPWARD:
            logMessage(logLevel, showLevel, "Current host floating point rounding mode: FE_UPWARD\n");
            break;
        default:
            logMessage(warn, showLevel, "Couldn't determine host floating point rounding mode!");
    }
}


int dumpFloatDeviceBufferToTextFile(const CUdeviceptr buffer,
                                    const size_t size,
                                    const char *filename)
{
    CUresult cuResult;
    float *hostBuffer;

    hostBuffer = (float*) calloc(size, sizeof(float));
    if(NULL == hostBuffer) {
        logMessage(error, true, "Error allocating CUDA host buffer: %i bytes!\n", sizeof(float) * size);
        return(RADPUL_CUDA_MEM_ALLOC_HOST);
    }
    logMessage(debug, true, "Allocated CUDA host buffer: %i bytes\n", sizeof(float) * size);

    cuResult = cuMemcpyDtoH(hostBuffer, buffer, size * sizeof(float));
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error during CUDA device->host data transfer (error: %d)\n", cuResult);
        return(RADPUL_CUDA_MEM_COPY_DEVICE_HOST);
    }

    logMessage(debug, true, "CUDA device->host buffer transfer successful...\n");

    FILE *output = fopen(filename, "w");
    if(NULL == output) {
        logMessage(error, true, "Error opening file \"%s\" for buffer dump!\n", filename);
        return(RADPUL_EFILE);
    }
    for(size_t i = 0; i < size; ++i) {
        fprintf(output, "%e\n", hostBuffer[i]);
    }
    fclose(output);

    logMessage(debug, true, "Successfully wrote buffer to \"%s\"...\n", filename);

    free(hostBuffer);

    return(CUDA_SUCCESS);
}
