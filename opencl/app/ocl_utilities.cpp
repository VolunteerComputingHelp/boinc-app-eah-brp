/***************************************************************************
 *   Copyright (C) 2011 by Oliver Bock                                     *
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

#include "ocl_utilities.h"

#include <cstdio>
#include <string>

#ifdef __APPLE__
    #include <OpenCL/cl_ext.h>
#else
    #include <CL/cl_ext.h>
#endif

#include "../../demod_binary.h"

// defined in OpenCL 1.1 (but Apple is still using 1.0)
#ifndef CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV
#define CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV 0x4000
#define CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV 0x4001
#endif

#define OCL_MAX_STRING 256
#define OCL_MAX_STRING_EXT 1024

using namespace std;


cl_device_id findBestFreeDevice(const cl_platform_id platform,
                                const cl_int minMajorVersion,
                                const cl_int minMinorVersion,
                                const cl_ulong minGlobalMemoryBytes,
                                const cl_device_type deviceType)
{
    // find best GPU (with maximum GFLOPS)
    cl_int oclResult = CL_SUCCESS;
    cl_device_id oclDevices[OCL_MAX_DEVICES] = {NULL};
    cl_uint oclDeviceCount = 0;
    cl_device_id oclBestDevice = NULL;
    float oclPerformance = 0.0f;
    float oclMaxPerformance = -1.0f;

    logMessage(debug, true, "Analyzing available OpenCL devices...\n");

    // retrieve all available devices (based on given device type)
    oclResult = clGetDeviceIDs(platform, deviceType, OCL_MAX_DEVICES, oclDevices, &oclDeviceCount);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Couldn't retrieve list of OpenCL devices (error: %i)!\n", oclResult);
        return(NULL);
    }

    // iterate over all devices, check their properties and find most powerful
    for(cl_uint i = 0; i < oclDeviceCount; ++i) {
        // generic OpenCL
        char deviceName[OCL_MAX_STRING] = "UNKNOWN";
        char deviceVersion[OCL_MAX_STRING] = "UNKNOWN";
        char deviceExtensions[OCL_MAX_STRING_EXT] = "UNKNOWN";
        cl_uint compUnits = 0;
        cl_uint clockFreq = 0;
        cl_ulong totalGlobalMem = 0;
        cl_int oclDeviceMajor = 0;
        cl_int oclDeviceMinor = 0;

        // NVIDIA-specific
        bool nvidia = false;
        cl_uint nvCompCapMajor = 0;
        cl_uint nvCompCapMinor = 0;

        // retrieve device properties
        oclResult = clGetDeviceInfo(oclDevices[i], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
        if(CL_SUCCESS != oclResult) {
            logMessage(warn, true, "Couldn't retrieve OpenCL device name (error: %i)!\n", oclResult);
        }
        else {
            logMessage(debug, true, "Evaluating properties of OpenCL device \"%s\"...\n", deviceName);
        }

        oclResult = clGetDeviceInfo(oclDevices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compUnits), &compUnits, NULL);
        if(CL_SUCCESS != oclResult) {
            logMessage(warn, true, "Couldn't retrieve required OpenCL device property of device \"%s\" (error: %i)! Trying next one...\n",
                       deviceName, oclResult);
            continue;
        }
        else {
            logMessage(debug, false, "Number of compute units of OpenCL device \"%s\": %i\n", deviceName, compUnits);
        }

        oclResult = clGetDeviceInfo(oclDevices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clockFreq), &clockFreq, NULL);
        if(CL_SUCCESS != oclResult) {
            logMessage(warn, true, "Couldn't retrieve required OpenCL device property of device \"%s\" (error: %i)! Trying next one...\n",
                       deviceName, oclResult);
            continue;
        }
        else {
            logMessage(debug, false, "Clock frequency of OpenCL device \"%s\": %i\n", deviceName, clockFreq);
        }

        if(0 < minGlobalMemoryBytes) {
            oclResult = clGetDeviceInfo(oclDevices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(totalGlobalMem), &totalGlobalMem, NULL);
            if(CL_SUCCESS != oclResult) {
                logMessage(warn, true, "Couldn't retrieve required OpenCL device property of device \"%s\" (error: %i)! Trying next one...\n",
                           deviceName, oclResult);
                continue;
            }
            else {
                logMessage(debug, false, "Global memory of OpenCL device \"%s\": %i bytes\n", deviceName, totalGlobalMem);
            }
        }

        if(0 < minMajorVersion || 0 < minMinorVersion) {
            oclResult = clGetDeviceInfo(oclDevices[i], CL_DEVICE_VERSION, sizeof(deviceVersion), &deviceVersion, NULL);
            if(CL_SUCCESS != oclResult) {
                logMessage(warn, true, "Couldn't retrieve required OpenCL device property of device \"%s\" (error: %i)! Trying next one...\n",
                           deviceName, oclResult);
                continue;
            }
            else {
                if(2 != sscanf(deviceVersion, "OpenCL %i.%i", &oclDeviceMajor, &oclDeviceMinor)) {
                    logMessage(warn, true, "Couldn't retrieve required OpenCL device version property of device \"%s\"! Trying next one...\n",
                               deviceName);
                    continue;
                }
                else {
                    logMessage(debug, false, "Supported OpenCL version of OpenCL device \"%s\": %i.%i\n", deviceName, oclDeviceMajor, oclDeviceMinor);
                }
            }
        }

        // check vendor-specific stuff
        oclResult = clGetDeviceInfo(oclDevices[i], CL_DEVICE_EXTENSIONS, sizeof(deviceExtensions), &deviceExtensions, NULL);
        if(CL_SUCCESS != oclResult) {
            logMessage(warn, true, "Couldn't retrieve OpenCL device extensions property of device \"%s\" (error: %i)!\n",
                       deviceName, oclResult);
        }
        else {
            // is this a NVIDIA device with extended attributes?
            string extensions(deviceExtensions);
            if(string::npos != extensions.find("cl_nv_device_attribute_query")) {
                logMessage(debug, true, "Evaluating extended NVIDIA properties of OpenCL device \"%s\"...\n", deviceName);
                nvidia = true;
            }
        }

        // check extended NVIDIA device attributes
        if(nvidia) {
            int flopsPerClockTick = 1;

            oclResult = clGetDeviceInfo(oclDevices[i], CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, sizeof(nvCompCapMajor), &nvCompCapMajor, NULL);
            if(CL_SUCCESS != oclResult) {
                logMessage(warn, true, "Couldn't retrieve NVIDIA OpenCL device compute capability (error: %i)!\n", oclResult);
            }
            oclResult = clGetDeviceInfo(oclDevices[i], CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, sizeof(nvCompCapMinor), &nvCompCapMinor, NULL);
            if(CL_SUCCESS != oclResult) {
                logMessage(warn, true, "Couldn't retrieve NVIDIA OpenCL device compute capability (error: %i)!\n", oclResult);
            }

            // count cores, not multiprocessors
            if(nvCompCapMajor == 1) {
                compUnits = compUnits * 8;
                flopsPerClockTick = 3;
            }
            else if(nvCompCapMajor == 2 && nvCompCapMinor == 0) {
                compUnits = compUnits * 32;
                flopsPerClockTick = 2;
            }
            else if(nvCompCapMajor == 2 && nvCompCapMinor >= 0) {
                compUnits = compUnits * 48;
                flopsPerClockTick = 2;
            }

            // calculate actual GFLOPS or stick to what's done for other devices
            oclPerformance = compUnits * clockFreq * flopsPerClockTick * 1e-3;
        }
        else {
            // crude peak performance estimation
            oclPerformance = compUnits * clockFreq;
        }

        logMessage(debug, false, "Peak performance metric value of OpenCL device \"%s\": %.2f\n", deviceName, oclPerformance);

        if(oclPerformance > oclMaxPerformance) {
            // check device against requirements
            if(oclDeviceMajor < minMajorVersion || (oclDeviceMajor == minMajorVersion && oclDeviceMinor < minMinorVersion)) {
                logMessage(debug, true, "Device \"%s\" doesn't meet requested OpenCL version (requested: %i.%i / found: %i.%i)! Trying next one...\n",
                           deviceName, minMajorVersion, minMinorVersion, oclDeviceMajor, oclDeviceMinor);
                continue;
            }
            if(totalGlobalMem < minGlobalMemoryBytes) {
                logMessage(debug, true, "Device \"%s\" doesn't meet global memory requirements (requested: %i bytes / found: %i bytes)! Trying next one...\n",
                           deviceName, minGlobalMemoryBytes, totalGlobalMem);
                continue;
            }
            // finally set device as current best option
            oclMaxPerformance = oclPerformance;
            oclBestDevice = oclDevices[i];
        }
    }

    // show max work items per dimension
    if(NULL != oclBestDevice) {
        cl_uint dimensions = 0;
        oclResult = clGetDeviceInfo(oclBestDevice, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(dimensions), &dimensions, NULL);
        if(CL_SUCCESS != oclResult) {
            logMessage(warn, true, "Couldn't retrieve OpenCL device number of work item dimensions (error: %i)!\n", oclResult);
        }
        else {
            size_t *itemsPerDimension = (size_t*) calloc(dimensions, sizeof(size_t));
            if(NULL == itemsPerDimension) {
                logMessage(warn, true, "Couldn't allocate %d bytes of host memory for OpenCL device work item dimensions sizes!\n", dimensions * sizeof(size_t));
            }
            else {
                oclResult = clGetDeviceInfo(oclBestDevice, CL_DEVICE_MAX_WORK_ITEM_SIZES, dimensions * sizeof(size_t), itemsPerDimension, NULL);
                if(CL_SUCCESS != oclResult) {
                    logMessage(warn, true, "Couldn't retrieve OpenCL device work item dimensions sizes (error: %i)!\n", oclResult);
                }
                else {
                    logMessage(debug, true, "OpenCL device work item dimension sizes of chosen device:\n");
                    for(cl_uint i = 0; i < dimensions; ++i) {
                        logMessage(debug, false, "Dimension %u: %u\n", i, itemsPerDimension[i]);
                    }
                }
                free(itemsPerDimension);
            }
        }
    }

    return(oclBestDevice);
}


cl_int dumpFloatDeviceBufferToTextFile(const cl_command_queue queue,
                                       const cl_mem buffer,
                                       const size_t size,
                                       const char *filename)
{
    cl_int oclResult;
    float *hostBuffer;

    hostBuffer = (float*) calloc(size, sizeof(float));
    if(NULL == hostBuffer) {
        logMessage(error, true, "Error allocating OpenCL host buffer: %i bytes!\n", sizeof(float) * size);
        return(RADPUL_OCL_MEM_ALLOC_HOST);
    }
    logMessage(debug, true, "Allocated OpenCL host buffer: %i bytes\n", sizeof(float) * size);

    oclResult = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, sizeof(float) * size, hostBuffer, 0, NULL, NULL);
    if(CL_SUCCESS != oclResult) {
        logMessage(error, true, "Error during OpenCL device->host buffer transfer (error: %d)\n", oclResult);
        return(RADPUL_OCL_MEM_COPY_DEVICE_HOST);
    }

    logMessage(debug, true, "OpenCL device->host buffer transfer successful...\n");

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

    return(CL_SUCCESS);
}


cl_int checkKernelWorkGroupInfo(const cl_device_id *devicePtr, const cl_kernel *kernelPtr, size_t workGroupSize, size_t *workGroupSizeLimit)
{
    cl_int oclResult;

    // sanity check
    if(NULL == devicePtr || NULL == kernelPtr) {
        logMessage(error, true, "Invalid device or kernel pointer!\n");
        return(CL_INVALID_VALUE);
    }

    const cl_device_id device = *devicePtr;
    const cl_kernel kernel = *kernelPtr;

    // get kernel name
    char kernelName[OCL_MAX_STRING] = "UNKNOWN";
    oclResult = clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, sizeof(kernelName), &kernelName, NULL);
    if(CL_SUCCESS != oclResult) {
        logMessage(warn, true, "Error during OpenCL kernel name query (error: %d)\n", oclResult);
    }

    // get kernel work group details
    size_t maxWorkGroupSize = 0;
    oclResult = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
    if(CL_SUCCESS != oclResult) {
        logMessage(warn, true, "Error during OpenCL kernel information query (error: %d)\n", oclResult);
        return(CL_INVALID_OPERATION);
    }
    logMessage(debug, true, "Device-specific work group information for OpenCL kernel \"%s\":\n", kernelName);
    logMessage(debug, false, "Maximum work group size: \t\t%u\n", maxWorkGroupSize);

    // sanity check
    if(maxWorkGroupSize < workGroupSize) {
        logMessage(warn, true, "Kernel \"%s\" exceeds device-specific maximum work group size (requested: %u)!\n", kernelName, workGroupSize);
        logMessage(warn, false, "Reducing kernel's work group size to allowed maximum of: %u work items\n", maxWorkGroupSize);
        *workGroupSizeLimit = maxWorkGroupSize;
    }
    else {
        *workGroupSizeLimit = 0;
    }

#if !defined(CL_VERSION_1_0)
    size_t workGroupSizeMultiple = 0;
    oclResult = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(workGroupSizeMultiple), &workGroupSizeMultiple, NULL);
    if(CL_SUCCESS != oclResult) {
        logMessage(warn, true, "Error during OpenCL kernel information query (error: %d)\n", oclResult);
        return(CL_INVALID_OPERATION);
    }
    logMessage(debug, false, "Preferred work group size multiple: \t%u\n", workGroupSizeMultiple);
#endif

    // get memory consumption
    cl_ulong localMemorySize = 0;
    oclResult = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(localMemorySize), &localMemorySize, NULL);
    if(CL_SUCCESS != oclResult) {
        logMessage(warn, true, "Error during OpenCL kernel information query (error: %d)\n", oclResult);
        return(CL_INVALID_OPERATION);
    }
    logMessage(debug, false, "Local memory consumption: \t\t%lu\n", localMemorySize);

#if !defined(CL_VERSION_1_0)
    cl_ulong privateMemorySize = 0;
    oclResult = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(privateMemorySize), &privateMemorySize, NULL);
    if(CL_SUCCESS != oclResult) {
        logMessage(warn, true, "Error during OpenCL kernel information query (error: %d)\n", oclResult);
        return(CL_INVALID_OPERATION);
    }
    logMessage(debug, false, "Private memory consumption: \t\t%lu\n", privateMemorySize);
#endif

    return(CL_SUCCESS);
}

void contextErrorCallback(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
    logMessage(error, true, "Error in OpenCL context: %s\n", errinfo);
}
