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

#ifndef OCL_UTILITIES_H
#define OCL_UTILITIES_H

#ifdef __APPLE__
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

#ifdef _WIN32
    #define STDCALL __stdcall
#else
    #define STDCALL
#endif

#include "../../erp_utilities.h"

#define MEGABYTES 1048576
#define OCL_MAX_DEVICES 64

#ifdef  __cplusplus
extern "C" {
#endif

cl_device_id findBestFreeDevice(const cl_platform_id platform,
                                const cl_int minMajorVersion,
                                const cl_int minMinorVersion,
                                const cl_ulong minGlobalMemoryBytes,
                                const cl_device_type deviceType);

// Note: OpenCL doesn't provide means to check available/used global memory!

cl_int dumpFloatDeviceBufferToTextFile(const cl_command_queue queue,
                                       const cl_mem buffer,
                                       const size_t size,
                                       const char *filename);

cl_int checkKernelWorkGroupInfo(const cl_device_id *device,
                               const cl_kernel *kernel,
                               size_t workGroupSize,
                               size_t *workGroupSizeLimit);

void STDCALL contextErrorCallback(const char *errinfo,
                                  const void *private_info,
                                  size_t cb,
                                  void *user_data);

#ifdef  __cplusplus
}
#endif

#endif
