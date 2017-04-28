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

#ifndef CUDA_UTILITIES_H
#define CUDA_UTILITIES_H

#include <cuda.h>
#include "../../erp_utilities.h"

#define KERNEL_PARAM_ALIGN_UP(offset, alignment) (offset) = ((offset) + (alignment) - 1) & ~((alignment) - 1)

#ifdef  __cplusplus
extern "C" {
#endif

extern int findBestFreeDevice(const int minMajorRevision,
                              const int minMinorRevision,
                              const unsigned int minGlobalMemoryBytes,
                              const int needExclusive);

extern void printDeviceGlobalMemStatus(const ERP_LOGLEVEL logLevel, const bool followUp);

extern void printHostRoundingMode(const ERP_LOGLEVEL logLevel, const bool showLevel);

extern int dumpFloatDeviceBufferToTextFile(const CUdeviceptr buffer,
                                           const size_t size,
                                           const char *filename);

#ifdef BOINCIFIED
extern int boinc_get_cuda_device_id(int argc, const char** argv, int * deviceId);
#endif

extern int running_standalone(void);

#ifdef  __cplusplus
}
#endif

#endif
