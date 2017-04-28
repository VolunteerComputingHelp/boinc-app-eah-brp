/***************************************************************************
 *   Copyright (C) 2011 by Oliver Bock                                     *
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

#ifndef DEMOD_BINARY_OCL_H
#define DEMOD_BINARY_OCL_H

#include <stdint.h>

#ifdef __APPLE__
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

#include "../../structs.h"
#include "../../diptr.h"

#ifdef  __cplusplus
extern "C" {
#endif

extern int initialize_ocl(int oclDeviceIdGiven, int *oclDeviceId, cl_platform_id boincPlatformId, cl_device_id boincDeviceId);

extern int set_up_resampling(DIfloatPtr input, DIfloatPtr *output, const RESAMP_PARAMS *const params, float *sinLUTsamples, float *cosLUTsamples);
extern int run_resampling(DIfloatPtr input, DIfloatPtr output, const RESAMP_PARAMS *const params);
extern int tear_down_resampling(DIfloatPtr output);

extern int set_up_fft(DIfloatPtr input, DIfloatPtr *output, uint32_t nsamples, unsigned int fft_size);
extern int run_fft(DIfloatPtr input, DIfloatPtr output, uint32_t nsamples, unsigned int fft_size, float norm_factor);
extern int tear_down_fft(DIfloatPtr output);

extern int set_up_harmonic_summing(float ** sumspec, int32_t** dirty, unsigned int * nr_pages_ptr, unsigned int fundamental_idx_hi,unsigned int harmonic_idx_hi);
extern int run_harmonic_summing(float ** sumspec, int32_t ** dirty, unsigned int nr_pages, DIfloatPtr  powerspectrum, unsigned int window_2,unsigned int fundamental_idx_hi,unsigned int harmonic_idx_hi, float * thresholds);
extern int tear_down_harmonic_summing(float ** sumspec, int32_t ** dirty);


extern int shutdown_ocl();

#ifdef  __cplusplus
}
#endif

#endif
