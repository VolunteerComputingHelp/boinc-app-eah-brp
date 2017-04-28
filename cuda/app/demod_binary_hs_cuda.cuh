/***************************************************************************
 *   Copyright (C) 2010 by Oliver Bock                                     *
 *   oliver.bock[AT]aei.mpg.de                                             *
 *   Copyright (C) 2010 by Heinz-Bernd Eggenstein                          *
 *                                                                         *
 *   This file is part of Einstein@Home (Radio Pulsar Edition).            *
 *                                                                         *
 *   Description:                                                          *
 *   Performs harmonic summing (2nd  ...16th harmonic) of powerspectrum    *
 *   CUDA variant.                                                          *
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

#ifndef DEMOD_BINARY_HS_CUDA_CUH
#define DEMOD_BINARY_HS_CUDA_CUH

#include <stdlib.h>

#include "../../diptr.h"

#ifdef __cplusplus
extern "C" {
#endif

extern int set_up_harmonic_summing(float ** sumspec, int32_t** dirty, unsigned int * nr_pages_ptr, unsigned int fundamental_idx_hi,unsigned int harmonic_idx_hi);
extern int run_harmonic_summing(float ** sumspec, int32_t ** dirty, unsigned int nr_pages, DIfloatPtr  powerspectrum, unsigned int window_2,unsigned int fundamental_idx_hi,unsigned int harmonic_idx_hi, float * thresholds);
extern int tear_down_harmonic_summing(float ** sumspec, int32_t ** dirty);

#ifdef __cplusplus
}
#endif

#endif
