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

#ifndef DEMOD_BINARY_RESAMP_CPU_H
#define DEMOD_BINARY_RESAMP_CPU_H

#include <stdint.h>
#include "structs.h"
#include "diptr.h" 

#ifdef __cplusplus
extern "C" {
#endif

extern int set_up_resampling(DIfloatPtr input, DIfloatPtr *output, const RESAMP_PARAMS *const params, float *sinLUTsamples, float *cosLUTsamples);
extern int run_resampling(DIfloatPtr input, DIfloatPtr output, const RESAMP_PARAMS *const params);
extern int tear_down_resampling(DIfloatPtr output);

#ifdef __cplusplus
}
#endif

#endif
