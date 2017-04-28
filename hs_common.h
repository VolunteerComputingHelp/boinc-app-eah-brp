/***************************************************************************
 *   Copyright (C) 2008 by Benjamin Knispel, Holger Pletsch                *
 *   benjamin.knispel[AT]aei.mpg.de                                        *
 *   Copyright (C) 2009, 2010, 2011 by Oliver Bock                         *
 *   oliver.bock[AT]aei.mpg.de                                             *
 *   Copyright (C) 2009, 2010 by Heinz-Bernd Eggenstein                    *
 *                                                                         *
 *   This file is part of Einstein@Home (Radio Pulsar Edition).            *
 *                                                                         *
 *   Description:                                                          *
 *   harmonic summing core implementation. This is the main function       *
 *   used for CPU variant, but is also called by the CUDA variant for      *
 *   fixing boundary values. Derived from ABP2 code for harmonic summing   *
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

#ifndef __HS_COMMON_H__
#define __HS_COMMON_H__


#include <stdint.h>


#define LOG_PS_PAGE_SIZE 10

extern "C" {

/* magic constants used by the GPU kernels to allow a uniform processing */
static const int h_lut[16] = {4,3,2,2,2,  1, 1, 1, 1 , 1, 1, 1     ,-1,   3,2,1};
static const int k_lut[16] = {0,4,2,6,10, 1, 3, 5, 7 , 9,11,13     ,-1,   0,2,3};

int harmonic_summing(float ** sumspec,
                     int32_t ** dirtyFlags,
                     const float * powerspectrum,
                     unsigned int window_2,
                     unsigned int fundamental_idx_hi,
                     unsigned int harmonic_idx_hi,
                     float * thr);


}
#endif
