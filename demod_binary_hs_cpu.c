/***************************************************************************
 *   Copyright (C) 2010 by Oliver Bock                                     *
 *   oliver.bock[AT]aei.mpg.de                                             *
 *   Copyright (C) 2010 by Heinz-Bernd Eggenstein                          *
 *                                                                         *
 *   This file is part of Einstein@Home (Radio Pulsar Edition).            *
 *                                                                         *
 *   Description:                                                          *
 *   Performs harmonic summing (2nd ... 16th harmonic) of powerspectrum    *
 *   CPU variant.                                                          *
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

#include <stdlib.h>
#include <string.h>
#include "demod_binary_hs_cpu.h"
#include "erp_utilities.h"
#include "hs_common.h"
#include "demod_binary.h"


int set_up_harmonic_summing(float **sumspec,int32_t ** dirty,unsigned int * nr_pages_ptr,unsigned int fundamental_idx_hi, unsigned int harmonic_idx_hi)
{
    int i;
    unsigned int nr_pages = (fundamental_idx_hi >> LOG_PS_PAGE_SIZE ) +1;
    *nr_pages_ptr = nr_pages;

    // allocate memory for the harmonic summed spectra and dirty pages array
    for( i = 1; i < 5; i++) {
        sumspec[i] = (float *) calloc(fundamental_idx_hi, sizeof(float));
        if(sumspec[i] == NULL) {
            logMessage(error, true, "Couldn't allocate %d bytes of memory for sumspec at bottom level.\n", fundamental_idx_hi * sizeof(float));
            return(RADPUL_EMEM);
        }
    }

    for( i = 0; i < 5; i++) {        
        dirty[i] = (int32_t *) calloc(nr_pages, sizeof(int32_t));
        if(dirty[i] == NULL) {
            logMessage(error, true, "Couldn't allocate %d bytes of memory for sumspec page flag at bottom level.\n", fundamental_idx_hi * sizeof(int32_t));
            return(RADPUL_EMEM);
        }
    }


    return 0;
}


int run_harmonic_summing(float **sumspec, int32_t ** dirty_flags, unsigned int nr_pages,
                         DIfloatPtr powerspectrum_dip, unsigned int window_2, unsigned int fundamental_idx_hi,
                         unsigned int harmonic_idx_hi, float *thresholds)
{
    int result,i;
    float * powerspectrum =  powerspectrum_dip.host_ptr; 
    // zero out sumspec array
    // TODO: necessary?

    for( i = 1; i < 5; i++) {
        memset(sumspec[i], 0, fundamental_idx_hi * sizeof(float));
    }

    // zero out dirty pages flags array

    for( i = 0; i < 5; i++) {
        memset(dirty_flags[i], 0, nr_pages * sizeof(float));
    }

    // add powerspectrum as first spectra element
    sumspec[0] = powerspectrum;

    result = harmonic_summing(sumspec, dirty_flags,powerspectrum, window_2, fundamental_idx_hi, harmonic_idx_hi,thresholds);
    return result;
}

int tear_down_harmonic_summing(float **sumspec, int32_t ** dirty)
{
    int i;

    // clean up (0th element is powerspectrum, freed separately)
    for(i = 1; i < 5; i++) {
        free(sumspec[i]);
    }

    for(i = 0; i < 5; i++) {
        free(dirty[i]);
    }

    return 0;
}


