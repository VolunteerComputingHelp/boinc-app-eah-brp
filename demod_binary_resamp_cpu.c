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

#include "demod_binary_resamp_cpu.h"

#include <stdlib.h>
#include <gsl/gsl_math.h>
#include <fftw3.h>
#include "demod_binary.h"
#include "erp_utilities.h"


// TODO: do we wanna keep those global (or use proper C++, or pass them around)?
float *del_t = NULL;


int set_up_resampling(DIfloatPtr input_dip, DIfloatPtr *output_dip, const RESAMP_PARAMS *const params, float *sinLUTsamples, 
float *cosLUTsamples)
{
    float * input = input_dip.host_ptr;
    float ** output = & (output_dip->host_ptr);

    // unused
    input = NULL;
    sinLUTsamples = NULL;
    cosLUTsamples = NULL;

    // allocate memory for time offsets in modulated time
#ifndef BRP_FFT_INPLACE
    del_t = (float *) calloc(params->nsamples_unpadded, sizeof(float));
    if(del_t == NULL)
    {
        logMessage(error, true, "Couldn't allocate %d bytes of memory for modulated time steps.\n", params->nsamples_unpadded * sizeof(float));
        return(RADPUL_EMEM);
    }
#endif

    // allocate memory for resampled time series
#ifdef BRP_FFT_INPLACE
    *output = (float *) fftwf_alloc_real(params->fft_size*2);
#else
    *output = (float *) fftwf_alloc_real(params->nsamples);
#endif
    if(*output == NULL)
    {
        logMessage(error, true, "Couldn't allocate %d bytes of memory for resampled time series.\n", params->nsamples * sizeof(float));
        return(RADPUL_EMEM);
    }

#ifdef BRP_FFT_INPLACE
	del_t= *output;
#endif

    return 0;
}


int run_resampling(DIfloatPtr input_dip, DIfloatPtr output_dip, const RESAMP_PARAMS *const params)
{
    float * input  = input_dip.host_ptr;
    float * output = output_dip.host_ptr;
    
    unsigned int i;
    float i_f;

    unsigned int n_steps;           // number of timesteps to take when resampling
    float mean = 0.0f;              // mean of the time series

    for(i = 0, i_f = 0.0f; i < params->nsamples_unpadded; i++, i_f += 1.0f)
    {
        float t = i_f * params->dt;
        float sinValue = 0.0f;
        float dummy = 0.0f;

        // lookup sin(Omega * t + Psi0)
        sincosLUTLookup(params->Omega * t + params->Psi0, &sinValue, &dummy);

        // compute time offsets as multiples of tsampm subtract zero time offset
        del_t[i] = params->tau * sinValue * params->step_inv - params->S0;
    }

    // number of timesteps that fit into the duration = at most the amount we had before
    n_steps = params->nsamples_unpadded - 1;

    // nearest_idx (see loop below) must not exceed n_unpadded - 1, so go back as far as needed to ensure that
    while(n_steps - del_t[n_steps] >=  params->nsamples_unpadded - 1)
      n_steps--;

    // loop over time at the pulsar (index i, i_f) and find the bin in detector time at which
    // a signal sent at i at the pulsar would arrive at the detector
    for(i = 0, i_f = 0.0f; i < n_steps; i++, i_f += 1.0f)
      {
	// sample i arrives at the detector at i_f - del_t[i], choose nearest neighbour
	int nearest_idx = (int)(i_f - del_t[i] + 0.5);

	// set i-th bin in resampled time series (at the pulsar) to nearest_idx bin from de-dispersed time series
	output[i] = input[nearest_idx];
	mean += output[i];
      }

    logMessage(debug, true,"Time series sum: %f\n",mean);

    mean /= i_f;

    logMessage(debug, true, "Actual time series mean is: %e (length: %i)\n", mean, n_steps);

    // fill up with mean if necessary
    for( ; i < params->nsamples; i++)
    {
        output[i] = mean;
    }

    return 0;
}


int tear_down_resampling(DIfloatPtr output)
{
#ifndef BRP_FFT_INPLACE
    free(del_t);
#endif
    fftwf_free(output.host_ptr);

    return 0;
}
