
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

#include "demod_binary_fft_fftw.h"

#include <stdlib.h>
#include <gsl/gsl_math.h>
#include "demod_binary.h"
#include "erp_utilities.h"

#ifdef EMBEDDED_WISDOM_HEADER
#define STREXP(S) #S
#define STREXPAND(S) STREXP(S)
#include STREXPAND(EMBEDDED_WISDOM_HEADER)
#endif


// TODO: do we wanna keep those global (or use proper C++, or pass them around)?
fftwf_complex *t_series_resamp_fft = NULL; 
fftwf_plan fft_plan;


int set_up_fft(DIfloatPtr input, DIfloatPtr *output, uint32_t nsamples, unsigned int fft_size)
{
    // allocate memory for FFT (use of fftwf_malloc recommended by FFTW, see manual: section 2.1, page 3)

#ifdef  BRP_FFT_INPLACE
    t_series_resamp_fft = (fftwf_complex*) ((void *) input.host_ptr);
#else
    t_series_resamp_fft = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * fft_size);
    if(t_series_resamp_fft == NULL)
    {
        logMessage(error, true, "Couldn't allocate %d bytes of memory for resampled time series FFT.\n", fft_size * sizeof(fftwf_complex));
        return(RADPUL_EMEM);
    }
#endif

    // create fft plan
    // if configured , load pre-canned wisom from string
    /// else load system wide wisdom if present

#ifdef EMBEDDED_WISDOM_HEADER
    fftwf_import_wisdom_from_string(EMBEDDED_WISDOM);
#else
    fftwf_import_system_wisdom();
#endif
    fft_plan = fftwf_plan_dft_r2c_1d(nsamples, input.host_ptr, t_series_resamp_fft, FFTW_ESTIMATE);

#ifdef  BRP_FFT_INPLACE 
   output->host_ptr =  input.host_ptr;
#else
    // allocate memory for the periodogramm
    output->host_ptr = (float *) calloc(fft_size, sizeof(float));

    if(output->host_ptr == NULL)
    {
        logMessage(error, true, "Couldn't allocate %d bytes of memory for power spectrum.\n", fft_size * sizeof(float));
        return(RADPUL_EMEM);
    }
#endif
    return 0;
}


int run_fft(DIfloatPtr input, DIfloatPtr output, uint32_t nsamples, unsigned int fft_size, float norm_factor)
{
    // unused, as the pointers for input & output of FFT are already encapsulated in the plan
    input.host_ptr = NULL;
    nsamples = 0;

    // execute FFT
    fftwf_execute(fft_plan);

    // compute powerspectrum

#ifdef BRP_FFT_INPLACE
    // avoid aliasing problems by explicitly using the output data pointer
    for(unsigned int i = 1; i < fft_size; i++) {
        output.host_ptr[i] = norm_factor * (gsl_pow_2(output.host_ptr[i+i]) + gsl_pow_2(output.host_ptr[i+i+1]));
    }

#else
    for(unsigned int i = 1; i < fft_size; i++) {
        output.host_ptr[i] = norm_factor * (gsl_pow_2(t_series_resamp_fft[i][0]) + gsl_pow_2(t_series_resamp_fft[i][1]));
    }
#endif
    // set DC power to 0
    output.host_ptr[0]=0.0f;
    return 0;
}


int tear_down_fft(DIfloatPtr output)
{
#ifndef BRP_FFT_INPLACE
    free(output.host_ptr);
    fftwf_free(t_series_resamp_fft);
#endif;
    fftwf_destroy_plan(fft_plan);

    return 0;
}
