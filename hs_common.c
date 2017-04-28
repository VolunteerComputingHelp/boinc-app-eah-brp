/***************************************************************************
 *   Copyright (C) 2008 by Benjamin Knispel, Holger Pletsch                *
 *   benjamin.knispel[AT]aei.mpg.de                                        *
 *   Copyright (C) 2009, 2010 by Oliver Bock                               *
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

#include "hs_common.h"

extern "C" {

int harmonic_summing(float ** sumspec,
                     int32_t ** dirty,
                     const float * powerspectrum,
                     unsigned int window_2,
                     unsigned int fundamental_idx_hi,
                     unsigned int harmonic_idx_hi,
                     float * thr) {	

      // loop over frequency bins and maximize the summed power at each bin for each number of summed harmonics
      // Semantics of "dirty" array:
      // if element at the ith index of powerspectrum of 2^k th  harmonic is exceeding the threshold thr[k], 
      // mark the surrounding "page" of (2^LOG_PS_PAGE_SIZE) elements as "dirty" by setting
      // dirty[k][i >> LOG_PS_PAGE_SIZE] 
      // The rationale is that only blocks of (2^LOG_PS_PAGE_SIZE) elements indicated by correctponding entries in dirty[][]
      // need to be inspected in the signal toplist candidate selection phase which is following after
      // the harmonic summing    

      // but first ... some precomputations
      // compute idx16[l] =  i *l + 8   , start with i = window_2
      // note that floor(idx16[l] / 16 )  = floor(i* l/16 +0.5) , without any floating point math


      unsigned int i,l;
      unsigned int idx16[16];
      float sum1;
      int j1=-1;
      float sum2;
      int j2=-1;
      float sum3;
      int j3=-1;
      float sum4;
      int j4=-1;

      for(l=0;l < 16 ; l++) {
	idx16[l] = window_2 * l + 8;
      }


      for(i = window_2; i < harmonic_idx_hi; i++)
	{
	  float sum;
          float power;
	  int j;

	  // first harmonic
	  sum = powerspectrum[i];
          if((sum > thr[0]) && (i < fundamental_idx_hi)) {dirty[0][i >> LOG_PS_PAGE_SIZE]=1; };
 


	  // up to second harmonic at j
	  j = idx16[8] >> 4;

	  sum += powerspectrum[j];
	  
	  // if we visit a new summspec cell, reset the cached sum
	  if(j != j1) {sum1 = 0.0;} 
	  if(j < fundamental_idx_hi) {
	    power=(sum > sum1) ? sum : sum1;
	    if (power > thr[1]) {
	      sumspec[1][j]=power;
	      dirty[1][j >> LOG_PS_PAGE_SIZE]=1;
	    }else {
	      // make sure the sumspec value is initialized even if no threshold 
	      // is passed
	      if(j != j1) {sumspec[1][j]=power;}       
	    }
 	  }  	
 	  j1=j;
	  sum1=power;
	  

	  // up to fourth harmonic at j
	  j = idx16[4] >> 4;
	  sum += powerspectrum[idx16[12]>>4] + powerspectrum[j];

	  if(j != j2) {sum2 = 0.0;} 
	  if(j < fundamental_idx_hi) {
	    power=(sum > sum2) ? sum : sum2;
	    if (power > thr[2]) {
	      sumspec[2][j]=power;
	      dirty[2][j >> LOG_PS_PAGE_SIZE]=1;
	    } else {
	      // make sure the sumspec value is initialized even if no threshold 
	      // is passed
	      if(j != j2) {sumspec[2][j]=power;}       
	    }
 	  }  	
 	  j2=j;
	  sum2=power;
	  // up to eighth harmonic at j
	  j = idx16[2] >>4;
	  sum += powerspectrum[idx16[14] >>4] + powerspectrum[idx16[10] >>4] + \
	    powerspectrum[idx16[6] >>4] + powerspectrum[j];

	  if(j != j3) {sum3 = 0.0;} 
	  if(j < fundamental_idx_hi) {
	    power=(sum > sum3) ? sum : sum3;
	    if (power > thr[3]) {
	      sumspec[3][j]=power;
	      dirty[3][j >> LOG_PS_PAGE_SIZE]=1;
	    } else {
	      // make sure the sumspec value is initialized even if no threshold 
	      // is passed
	      if(j != j3) {sumspec[3][j]=power;}       
	    }
 	  }  	
 	  j3=j;
	  sum3=power;

	  // up to sixteenth harmonic at j
	  j = idx16[1] >>4;
	  sum += powerspectrum[idx16[15] >>4] + powerspectrum[idx16[13] >>4] + \
	    powerspectrum[idx16[11] >>4] + powerspectrum[idx16[9] >>4] + \
	    powerspectrum[idx16[7] >>4] + powerspectrum[idx16[5] >>4] + \
	    powerspectrum[idx16[3] >>4] + powerspectrum[j];

	  if(j != j4) {sum4 = 0.0;} 
	  if(j < fundamental_idx_hi) {
	    power=(sum > sum4) ? sum : sum4;
	    if (power > thr[4]) {
	      sumspec[4][j]=power;
	      dirty[4][j >> LOG_PS_PAGE_SIZE]=1;
	    }else {
	      // make sure the sumspec value is initialized even if no threshold 
	      // is passed
	      if(j != j4) {sumspec[4][j]=power;}       
	    }
 	  }  	
 	  j4=j;
	  sum4=power;	  // update the indices
	  
	  for(l=0;l < 16 ; l++) {
		idx16[l]+=l;
      	  }

	}// end of loop over frequency bins
    return 0;
}
} /* extern "C" */




