#ifndef __HARMONIC_SUMMING_KERNEL_CUH__
#define __HARMONIC_SUMMING_KERNEL_CUH__
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
 *   CUDA kernel for harmonic summing. The code is closely following the   *
 *   ABP2 harmonic summing implementation for CPUs.                         *
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

#include "../../hs_common.h"

#ifdef  __cplusplus
extern "C" {
#endif

texture <float,1,cudaReadModeElementType> powerspectrumTex;   /* Texture reference for PowerSpectrum */
texture <float,1,cudaReadModeElementType> thrATex;	      /* threshold texture reference         */
texture <int,1,cudaReadModeElementType> h_lutTex;             /* index lookup texture reference      */
texture <int,1,cudaReadModeElementType> k_lutTex;             /* yet another index lookup texture reference      */

#ifndef NO_TEXTURES
#define FETCH(t, i) tex1Dfetch(t##Tex, i)                     /* macro to perform texture lookup or array lookup */
#else                                                         /* from CUDA SDK examples */
#define FETCH(t, i) (t[i])                                    /* macro to perform texture lookup or array lookup */
#endif

/* macro to toggle whether only those sumspec values above a threshold value (one threhiold per harmonic array) should be  */
/* updated in global memory */
/* even if set, you can get the old functionality (e.g. for screensaver display code compatibility) by setting threshold to 0.0 */





/* main kernel for harmonic summing.
 * Constraint: Blocksize must be a multiple of 16 (each sub-block of 16 consecutive threads will be independent)
 *
 * Kernel consists of two logical steps:
 *
 * Step 1 : each thread looks at the values powerspectrum[i*k/16+0.5], for some thread specific i and for all k=1..15
 *         and computes sums corresponding to the harmonics. These harmonic sum spectrum candidates are written to
 *         shared memory. These are candidates only because only the maximum value of all candidates
 *         for any given sumspec slot will be needed, and several threads will compute candidates for the same sumspec slot.
 *
 * Step 2: after all threads have finished step 1:
 *         Each sub-block of 16 consecutive threads now has created sums in shared memory that need to be filtered
 *         for maximum values to compute 12 final entries into sumspec[1][*] thru sumspec[4][*] slots
 *         (done one per thread, so 4 threads idle in this step). sumspec[0]is powerspectrum.
 *         This leaves 3 sumspec slots unfilled "on each side" of a sub-block because data dependencies
 *         from more than one sub-block (overlap). Those missing slots are taken care of by a second kernel
 *
 *         In step 2, each of the N*12 x threads will
 *            -read a certain number of entries from shared memory
 *            -find the maximum entry of those values
 *            -store the maximum value in a target sumspec[n][m] slot (if above threshold).
 *
 *         Each thread selects its starting index in shared memory, the number of entries to read (and max), and the
 *         target indices for sumspec by looking up values from two small integer LUTs, cached via texture memory.
 */

  __global__ void harmonic_summing_kernel (float * sumspec1, float * sumspec2,float * sumspec3, float * sumspec4,int * dirty,
                                           const float * powerspectrum,unsigned int window_2,
                                           unsigned int fundamental_idx_hi,unsigned int harmonic_idx_hi
) {

  /* shared memory */
  __shared__ float sspec_cand [4*HS_BLOCKSIZE];       /* temp. space to hold sumspec candidates */
  __shared__ float * sumspec[4];                      /* pointers to result arrays */


/* initializing an array (here: lookup table h_lut and k_lut) as automatic variables would place it in local memory
 * which is NOT cached => bad
 * instead, use a texture for look up tables
 * for reference, given here:
 *     const int  h_lut[16] = {4,3,2,2,2,  1, 1, 1, 1 , 1, 1, 1     ,-1,   3,2,1  } ;
 *     const int  k_lut[16] = {0,4,2,6,10, 1, 3, 5, 7 , 9,11,13     ,-1,   0,2,3  } ;
 *
 */

  int idx_j= (blockIdx.y << 4)+blockIdx.x ;
 // int idx_j_offset =  (idx_j<< HS_LOG_BLOCKSIZE) + (l1 << 4);
  int idx_j_offset =  (idx_j<< HS_LOG_BLOCKSIZE) + -16 ; // negative index to handle let border

  // Thread index calculation
  // for step one, one thread is doing roughly the same as the sequential ABP2 CPU version for index i */

  int i= idx_j_offset + threadIdx.x +8; // start index of block must always be congruent to 8 modulo 16
                                        // can still be negative for left border of spectrum, or can exceed
                                        // harmonic_idx_hi-1
  int k;
  int h =i;
  int j,jj,len,offset,lend2,lenM1;
  float sum;
  float p;
  int i2,i4,i8; /* multiples of i. looks silly but integer multiplications are expensive (!) */
                /* OTOH, using more registers then necessary is not good either ... */
  int iN;

 if(i < window_2 || i >= harmonic_idx_hi ) {
    // no candidate contribution from this index
    sspec_cand[threadIdx.x] = 0.0f;
    sspec_cand[HS_BLOCKSIZE + threadIdx.x] = 0.0f;
    sspec_cand[2*HS_BLOCKSIZE + threadIdx.x] = 0.0f;
    sspec_cand[3*HS_BLOCKSIZE + threadIdx.x] = 0.0f;
 } else {

  p=FETCH(powerspectrum,i);
  i2=i+i;
  i4 = i << 2;
  i8 = i4+i4;
  iN= i8+8;
  if( (p > FETCH(thrA,0)) && (i < fundamental_idx_hi)) { 
      dirty[ (i>> LOG_PS_PAGE_SIZE)] = 1;
  }

  p+=FETCH(powerspectrum,iN>>4);  //(8*i+8) / 16
  sspec_cand[threadIdx.x] = p;

  iN=i4+8;
  sum =  FETCH(powerspectrum,iN>>4);   //(4*i+8) / 16
  iN+=i8;
  sum+=  FETCH(powerspectrum,iN>>4);   //(12*i+8) / 16
  p+=sum;

  sspec_cand[HS_BLOCKSIZE + threadIdx.x] = p;

  iN=i2+8;
  sum= FETCH(powerspectrum,iN>>4);       //(2*i+8) / 16
  iN+=i4;
  sum += FETCH(powerspectrum,iN>>4);     //(6*i+8) / 16
  iN+=i4;
  sum += FETCH(powerspectrum,iN>>4);     //(10*i+8) / 16
  iN+=i4;
  sum += FETCH(powerspectrum,iN>>4);     //(14*i+8) / 16
  p+= sum;

  sspec_cand[2*HS_BLOCKSIZE + threadIdx.x] = p;

  iN=i+8;
  sum= FETCH(powerspectrum,iN>>4);       //(1*i+8) / 16
  iN+=i2;
  sum += FETCH(powerspectrum,iN>>4);     //(3*i+8) / 16
  iN+=i2;
  sum += FETCH(powerspectrum,iN>>4);     //(5*i+8) / 16
  iN+=i2;
  sum += FETCH(powerspectrum,iN>>4);     //(7*i+8) / 16
  iN+=i2;
  sum += FETCH(powerspectrum,iN>>4);     //(9*i+8) / 16
  iN+=i2;
  sum += FETCH(powerspectrum,iN>>4);     //(11*i+8) / 16
  iN+=i2;
  sum += FETCH(powerspectrum,iN>>4);     //(13*i+8) / 16
  iN+=i2;
  sum += FETCH(powerspectrum,iN>>4);     //(15*i+8) / 16

  p+= sum;
  sspec_cand[3*HS_BLOCKSIZE + threadIdx.x] = p;
}
/* It seems to be faster to let all threads do the same as opposed to have 1 thread do this and
   let the others idle. Still better is to have this in texture memory maybe*/

  sumspec[0] = sumspec1;
  sumspec[1] = sumspec2;
  sumspec[2] = sumspec3;
  sumspec[3] = sumspec4;

  /* finished Step 1 */
  __syncthreads();

  /* selecting 12 threads out of every 16-thread sub block for step 2, one for every sumspec[][] slot to fill
   * with max candidate for that slot.
   */

  if( (threadIdx.x & 15) < 12 ) {
    /* lookup two values that are sufficient to parameterize step 2 so that the same code
       can be used for all threads without many ifs and whiles */

    h= FETCH(h_lut,threadIdx.x&15);
    k= FETCH(k_lut,threadIdx.x&15) + ((threadIdx.x>>4) << 4);

    len = 1 << h; /* how many sumspec candidates we have to inspect for maximum value */
    lend2=len>>1;
    offset= ((h-1) << HS_LOG_BLOCKSIZE) + k; /* base index for shared memory access: start index of candidates */
    lenM1=len-1;

    /* This loop is usually from 0 .. len-1 only, len<=16  */
    /* but conditionals are no good in a thread block (warp splits), */
    /* so we loop from 0 ..15 and potentially re-visit some */
    /* elements over and over. It doesn't matter   */
    /* when computing the max */

    sum = sspec_cand[offset];
#pragma unroll 15
    for (j =1 ; j < 16; j++) {
      sum = fmaxf(sum,sspec_cand[offset+(j & lenM1)]);
    }

    /* compute target index for the sumspec slot. This corresponds to variable j in ABP2 hs code. */
    /* on the left border of the spectrum, ( idx_j_offset + k +8 + lend2 ) can be negative!) */


    jj=( idx_j_offset + k +8 + lend2 );
    j= (jj>= 0) ? (jj  >> h) : -1 ;

    /* if we do threshold checks and the thresholds are chosen e.g. using information from the toplist
       of previous template runs, finding new, better sumspec values will be rare and the whole kernel
       essentially becomes a read-only task, with only sporadic exceptions. */

    if (
       ( sum > FETCH(thrA,h) ) && j >=0 &&
       (j < fundamental_idx_hi) ) {
      sumspec[h-1][j]=sum;

      /* mark this page of the sumspec array as dirty */
      /* TODO: maybe write to shared memory buffer first, then (by thread 0) to global memory if set to 1, avoiding 
               simultaneous writes to global memory from the same block */

      dirty[((fundamental_idx_hi >> LOG_PS_PAGE_SIZE) +1)  * h + (j>> LOG_PS_PAGE_SIZE)] = 1;  	
    }
  }

}

/* secondary kernel for harmonic summing.
 * Constraint: Blocksize must be a multiple of 16 (Each sub-block of 16 consecutive threads will be independent)
 *
 * Essentially this kernel does the same as the main kernel, except that it uses another "tiling" to fill
 * the gaps in sumspec arrays left unfilled by the main kernel. Logically the processing can be
 * interpreted as sub-blocks of 8 threads each (compared to 16 threads in main kernel)
 * Note that the CUDA blocksize for this kernel is half the blocksize of the main kernel.
 *
 * Kernel consists of two logical steps:
 *
 * Step 1 : each thread looks at the values powerspectrum[i*k/16+0.5], for some thread specific i and for all k=1..15
 *         and computes sums corresponding to the harmonics. These harmonic sum spectrum candidates are written to
 *         shared memory. These are candidates only because only the maximum value of the candidates
 *         for any given sumspec slot will be needed, and several threads will compute candidates for the same sumspec slot.
 *
 * Step 2: after all threads have finished step 1:
 *         Each sub-block of 2*8 consecutive threads now has created sums in shared memory that need to be filtered
 *         for maximum values to compute 2*3 final entries into sumspec[1][*] thru sumspec[4][*] slots
 *         (done one per thread).
 *
 *         In step 2, each of the 2*N*3 threads will
 *            -read a certain number of entries from shared memory
 *            -find the maximum entry of those values
 *            -store the maximum value in a sumspec[n][m] slot.
 *
 *         Each thread selects its starting index in shared memory, the number of entries to read (and max), and the
 *         target indices for sumspec by looking up values from one tiny integer LUT, cached via texture memory.
 *
 */


__global__ void harmonic_summing_kernel_gaps (float * sumspec1, float * sumspec2,float * sumspec3, float * sumspec4, int * dirty, const 
float * powerspectrum,unsigned int window_2,unsigned int fundamental_idx_hi,unsigned int harmonic_idx_hi) {
  // block index

  __shared__ float sspec_cand [2*HS_BLOCKSIZE];
  __shared__ float * sumspec[4]; /*= {sumspec1,sumspec2,sumspec3,sumspec4} */


  /* FIXME step 1 is almost identical to that in the main kernel... I guess one should
     finally put this into a little inlined device function */

  int idx_j= (blockIdx.y << 4)+blockIdx.x;
  int idx_j_offset= (idx_j << HS_LOG_BLOCKSIZE);

  // Thread index
  int idx_i_offset = threadIdx.x;
  // (threadIdx.x & 7) + ((threadIdx.x >> 3 ) << 3);

  int i = idx_j_offset +4 + (threadIdx.x & 7) + ((threadIdx.x >> 3 ) << 4);

  int k;
  int h =i;
  int j,len,offset,lend2,lenM1;
  float sum;
  float p;


  int i2,i4,i8; /* multiples of i. looks silly but integer multiplications are expensive (!) */
                               /* OTOH, using more registers then necessary is not good either ... */
  int iN;

  /* for this kernel , there can be now overlap with the left border of the spectrum (index 0),
     but we could be lower than window_2 or higher than harmonic_idx_hi */

 if(i < window_2 || i >= harmonic_idx_hi ) {
   // no candidate contribution from this index
   sspec_cand[idx_i_offset] = 0.0f;
   sspec_cand[HS_BLOCKSIZE/2 + idx_i_offset] = 0.0f;
   sspec_cand[2*(HS_BLOCKSIZE/2) + idx_i_offset] = 0.0f;
   sspec_cand[3*(HS_BLOCKSIZE/2) + idx_i_offset] = 0.0f;
 } else {

  p=FETCH(powerspectrum,i);
  i2=i+i;
  i4 = i << 2;
  i8 = i4+i4;
  iN= i8+8;
  if((p > FETCH(thrA,0)) && (i < fundamental_idx_hi)) { 
      dirty[ (i>> LOG_PS_PAGE_SIZE)] = 1;
  }
  p+=FETCH(powerspectrum,iN>>4);
  sspec_cand[idx_i_offset] = p;

  iN=i4+8;
  sum =  FETCH(powerspectrum,iN>>4);
  iN+=i8;
  sum+=  FETCH(powerspectrum,iN>>4);
  p+=sum;

  sspec_cand[(HS_BLOCKSIZE/2) + idx_i_offset] = p;

  iN=i2+8;
  sum= FETCH(powerspectrum,iN>>4);
  iN+=i4;
  sum += FETCH(powerspectrum,iN>>4);
  iN+=i4;
  sum += FETCH(powerspectrum,iN>>4);
  iN+=i4;
  sum += FETCH(powerspectrum,iN>>4);
  p+= sum;

  sspec_cand[2*(HS_BLOCKSIZE/2) + idx_i_offset] = p;

  iN=i+8;
  sum= FETCH(powerspectrum,iN>>4);
  iN+=i2;
  sum += FETCH(powerspectrum,iN>>4);
  iN+=i2;
  sum += FETCH(powerspectrum,iN>>4);
  iN+=i2;
  sum += FETCH(powerspectrum,iN>>4);
  iN+=i2;
  sum += FETCH(powerspectrum,iN>>4);
  iN+=i2;
  sum += FETCH(powerspectrum,iN>>4);
  iN+=i2;
  sum += FETCH(powerspectrum,iN>>4);
  iN+=i2;
  sum += FETCH(powerspectrum,iN>>4);

  p+= sum;
  sspec_cand[3*(HS_BLOCKSIZE/2) + idx_i_offset] = p;
 }
  sumspec[0] = sumspec1;
  sumspec[1] = sumspec2;
  sumspec[2] = sumspec3;
  sumspec[3] = sumspec4;


  __syncthreads();
  /*step 2*/


  /* in step 2, there will be 2 sub-blocks of 3 active threads for each 16 (2*8) sub-block in step 1  */
  if((threadIdx.x < 4*(HS_BLOCKSIZE/16))  && ((threadIdx.x & 3) !=3 ) ) {
    /* for this kernel, the lookup values for the first lookup table h_lut can directly be computed from index,
       so no need for a lookup here in this special case
     */

//    h= FETCH(h_lut,(threadIdx.x&3) +13);

    h= 3- (threadIdx.x&3);
    k= FETCH(k_lut,(threadIdx.x&3)+13) + ((threadIdx.x >>2)<<3) ;
    len = 1 << h;
    lend2=len>>1;
    offset= ((h-1) << (HS_LOG_BLOCKSIZE-1)) + k;
    lenM1= len-1;

    sum=sspec_cand[offset];

    /* for this kernel, the maximum nr of sumspec candidate values to max is 8 */

#pragma unroll 7
    for (j =1 ; j < 8; j++) {
      sum = fmaxf(sum,sspec_cand[offset+(j & lenM1)]);
    }

    j=(( idx_j_offset + k + ((threadIdx.x >>2)<<3) + 4 + lend2 )  >> h) ;


    if (
       ( sum > FETCH(thrA,h) ) &&
        (j < fundamental_idx_hi) ) {
      sumspec[h-1][j]=sum;
      /* mark this page of the sumspec array as dirty */
      /* TODO: maybe write to shared memory buffer first, then (by thread 0) to global memory if set to 1, avoiding 
               simultaneous writes to global memory from the same block */

      dirty[((fundamental_idx_hi >> LOG_PS_PAGE_SIZE) +1)  * h + (j>> LOG_PS_PAGE_SIZE)] = 1;  
    }
  }
}



#ifdef  __cplusplus
}
#endif

#endif
