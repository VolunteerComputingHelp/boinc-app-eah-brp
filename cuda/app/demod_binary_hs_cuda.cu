/***************************************************************************
 *   Copyright (C) 2008 by Benjamin Knispel, Holger Pletsch                *
 *   benjamin.knispel[AT]aei.mpg.de                                        *
 *   Copyright (C) 2009,2010 by Oliver Bock                                *
 *   oliver.bock[AT]aei.mpg.de                                             *
 *   Copyright (C) 2009,2010 by Heinz-Bernd Eggenstein                     *
 *                                                                         *
 *   This file is part of Einstein@Home (Radio Pulsar Edition).            *
 *                                                                         *
 *   Description:                                                          *
 *   Performs harmonic summing (2nd ... 16th harmonic) of powerspectrum    *
 *   CUDA variant.                                                         *
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

#include <cuda.h>

#include "demod_binary_hs_cuda.cuh"
#include "cuda_utilities.h"
#include "../../erp_utilities.h"
#include "../../hs_common.h"
#include "../../demod_binary.h"


#define HS_BLOCKSIZE 256    // must be an integer power of 2 (because of the following constraint)
#define HS_LOG_BLOCKSIZE 8  // constraint: HS_LOG_BLOCKSIZE = lrint(log2(HS_BLOCKSIZE))

#include "harmonic_summing_kernel.cuh"


float * powerspectrumHost = 0; 
int     stdmem_powerspectrum = 0;  // Flag to distinguish page-locked allocation



// module global variables

CUdeviceptr h_lutDev; // look up tables in global device memory
CUdeviceptr k_lutDev;
CUdeviceptr thrADev;  // threshold for 1st , 2nd, 4th, 8th, 16th harmonics on device
CUdeviceptr sumspecDev[5] ;  /* an array of device memory pointers (NOT an array on the device!) */

CUmodule cuModuleHS;   // device module and kernel handles
CUfunction kernelHarmonicSumming;
CUfunction kernelHarmonicSummingGaps;
CUtexref h_lutTexRef;
CUtexref k_lutTexRef;
CUtexref thrATexRef;
CUtexref powerspectrumTexRef;


int set_up_harmonic_summing(float ** sumspec,int32_t ** dirty, unsigned int * nr_pages_ptr, unsigned int fundamental_idx_hi,unsigned int harmonic_idx_hi)
{
    CUresult cuResult = CUDA_SUCCESS;
    int i;
    unsigned int nr_pages;

    // load device modules / kernels
    char modulePath[1024] = {0};
    i = resolveFilename("dbhs.dev", modulePath, 1023);
    if(i) {
	logMessage(error, true, "Couldn't retrieve HS CUDA device module path (error: %i)!\n", i);
	return(RADPUL_CUDA_LOAD_MODULE);
    }
    cuResult = cuModuleLoad(&cuModuleHS, modulePath);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Couldn't load HS CUDA device module (error: %i)!\n", cuResult);
        return(RADPUL_CUDA_LOAD_MODULE);
    }

    cuResult = cuModuleGetFunction(&kernelHarmonicSumming, cuModuleHS, "harmonic_summing_kernel");
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Couldn't get CUDA HS kernel handle (error: %i)!\n", cuResult);
        return(RADPUL_CUDA_LOOKUP_KERNEL);
    }

    cuResult = cuModuleGetFunction(&kernelHarmonicSummingGaps, cuModuleHS, "harmonic_summing_kernel_gaps");
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Couldn't get CUDA HS kernel handle (error: %i)!\n", cuResult);
        return(RADPUL_CUDA_LOOKUP_KERNEL);
    }

    // allocate memory for the harmonic summed spectra
    // in CUDA version, this includes the 1st harmonics
    // TODO : sumspec [0] has to be treated differently once the powerspectrum is left on device
    //        initially and only copied async. later
    for( i = 1; i < 5; i++)
    {
      sumspec[i] = (float *) calloc(fundamental_idx_hi, sizeof(float));
      if(sumspec[i] == NULL)
      {
        logMessage(error, true, "Couldn't allocate %d bytes of memory for sumspec at bottom level.\n", fundamental_idx_hi*sizeof(float ));
        return(RADPUL_EMEM);
      }
    }

    sumspecDev[0]=0;
    for( i =1; i  < 5 ; i++) {
        cuResult = cuMemAlloc(&(sumspecDev[i]), sizeof(float) * fundamental_idx_hi);
        if(cuResult != CUDA_SUCCESS) {
            logMessage(error, true, "Couldn't allocate %d bytes of CUDA HS summing memory (error: %i)!\n", sizeof(float) * fundamental_idx_hi, cuResult);
            return(RADPUL_CUDA_MEM_ALLOC_DEVICE);
        }
    }

    nr_pages=(fundamental_idx_hi >> LOG_PS_PAGE_SIZE)+1;
    *nr_pages_ptr = nr_pages;
    for(i = 0; i < 5 ; i++) {
      dirty[i] = (int32_t *) calloc(nr_pages, sizeof(int32_t));
      if(dirty[i] == NULL)
      {
        logMessage(error, true, "Couldn't allocate %d bytes of memory for sumspec page flags at bottom level.\n", fundamental_idx_hi*sizeof(float ));
        return(RADPUL_EMEM);
      }

    }


    // allocate texture memory
    cuResult = cuMemAlloc(&h_lutDev, sizeof(int32_t) * 16);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Couldn't allocate %d bytes of CUDA HSH texture memory (error: %i)!\n", sizeof(int32_t) * 16, cuResult);
        return(RADPUL_CUDA_MEM_ALLOC_DEVICE);
    }
    cuResult = cuMemAlloc(&k_lutDev, sizeof(int32_t) * 16);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Couldn't allocate %d bytes of CUDA HSK texture memory (error: %i)!\n", sizeof(int32_t) * 16, cuResult);
        return(RADPUL_CUDA_MEM_ALLOC_DEVICE);
    }
    cuResult = cuMemAlloc(&thrADev, sizeof(float) * 5);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Couldn't allocate %d bytes of CUDA HST texture memory (error: %i)!\n", sizeof(float) * 4, cuResult);
        return(RADPUL_CUDA_MEM_ALLOC_DEVICE);
    }

    // copy LUTs to texture memory
    cuResult = cuMemcpyHtoD(h_lutDev, h_lut, sizeof(int32_t) * 16);
    if(cuResult != CUDA_SUCCESS)
    {
        logMessage(error, true, "Error during CUDA host->device HSH lookup table data transfer (error: %i)\n", cuResult);
        return(RADPUL_CUDA_MEM_COPY_HOST_DEVICE);
    }
    cuResult = cuMemcpyHtoD(k_lutDev, k_lut, sizeof(int32_t) * 16);
    if(cuResult != CUDA_SUCCESS)
    {
        logMessage(error, true, "Error during CUDA host->device HSK lookup table data transfer (error: %i)\n", cuResult);
        return(RADPUL_CUDA_MEM_COPY_HOST_DEVICE);
    }

    // bind textures
    cuResult = cuModuleGetTexRef(&h_lutTexRef, cuModuleHS, "h_lutTex");
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Couldn't get CUDA HSH texture handle (error: %i)!\n", cuResult);
        return(RADPUL_CUDA_MEM_COPY_HOST_DEVICE);
    }
    cuResult = cuTexRefSetAddress(NULL, h_lutTexRef, h_lutDev, sizeof(int32_t) * 16);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Couldn't bind CUDA HSH texture (error: %i)!\n", cuResult);
        return(RADPUL_CUDA_MEM_COPY_HOST_DEVICE);
    }

    cuResult = cuModuleGetTexRef(&k_lutTexRef, cuModuleHS, "k_lutTex");
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Couldn't get CUDA HSK texture handle (error: %i)!\n", cuResult);
        return(RADPUL_CUDA_MEM_COPY_HOST_DEVICE);
    }
    cuResult = cuTexRefSetAddress(NULL, k_lutTexRef, k_lutDev, sizeof(int32_t) * 16);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Couldn't bind CUDA HSK texture (error: %i)!\n", cuResult);
        return(RADPUL_CUDA_MEM_COPY_HOST_DEVICE);
    }

    cuResult = cuModuleGetTexRef(&thrATexRef, cuModuleHS, "thrATex");
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Couldn't get CUDA HST texture handle (error: %i)!\n", cuResult);
        return(RADPUL_CUDA_MEM_COPY_HOST_DEVICE);
    }
    cuResult = cuTexRefSetAddress(NULL, thrATexRef, thrADev, sizeof(float) * 5);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Couldn't bind CUDA HST texture (error: %i)!\n", cuResult);
        return(RADPUL_CUDA_MEM_COPY_HOST_DEVICE);
    }

    cuResult = cuModuleGetTexRef(&powerspectrumTexRef, cuModuleHS, "powerspectrumTex");
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Couldn't get CUDA HSP texture handle (error: %i)!\n", cuResult);
        return(RADPUL_CUDA_MEM_COPY_HOST_DEVICE);
    }


    // allocate host memory for power spectrum
    cuResult = cuMemAllocHost((void**) &powerspectrumHost , harmonic_idx_hi * sizeof(float));
    if(cuResult != CUDA_SUCCESS)
    {
        logMessage(warn, true, "Couldn't allocate %d bytes of pinned host memory for power spectrum (error: %i)! Trying fallback...\n", harmonic_idx_hi * sizeof(float), cuResult);
        powerspectrumHost = (float *) calloc(harmonic_idx_hi, sizeof(float));
        if(powerspectrumHost == NULL)
        {
            logMessage(error, true, "Couldn't allocate %d bytes of memory for power spectrum!\n", harmonic_idx_hi * sizeof(float));
            return(RADPUL_CUDA_MEM_ALLOC_HOST);
        }
        // set flag to indicate conventional memory allocation
        stdmem_powerspectrum = 1;
    }
    logMessage(debug, true, "Allocated host memory for power spectrum: %i bytes\n", sizeof(float) * harmonic_idx_hi);


    return 0;
}



int tear_down_harmonic_summing(float ** sumspec, int32_t ** dirty)
{
    CUresult cuResult = CUDA_SUCCESS;
    int i;

    // clean up. (0th element is powerspectrum, freed separately)
    for(i = 1; i < 5; i++) {
      free(sumspec[i]);
    }

    for(i = 0; i < 5; i++) {
      free(dirty[i]);
    }

    for( i =1; i  < 5 ; i++) {
        cuResult = cuMemFree(sumspecDev[i]);
        if(cuResult != CUDA_SUCCESS) {
            logMessage(error, true, "Error freeing CUDA HS device memory (error: %d)\n", cuResult);
            return(RADPUL_CUDA_MEM_FREE_DEVICE);
        }
    }

    // unbind textures (is this the only/recommended way?!)
    cuResult = cuTexRefSetAddress(NULL, h_lutTexRef, NULL, 0);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Couldn't unbind CUDA HSH texture (error: %i)!\n", cuResult);
        return(RADPUL_CUDA_MEM_FREE_DEVICE);
    }
    cuResult = cuTexRefSetAddress(NULL, k_lutTexRef, NULL, 0);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Couldn't unbind CUDA HSK texture (error: %i)!\n", cuResult);
        return(RADPUL_CUDA_MEM_FREE_DEVICE);
    }
    cuResult = cuTexRefSetAddress(NULL, thrATexRef, NULL, 0);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Couldn't unbind CUDA HST texture (error: %i)!\n", cuResult);
        return(RADPUL_CUDA_MEM_FREE_DEVICE);
    }

    // free texture memory
    cuResult = cuMemFree(h_lutDev);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error freeing CUDA HSH texture memory (error: %d)\n", cuResult);
        return(RADPUL_CUDA_MEM_FREE_DEVICE);
    }
    cuResult = cuMemFree(k_lutDev);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error freeing CUDA HSK texture memory (error: %d)\n", cuResult);
        return(RADPUL_CUDA_MEM_FREE_DEVICE);
    }
    cuResult = cuMemFree(thrADev);
    if(cuResult != CUDA_SUCCESS) {
        logMessage(error, true, "Error freeing CUDA HST texture memory (error: %d)\n", cuResult);
        return(RADPUL_CUDA_MEM_FREE_DEVICE);
    }


    if(stdmem_powerspectrum) {
        free(powerspectrumHost);
    }
    else {
        cuResult = cuMemFreeHost(powerspectrumHost);
        if(cuResult != CUDA_SUCCESS) {
            logMessage(error, true, "Error deallocating CUDA pinned host powerspectrum memory (error: %i)\n", cuResult);
            return(RADPUL_CUDA_MEM_FREE_HOST);
        }
    }


    return 0;
}



int run_harmonic_summing(float ** sumspec, int32_t ** dirty, unsigned int nr_pages, DIfloatPtr  powerspectrum_dip , unsigned int 
window_2,unsigned int fundamental_idx_hi,unsigned int harmonic_idx_hi, float *thresholds)
{
      unsigned int l1,l2,i,j,k;

      CUresult cuResult = CUDA_SUCCESS;
      dim3 dg1,dg2;
      dim3 db1,db2;
      int nr_pages_total = nr_pages * 5;
   
      float * powerspectrum = powerspectrumHost; // global variable

      /* borders for main kernel computation in a 16 index grid */
      /* for simplicity we always start at the left border of the spectrum,
        the kernel itself will take care of the window_2 offset */
      l1= 0;
      /* the number of main kernel blocks of width 16 that is needed to fully cover
         the spectrum up to index harmonic_idx_hi -1 (inclusive) */
      l2= ((harmonic_idx_hi -1 + 8) >> 4) +1 ;

      CUdeviceptr powerspectrumDev=powerspectrum_dip.device_ptr;;
      CUdeviceptr dirtyDev;
      CUstream stream[2];
      int32_t * dirtyTmp; 
      cuStreamCreate(&stream[0], 0);
      cuStreamCreate(&stream[1], 0);


      // add powerspectrum as first spectra element

      // TODO : sumspec [0] has to be treated differently once the powerspectrum is left on device
      //        initially and only copied (possibly async.) later

      sumspec[0] = powerspectrum;


      /* allocate sumspec arrays on device */
      sumspecDev[0] = powerspectrumDev;  
      for( i =1; i  < 5 ; i++) {
          cuResult = cuMemsetD8(sumspecDev[i], 0, sizeof(float) * fundamental_idx_hi);
          if(cuResult != CUDA_SUCCESS) {
              logMessage(error, true, "Couldn't erase %d bytes of CUDA HS summing memory (error: %i)!\n", sizeof(float) * fundamental_idx_hi, cuResult);
              return(RADPUL_CUDA_MEM_COPY_HOST_DEVICE);
          }
      }

      cuResult = cuMemAlloc(&dirtyDev, sizeof(int32_t) *  nr_pages_total);
      if(cuResult != CUDA_SUCCESS) {
          logMessage(error, true, "Couldn't allocate %d bytes of CUDA HS summing memory (error: %i)!\n", sizeof(int32_t) * 
nr_pages_total,cuResult);
              return(RADPUL_CUDA_MEM_ALLOC_DEVICE);
      }
      cuResult = cuMemsetD8(dirtyDev, 0, sizeof(int32_t) * nr_pages_total);
      if(cuResult != CUDA_SUCCESS) {
              logMessage(error, true, "Couldn't erase %d bytes of CUDA HS summing memory (error: %i)!\n", sizeof(int32_t) * nr_pages_total, cuResult);
              return(RADPUL_CUDA_MEM_COPY_HOST_DEVICE);
      }
 


      /* copy thresholds to device */


      cuResult = cuMemcpyHtoD(thrADev, thresholds, sizeof(float) * 5);
      if(cuResult != CUDA_SUCCESS)
      {
          logMessage(error, true, "Error during CUDA host->device HS thresholds data transfer (error: %i)\n", cuResult);
          return(RADPUL_CUDA_MEM_COPY_HOST_DEVICE);
      }

      /* the powerspectrum is a not so obvious candidate for texture memory. Each kernel
         reads powerspectrum[i*k /16 +0.5] for k = 1..15, so many cells are read more than
         once, so caching should be beneficial. Cached access should also benefit from data locality */
      cuResult = cuTexRefSetAddress(NULL, powerspectrumTexRef, powerspectrumDev, sizeof(float) * harmonic_idx_hi);
      if(cuResult != CUDA_SUCCESS) {
          logMessage(error, true, "Couldn't bind CUDA HSP texture (error: %i)!\n", cuResult);
          return(RADPUL_CUDA_MEM_COPY_HOST_DEVICE);
      }

/* Execute kernel to perform harmonic summing (with some gaps where sumspec target values would overlap */

/* somehow this seems to work better than y=16 , x = (l2-l1)/16 */
/* anyway we have to use a 2 dim because index in each dim is limited */
      dg1.y=(l2-l1)/HS_BLOCKSIZE ;

      /* add one block if not perfectly aligned */
      if((l2-l1) % HS_BLOCKSIZE !=0 ) {
        dg1.y++;
      }
      dg1.x=16; // rather arbitrarily, there is no algorithmic reason for the value 16
      dg1.z=1;

      db1.x=HS_BLOCKSIZE;
      db1.y=1;
      db1.z=1;

      logMessage(debug, true, "Executing harmonic summing CUDA kernel (%u threads each in %u blocks)...\n", db1.x, dg1.x * dg1.y * dg1.z);

      /* the lowest index i for which sumspec[h][i/(1<<h)+0.5] is computed by the kernels*/

      /* first kernel in first stream*/
      /* TODO: check effect on performance of having concurrent streams */

      // prepare parameters
      int kernelParamOffset = 0;
      for(i = 1; i < 5; ++i) {
          KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(sumspecDev[i]));
          cuResult = cuParamSetv(kernelHarmonicSumming, kernelParamOffset, &(sumspecDev[i]), sizeof(sumspecDev[i]));
          kernelParamOffset += sizeof(sumspecDev[i]);
          if(cuResult != CUDA_SUCCESS) {
              logMessage(error, true, "Error during CUDA HS kernel parameter setup (error: %d)\n", cuResult);
              return(RADPUL_CUDA_KERNEL_PREPARE);
          }
      }

      KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(dirtyDev));
      cuResult = cuParamSetv(kernelHarmonicSumming, kernelParamOffset, &dirtyDev, sizeof(dirtyDev));
      kernelParamOffset += sizeof(dirtyDev);
      if(cuResult != CUDA_SUCCESS) {
          logMessage(error, true, "Error during CUDA HS kernel parameter setup (error: %d)\n", cuResult);
          return(RADPUL_CUDA_KERNEL_PREPARE);
      }

      KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(powerspectrumDev));
      cuResult = cuParamSetv(kernelHarmonicSumming, kernelParamOffset, &powerspectrumDev, sizeof(powerspectrumDev));
      kernelParamOffset += sizeof(powerspectrumDev);
      if(cuResult != CUDA_SUCCESS) {
          logMessage(error, true, "Error during CUDA HS kernel parameter setup (error: %d)\n", cuResult);
          return(RADPUL_CUDA_KERNEL_PREPARE);
      }

      KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(window_2));
      cuResult = cuParamSeti(kernelHarmonicSumming, kernelParamOffset, window_2);
      kernelParamOffset += sizeof(window_2);
      if(cuResult != CUDA_SUCCESS) {
          logMessage(error, true, "Error during CUDA HS kernel parameter setup (error: %d)\n", cuResult);
          return(RADPUL_CUDA_KERNEL_PREPARE);
      }

      KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(fundamental_idx_hi));
      cuResult = cuParamSeti(kernelHarmonicSumming, kernelParamOffset, fundamental_idx_hi);
      kernelParamOffset += sizeof(fundamental_idx_hi);
      if(cuResult != CUDA_SUCCESS) {
          logMessage(error, true, "Error during CUDA HS kernel parameter setup (error: %d)\n", cuResult);
          return(RADPUL_CUDA_KERNEL_PREPARE);
      }

      KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(harmonic_idx_hi));
      cuResult = cuParamSeti(kernelHarmonicSumming, kernelParamOffset, harmonic_idx_hi);
      kernelParamOffset += sizeof(harmonic_idx_hi);
      if(cuResult != CUDA_SUCCESS) {
          logMessage(error, true, "Error during CUDA HS kernel parameter setup (error: %d)\n", cuResult);
          return(RADPUL_CUDA_KERNEL_PREPARE);
      }

      cuResult = cuParamSetSize(kernelHarmonicSumming, kernelParamOffset);
      if(cuResult != CUDA_SUCCESS) {
          logMessage(error, true, "Error finalizing CUDA HS kernel parameter setup (error: %d)\n", cuResult);
          return(RADPUL_CUDA_KERNEL_PREPARE);
      }

      // prepare block size
      cuResult = cuFuncSetBlockShape(kernelHarmonicSumming, db1.x, db1.y, db1.z);
      if(cuResult != CUDA_SUCCESS) {
          logMessage(error, true, "Error during CUDA HS kernel block setup (error: %d)\n", cuResult);
          return(RADPUL_CUDA_KERNEL_PREPARE);
      }

      // launch kernel grid
      cuResult = cuLaunchGridAsync(kernelHarmonicSumming, dg1.x, dg1.y, stream[0]);
      if(cuResult != CUDA_SUCCESS) {
          logMessage(error, true, "Error launching CUDA HS kernel (error: %d)\n", cuResult);
          return(RADPUL_CUDA_KERNEL_INVOKE);
      }

/* execute second kernel, this time to fill the gaps */

      l1=0;
      /* the number of gap kernel blocks (where each block covers 2 segments of length 8 indices)
      l2= (((harmonic_idx_hi -1 + 12) >> 4 ) +1) /*>> 1*/ ; /* TODO CHECK!!!!!*/

      dg2.y=(l2-l1)/HS_BLOCKSIZE;
      /* add one if not perfectly aligned to HS_BLOCKSIZE */
      if((l2-l1) % HS_BLOCKSIZE != 0) {
        dg2.y++;
      }
      dg2.x=16;  /* again, 16 is rather arbitrary */
      dg2.z=1;

      db2.x=HS_BLOCKSIZE/2;  /* sic! it is essential that the blocksize is half that of the first kernel */
      db2.y=1;
      db2.z=1;

      logMessage(debug, true, "Executing harmonic summing gaps CUDA kernel (%u threads each in %u blocks)...\n", db2.x, dg2.x * dg2.y * dg2.z);

      // prepare parameters
      kernelParamOffset = 0;
      for(i = 1; i < 5; ++i) {
          KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(sumspecDev[i]));
          cuResult = cuParamSetv(kernelHarmonicSummingGaps, kernelParamOffset, &sumspecDev[i], sizeof(sumspecDev[i]));
          kernelParamOffset += sizeof(sumspecDev[i]);
          if(cuResult != CUDA_SUCCESS) {
              logMessage(error, true, "Error during CUDA HSG kernel parameter setup (error: %d)\n", cuResult);
              return(RADPUL_CUDA_KERNEL_PREPARE);
          }
      }
      
      KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(dirtyDev));
      cuResult = cuParamSetv(kernelHarmonicSummingGaps, kernelParamOffset, &dirtyDev, sizeof(dirtyDev));
      kernelParamOffset += sizeof(dirtyDev);
      if(cuResult != CUDA_SUCCESS) {
          logMessage(error, true, "Error during CUDA HS kernel parameter setup (error: %d)\n", cuResult);
          return(RADPUL_CUDA_KERNEL_PREPARE);
      }
      
      KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(powerspectrumDev));
      cuResult = cuParamSetv(kernelHarmonicSummingGaps, kernelParamOffset, &powerspectrumDev, sizeof(powerspectrumDev));
      kernelParamOffset += sizeof(powerspectrumDev);
      if(cuResult != CUDA_SUCCESS) {
          logMessage(error, true, "Error during CUDA HSG kernel parameter setup (error: %d)\n", cuResult);
          return(RADPUL_CUDA_KERNEL_PREPARE);
      }

      KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(window_2));
      cuResult = cuParamSeti(kernelHarmonicSummingGaps, kernelParamOffset, window_2);
      kernelParamOffset += sizeof(window_2);
      if(cuResult != CUDA_SUCCESS) {
          logMessage(error, true, "Error during CUDA HSG kernel parameter setup (error: %d)\n", cuResult);
          return(RADPUL_CUDA_KERNEL_PREPARE);
      }

      KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(fundamental_idx_hi));
      cuResult = cuParamSeti(kernelHarmonicSummingGaps, kernelParamOffset, fundamental_idx_hi);
      kernelParamOffset += sizeof(fundamental_idx_hi);
      if(cuResult != CUDA_SUCCESS) {
          logMessage(error, true, "Error during CUDA HSG kernel parameter setup (error: %d)\n", cuResult);
          return(RADPUL_CUDA_KERNEL_PREPARE);
      }

      KERNEL_PARAM_ALIGN_UP(kernelParamOffset, __alignof(harmonic_idx_hi));
      cuResult = cuParamSeti(kernelHarmonicSummingGaps, kernelParamOffset, harmonic_idx_hi);
      kernelParamOffset += sizeof(harmonic_idx_hi);
      if(cuResult != CUDA_SUCCESS) {
          logMessage(error, true, "Error during CUDA HSG kernel parameter setup (error: %d)\n", cuResult);
          return(RADPUL_CUDA_KERNEL_PREPARE);
      }

      cuResult = cuParamSetSize(kernelHarmonicSummingGaps, kernelParamOffset);
      if(cuResult != CUDA_SUCCESS) {
          logMessage(error, true, "Error finalizing CUDA HSG kernel parameter setup (error: %d)\n", cuResult);
          return(RADPUL_CUDA_KERNEL_PREPARE);
      }

      // prepare block size
      cuResult = cuFuncSetBlockShape(kernelHarmonicSummingGaps, db2.x, db2.y, db2.z);
      if(cuResult != CUDA_SUCCESS) {
          logMessage(error, true, "Error during CUDA HSG kernel block setup (error: %d)\n", cuResult);
          return(RADPUL_CUDA_KERNEL_PREPARE);
      }

      // launch kernel grid
      cuResult = cuLaunchGridAsync(kernelHarmonicSummingGaps, dg2.x, dg2.y, stream[1]);
      if(cuResult != CUDA_SUCCESS) {
          logMessage(error, true, "Error launching CUDA HSG kernel (error: %d)\n", cuResult);
          return(RADPUL_CUDA_KERNEL_INVOKE);
      }

      // wait until all device processing finished (errors should indicate earlier launch failures)
      cuResult = cuCtxSynchronize();
      if(cuResult != CUDA_SUCCESS) {
          logMessage(error, true, "Error during CUDA HS/HSG kernel launch and/or device synchronization (error: %d)\n", cuResult);
          return(RADPUL_CUDA_KERNEL_INVOKE);
      }

      // destroy both streams
      cuStreamDestroy(stream[0]);
      cuStreamDestroy(stream[1]);
      // unbind texture
      cuResult = cuTexRefSetAddress(NULL, powerspectrumTexRef, NULL, 0);
      if(cuResult != CUDA_SUCCESS) {
          logMessage(error, true, "Couldn't unbind CUDA HSP texture (error: %i)!\n", cuResult);
          return(RADPUL_CUDA_MEM_FREE_DEVICE);
      }


      // copy back dirty page flags to memory

      dirtyTmp = (int32_t*) malloc (nr_pages_total * sizeof(int32_t));
      if(dirtyTmp == NULL) { 
          logMessage(error, true, "Couldn't allocate %d bytes of memory for temp mem (HS).\n", nr_pages_total * sizeof(int32_t) );
          return(RADPUL_EMEM);
      };
      cuResult = cuMemcpyDtoH(dirtyTmp,dirtyDev,sizeof(int32_t) * nr_pages_total);
      if(cuResult != CUDA_SUCCESS) {  
          logMessage(error, true, "Error during CUDA device->host HS data transfer (dirty) (error: %d)\n", cuResult);
          return(RADPUL_CUDA_MEM_COPY_DEVICE_HOST);
      }


      int dirty_idx_min[5] = {0,0,0,0,0};
      int dirty_idx_max[5] = {0,0,0,0,0};
      int d,d_min,d_max;
      

      k=0;
      for(i=0 ; i < 5 ; i++) {
        d_min=nr_pages;
        d_max=-1;

        // find the first dirty page
	for(j=0; j < nr_pages ; j++) {
	   d=dirty[i][j] = dirtyTmp[k++];
	   if(d!=0) {
		d_min = j ; 
               d_max = j;
               j++; 
		break;
	   }
        }
        // go thru the rest and record the last dirty page we find

        for(    ; j < nr_pages ; j++) {
           d=dirty[i][j] = dirtyTmp[k++];
           if(d!=0) {
                d_max = j ;                          
           }
        }

        dirty_idx_min[i]=d_min;
        dirty_idx_max[i]=d_max;

      } 		

      free(dirtyTmp);
      cuResult = cuMemFree(dirtyDev);
      if(cuResult != CUDA_SUCCESS) {
          logMessage(error, true, "Error freeing CUDA HS device memory (error: %d)\n", cuResult);
          return(RADPUL_CUDA_MEM_FREE_DEVICE);
      }



      /* copy back the results from the CUDA kernel.
       * make sure to copy only those cells from sumspec
       * (including the "1st harmonics" powerspectrum itself)
       * that have a chance to include a candidate that makes it to the toplist
       *
       * TODO : look at possibility to copy only a subarray
       */

      for( i =0; i  < 5 ; i++) {
          /* no need to copy anything if there is no potential candidate at all */
          if (dirty_idx_max[i]!=-1) {
                  size_t seg_offset =  dirty_idx_min[i] << LOG_PS_PAGE_SIZE;
                  size_t seg_length = (dirty_idx_max[i]-dirty_idx_min[i] +1)  << LOG_PS_PAGE_SIZE;		
                  // clip the segment to be copied at the max length of the array
                  size_t seg_length_limit = fundamental_idx_hi - seg_offset;
                  if (seg_length > seg_length_limit) {
                      seg_length = seg_length_limit;
                  }

		  // do some pointer arithmetic to get the the right subsegment of memory to copy
		  cuResult = cuMemcpyDtoH(sumspec[i]+seg_offset , (CUdeviceptr) (((float*) sumspecDev[i])+seg_offset)   , sizeof(float) * seg_length);
		  if(cuResult != CUDA_SUCCESS) {
		      logMessage(error, true, "Error during CUDA device->host HS data transfer (error: %d)\n", cuResult);
		      return(RADPUL_CUDA_MEM_COPY_DEVICE_HOST);
                  }
          }
      }





  return 0;
}
