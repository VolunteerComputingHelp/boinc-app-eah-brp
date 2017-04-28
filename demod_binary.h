/***************************************************************************
 *   Copyright (C) 2008 by Oliver Bock                                     *
 *   oliver.bock[AT]aei.mpg.de                                             *
 *                                                                         *
 *   This file is part of Einstein@Home (Radio Pulsar Edition).            *
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

#ifndef DEMOD_BINARY_H
#define DEMOD_BINARY_H

// global error codes
#define RADPUL_EMEM 1
#define RADPUL_EFILE 2
#define RADPUL_EIO 3
#define RADPUL_EVAL 4
#define RADPUL_EMISC 5

#define RADPUL_CUDA_DEVICE_FIND 1001
#define RADPUL_CUDA_DEVICE_SET 1002
#define RADPUL_CUDA_DEVICE_PROP 1003
#define RADPUL_CUDA_DEVICE_VERSION 1004
#define RADPUL_CUDA_MEM_ALLOC_HOST 1005
#define RADPUL_CUDA_MEM_ALLOC_DEVICE 1006
#define RADPUL_CUDA_MEM_COPY_HOST_DEVICE 1007
#define RADPUL_CUDA_MEM_COPY_DEVICE_HOST 1008
#define RADPUL_CUDA_MEM_FREE_HOST 1009
#define RADPUL_CUDA_MEM_FREE_DEVICE 1010
#define RADPUL_CUDA_FFT_PLAN 1011
#define RADPUL_CUDA_FFT_EXEC 1012
#define RADPUL_CUDA_FFT_DESTROY 1013
#define RADPUL_CUDA_EMULATION_MODE 1014
#define RADPUL_CUDA_KERNEL_INVOKE 1015
#define RADPUL_CUDA_LOAD_MODULE 1016
#define RADPUL_CUDA_LOOKUP_KERNEL 1017
#define RADPUL_CUDA_LOOKUP_SYMBOL 1018
#define RADPUL_CUDA_KERNEL_PREPARE 1019
#define RADPUL_CUDA_DRIVER_INIT 1020

#define RADPUL_OCL_PLATFORM_FIND 2001
#define RADPUL_OCL_PLATFORM_UNAVAILABLE 2002
#define RADPUL_OCL_PLATFORM_DETAILS 2003
#define RADPUL_OCL_DEVICE_FIND 2004
#define RADPUL_OCL_COMPILER_UNAVAILABLE 2005
#define RADPUL_OCL_DEVICE_UNAVAILABLE 2006
#define RADPUL_OCL_CONTEXT_CREATE 2007
#define RADPUL_OCL_CMDQUEUE_CREATE 2008
#define RADPUL_OCL_PROGRAM_CREATE 2009
#define RADPUL_OCL_PROGRAM_BUILD 2010
#define RADPUL_OCL_KERNEL_CREATE 2011
#define RADPUL_OCL_MEM_ALLOC_HOST 2012
#define RADPUL_OCL_MEM_ALLOC_DEVICE 2013
#define RADPUL_OCL_MEM_COPY_HOST_DEVICE 2014
#define RADPUL_OCL_MEM_COPY_DEVICE_HOST 2015
#define RADPUL_OCL_MEM_FREE_HOST 2016
#define RADPUL_OCL_MEM_FREE_DEVICE 2017
#define RADPUL_OCL_KERNEL_SETUP 2018
#define RADPUL_OCL_KERNEL_INVOKE 2019
#define RADPUL_OCL_FFT_PLAN 2020
#define RADPUL_OCL_FFT_EXEC 2021
#define RADPUL_OCL_FFT_DESTROY 2022

#ifdef __cplusplus
extern "C" {
#endif

extern int MAIN (int argc, char *argv[]);

#ifdef __cplusplus
}
#endif

#endif
