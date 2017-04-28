/***************************************************************************
 *   Copyright (C) 2012 by Heinz-Bernd Eggenstein                          *
 *   heinz-bernd.eggenstein[AT]aei.mpg.de                                  *
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

#ifndef DIPTR_H
#define DIPTR_H



#if defined USE_CUDA
#include "cuda/app/deviceptr.h"
#elif defined USE_OPENCL
#include "opencl/app/deviceptr.h"
#else 
  typedef void *  device_ptr_t; // dummy , not needed in this case
#endif

// TODO: One day, when we refactor the code in an Object Oriented way, this should become a "smart" like pointer
// TODO: that supports device memory and host memory across platforms and has utility member functions
// TODO: for memory management and debugging support. For now, just use a simple union 

typedef union ptr_union_float {
	float        * host_ptr;
	device_ptr_t   device_ptr;
} DIfloatPtr;




// TODO : define some generic functions, e.g. for dumping to file or reading to file, allocating, deallocating ...




#endif

