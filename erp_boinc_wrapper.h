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

#ifndef ERP_BOINC_WRAPPER_H
#define ERP_BOINC_WRAPPER_H

#include <sys/types.h>
#include <stdio.h>
#include <zlib.h>

/* BOINC includes */
#include "filesys.h"

#define ERP_BOINC_WRAPPER_H_RCSID "$Id:$"

/* Reroute syscalls tp BOINC */
#define fopen boinc_fopen
#define gzopen boinc_gzopen

#ifdef  __cplusplus
extern "C" {
#endif

    extern int MAIN(int,char**);
    extern gzFile boinc_gzopen(const char* path, const char* mode);

#ifdef  __cplusplus
}
#endif

#endif
