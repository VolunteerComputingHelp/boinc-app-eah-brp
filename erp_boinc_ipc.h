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

#ifndef ERP_BOINC_IPC_H
#define ERP_BOINC_IPC_H

#include "boinc_api.h"

#define ERP_BOINC_IPC_H_RCSID "$Id:$"

#define ERP_SHMEM_APP_NAME "EinsteinRadio"
#define ERP_SHMEM_SIZE 1024
#define POWERSPECTRUM_BINS 40

#ifdef __cplusplus
extern "C" {
#endif

    typedef struct {
        double skypos_rac;
        double skypos_dec;
        double dispersion_measure;
        double orbital_radius;
        double orbital_period;
        double orbital_phase;
        unsigned char power_spectrum[POWERSPECTRUM_BINS];
    } t_pulsar_search;

    extern t_pulsar_search erp_search_info;

    extern int erp_setup_shmem(void);

    extern void erp_update_shmem(void);

    extern void erp_set_boinc_options(void);

    extern void erp_boinc_init(void);

    extern void erp_fraction_done(double);
#ifdef __cplusplus
}
#endif

#endif
