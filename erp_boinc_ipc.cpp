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


#ifdef _MSC_VER
extern ostream cout;
extern ostream cerr;
// TODO: add the other iostream stuff!
#else
#include <iostream>
#include <sstream>
#include <iomanip>
using namespace std;
#endif

#include <libxml/tree.h>

#include "erp_boinc_ipc.h"
#include "boinc_api.h"
#include "util.h"

BOINC_OPTIONS erp_boinc_options;
APP_INIT_DATA erp_app_init_data;
t_pulsar_search erp_search_info;

#include "graphics2.h"

char* shmem = NULL;

void erp_update_shmem(void)
{
    BOINC_STATUS boinc_status;

    if (!shmem) return;

    // serialize power spectrum data;
    ostringstream powerSpectrumString;
    powerSpectrumString.exceptions(ios_base::badbit | ios_base::failbit);
    for (int i = 0; i < POWERSPECTRUM_BINS; ++i) {
        // add zero-padded hex string of bin value
        try {
            powerSpectrumString	<< setw(2) << setfill('0') << hex
            << (int)erp_search_info.power_spectrum[i];
        }
        catch(ios_base::failure)
        {
            fprintf(stderr, "Error preparing power spectrum shared memory data!\n");
        }
    }

    // update BOINC status information
    boinc_get_status(&boinc_status);

    // reset shared memory area
    memset(shmem, 0, ERP_SHMEM_SIZE);

    // prepare XML serialization
    xmlThrDefIndentTreeOutput(1);
    xmlDocPtr xmlDoc = NULL;
    xmlNodePtr rootNode = NULL, boincStatusNode = NULL;
    ostringstream converter;
    converter.exceptions(ios_base::badbit | ios_base::failbit);
    converter.precision(3);

    // setup XML document
    xmlDoc = xmlNewDoc(BAD_CAST("1.0"));
    rootNode = xmlNewNode(NULL, BAD_CAST("graphics_info"));
    xmlDocSetRootElement(xmlDoc, rootNode);

    // add child nodes (with content) to root
    try {
        converter << fixed << erp_search_info.skypos_rac;
        xmlNewChild(rootNode, NULL, BAD_CAST("skypos_rac"), BAD_CAST(converter.str().c_str()));
        converter.str("");

        converter << fixed << erp_search_info.skypos_dec;
        xmlNewChild(rootNode, NULL, BAD_CAST("skypos_dec"), BAD_CAST(converter.str().c_str()));
        converter.str("");

        converter << fixed << erp_search_info.dispersion_measure;
        xmlNewChild(rootNode, NULL, BAD_CAST("dispersion"), BAD_CAST(converter.str().c_str()));
        converter.str("");

        converter << fixed << erp_search_info.orbital_radius;
        xmlNewChild(rootNode, NULL, BAD_CAST("orb_radius"), BAD_CAST(converter.str().c_str()));
        converter.str("");

        converter << fixed << erp_search_info.orbital_period;
        xmlNewChild(rootNode, NULL, BAD_CAST("orb_period"), BAD_CAST(converter.str().c_str()));
        converter.str("");

        converter << fixed << erp_search_info.orbital_phase;
        xmlNewChild(rootNode, NULL, BAD_CAST("orb_phase"), BAD_CAST(converter.str().c_str()));
        converter.str("");

        xmlNewChild(rootNode, NULL, BAD_CAST("power_spectrum"), BAD_CAST(powerSpectrumString.str().data()));

        converter << fixed << boinc_get_fraction_done();
        xmlNewChild(rootNode, NULL, BAD_CAST("fraction_done"), BAD_CAST(converter.str().c_str()));
        converter.str("");

        converter << fixed << boinc_worker_thread_cpu_time();
        xmlNewChild(rootNode, NULL, BAD_CAST("cpu_time"), BAD_CAST(converter.str().c_str()));
        converter.str("");

        converter << fixed << dtime();
        xmlNewChild(rootNode, NULL, BAD_CAST("update_time"), BAD_CAST(converter.str().c_str()));
        converter.str("");

        // add boinc status (with child nodes) to root
        // TODO: use the following to control the graphics app (e.g. trigger app_init_data update!)
        boincStatusNode = xmlNewChild(rootNode, NULL, BAD_CAST("boinc_status"), NULL);

        converter << dec << boinc_status.no_heartbeat;
        xmlNewChild(boincStatusNode, NULL, BAD_CAST("no_heartbeat"), BAD_CAST(converter.str().c_str()));
        converter.str("");

        converter << dec << boinc_status.suspended;
        xmlNewChild(boincStatusNode, NULL, BAD_CAST("suspended"), BAD_CAST(converter.str().c_str()));
        converter.str("");

        converter << dec << boinc_status.quit_request;
        xmlNewChild(boincStatusNode, NULL, BAD_CAST("quit_request"), BAD_CAST(converter.str().c_str()));
        converter.str("");

        converter << dec << boinc_status.reread_init_data_file;
        xmlNewChild(boincStatusNode, NULL, BAD_CAST("reread_init_data_file"), BAD_CAST(converter.str().c_str()));
        converter.str("");

        converter << dec << boinc_status.abort_request;
        xmlNewChild(boincStatusNode, NULL, BAD_CAST("abort_request"), BAD_CAST(converter.str().c_str()));
        converter.str("");

        // use default float format
        converter.unsetf(ios_base::floatfield);

        converter << boinc_status.working_set_size;
        xmlNewChild(boincStatusNode, NULL, BAD_CAST("working_set_size"), BAD_CAST(converter.str().c_str()));
        converter.str("");

        converter << boinc_status.max_working_set_size;
        xmlNewChild(boincStatusNode, NULL, BAD_CAST("max_working_set_size"), BAD_CAST(converter.str().c_str()));
    }
    catch(ios_base::failure) {
        fprintf(stderr, "Error converting shared memory data!\n");
    }

    // dump xml document
    xmlChar *buffer;
    int bufferSize = -1;
    xmlDocDumpFormatMemoryEnc(xmlDoc, &buffer, &bufferSize, "UTF-8", 1);

    if(bufferSize > 0 && bufferSize < ERP_SHMEM_SIZE) {
        // copy xml string to shared memory area
        snprintf(shmem, bufferSize, "%s", buffer);
    }
    else {
        fprintf(stderr, "Error writing shared memory data (size limit exceeded)!\n");
    }

    //clean up
    xmlFree(buffer);
    xmlFreeDoc(xmlDoc);
    xmlCleanupParser();
}

int erp_setup_shmem(void)
{
    boinc_get_init_data(erp_app_init_data);

    shmem = (char*) boinc_graphics_make_shmem(ERP_SHMEM_APP_NAME, ERP_SHMEM_SIZE);
    if (!shmem) {
        fprintf(stderr, "Failed to create shared memory area!\n");
        return (-1);
    }
    erp_update_shmem();
    return (0);
}

void erp_set_boinc_options(void)
{
    boinc_get_init_data(erp_app_init_data);
    boinc_options_defaults(erp_boinc_options);
#ifdef USE_CUDA
    erp_boinc_options.normal_thread_priority = 1;
#endif
}

void erp_boinc_init(void) {
    boinc_init_options(&erp_boinc_options);
}
