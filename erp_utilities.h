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

#ifndef ERP_UTILITIES_H
#define ERP_UTILITIES_H

#define ERP_LITTLE_ENDIAN   0
#define ERP_BIG_ENDIAN      1

#define ERP_SINCOS_LUT_RES          64
#define ERP_SINCOS_LUT_SIZE         (ERP_SINCOS_LUT_RES + 1)
#define ERP_SINCOS_LUT_RES_F        (1.0f * ERP_SINCOS_LUT_RES)
#define ERP_SINCOS_LUT_RES_F_INV    (1.0f / ERP_SINCOS_LUT_RES)
#define ERP_TWO_PI                  6.283185f
#define ERP_TWO_PI_INV              (1.0f/ERP_TWO_PI)

#include <stdarg.h>
#include <stdint.h>
#include <sys/types.h>

#ifdef  __cplusplus
extern "C" {
#endif

    typedef enum
    {
        error = 1,
        warn = 2,
        info = 3,
        debug = 4
    } ERP_LOGLEVEL;

    extern int check_byte_order();
    extern void endian_swap(uint8_t* pdata, const size_t dsize, const size_t nelements);
    extern void logMessage(const ERP_LOGLEVEL logLevel, const bool showLevel, const char* msg, ...);

    extern void sincosLUTInitialize(float **sinLUT, float **cosLUT);
    extern bool sincosLUTLookup(float x, float *sinX, float *cosX);

    extern int resolveFilename(const char *logical, char *physical, int maxLength);

    extern int dumpFloatBufferToTextFile(const float *buffer, const size_t size, const char *filename);

    extern int findNextPowerofTwo(int value);

#ifdef  __cplusplus
}
#endif

#endif
