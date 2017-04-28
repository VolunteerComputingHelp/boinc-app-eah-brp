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

#include "erp_utilities.h"

#include <string.h>
#include <stdio.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>
#include <limits.h>

#include <boinc_api.h>

#include "demod_binary.h"


#define TIME_BUFFER_SIZE 10
#define LEVEL_BUFFER_SIZE (TIME_BUFFER_SIZE + 30)

#ifdef LOGLEVEL
extern const ERP_LOGLEVEL c_loglevel = LOGLEVEL;
#else
extern const ERP_LOGLEVEL c_loglevel = debug;
#endif

float sinSamples[] = {0.000000f, 0.098017f, 0.195090f, 0.290285f, 0.382683f, 0.471397f, 0.555570f, 0.634393f, 0.707107f, 0.773010f, 0.831470f, 0.881921f, 0.923880f, 0.956940f, 0.980785f, 0.995185f, 1.000000f, 0.995185f, 0.980785f, 0.956940f, 0.923880f, 0.881921f, 0.831470f, 0.773010f, 0.707107f, 0.634393f, 0.555570f, 0.471397f, 0.382683f, 0.290285f, 0.195091f, 0.098017f, 0.000000f, -0.098017f, -0.195090f, -0.290284f, -0.382683f, -0.471397f, -0.555570f, -0.634393f, -0.707107f, -0.773010f, -0.831469f, -0.881921f, -0.923880f, -0.956940f, -0.980785f, -0.995185f, -1.000000f, -0.995185f, -0.980785f, -0.956940f, -0.923880f, -0.881921f, -0.831470f, -0.773011f, -0.707107f, -0.634394f, -0.555570f, -0.471397f, -0.382684f, -0.290285f, -0.195091f, -0.098017f, -0.000000f};
float cosSamples[] = {1.000000f, 0.995185f, 0.980785f, 0.956940f, 0.923880f, 0.881921f, 0.831470f, 0.773010f, 0.707107f, 0.634393f, 0.555570f, 0.471397f, 0.382683f, 0.290285f, 0.195090f, 0.098017f, 0.000000f, -0.098017f, -0.195090f, -0.290285f, -0.382683f, -0.471397f, -0.555570f, -0.634393f, -0.707107f, -0.773010f, -0.831470f, -0.881921f, -0.923880f, -0.956940f, -0.980785f, -0.995185f, -1.000000f, -0.995185f, -0.980785f, -0.956940f, -0.923880f, -0.881921f, -0.831470f, -0.773011f, -0.707107f, -0.634393f, -0.555570f, -0.471397f, -0.382684f, -0.290285f, -0.195090f, -0.098017f, 0.000000f, 0.098017f, 0.195090f, 0.290285f, 0.382683f, 0.471397f, 0.555570f, 0.634393f, 0.707107f, 0.773010f, 0.831470f, 0.881921f, 0.923879f, 0.956940f, 0.980785f, 0.995185f, 1.000000f};


int check_byte_order()
{
    uint16_t word = 0x0001;
    uint8_t* byte = (uint8_t*) &word;
    return(byte[0] ? ERP_LITTLE_ENDIAN : ERP_BIG_ENDIAN);
}

void endian_swap(uint8_t* pdata, size_t dsize, size_t nelements)
{
    size_t i, j, indx;
    uint8_t tempbyte;

    if (dsize <= 1) return;

    for (i = 0; i < nelements; ++i) {
        indx = dsize;
        for (j = 0; j < dsize/2; ++j) {
            tempbyte = pdata[j];
            indx = indx - 1;
            pdata[j] = pdata[indx];
            pdata[indx] = tempbyte;
        }
        pdata = pdata + dsize;
    }
    return;
}

void getLocalTime(char* buffer) {
    time_t timeNow = time(0);
    struct tm* timeLocal = localtime(&timeNow);
    strftime(buffer, TIME_BUFFER_SIZE - 1, "%H:%M:%S", timeLocal);
}

void logMessage(const ERP_LOGLEVEL logLevel, const bool showLevel, const char* msg, ...)
{
    va_list varargs;
    FILE* output;
    char timeBuffer[TIME_BUFFER_SIZE] = {0};
    char levelBuffer[LEVEL_BUFFER_SIZE] = {0};
    static const pid_t pid = getpid();

    // return if under logging threshold
    if(logLevel > c_loglevel) {
        return;
    }

    // get current timestamp
    getLocalTime(timeBuffer);

    // prepare output
    switch(logLevel) {
        case error:
            output = stderr;
            snprintf(levelBuffer, LEVEL_BUFFER_SIZE, "[%s][%i][ERROR] ", timeBuffer, pid);
            break;
        case warn:
            output = stderr;
            snprintf(levelBuffer, LEVEL_BUFFER_SIZE, "[%s][%i][WARN ] ", timeBuffer, pid);
            break;
        case info:
            // we want these to be logged by BOINC
            output = stderr;
            snprintf(levelBuffer, LEVEL_BUFFER_SIZE, "[%s][%i][INFO ] ", timeBuffer, pid);
            break;
        case debug:
            output = stdout;
            snprintf(levelBuffer, LEVEL_BUFFER_SIZE, "[%s][%i][DEBUG] ", timeBuffer, pid);
            break;
        default:
            output = stderr;
            snprintf(levelBuffer, LEVEL_BUFFER_SIZE, "[%s][%i][UNKWN] ", timeBuffer, pid);
            break;
    }

    // prepend line feed if requested
    if(strlen(msg) > 0 && strncmp(msg, "\n", 1) == 0) {
        fprintf(output, "\n");
        // remove line feed if not single char
        if(strlen(msg) > 1) {
            msg++;
        }
    }

    // prepend log level if requested
    if(showLevel) {
        fprintf(output, levelBuffer);
    }
    else {
        fprintf(output, "------> ");
    }

    // print log message
    va_start(varargs, msg);
    vfprintf(output, msg, varargs);
    fflush(output);
    va_end(varargs);
}

void sincosLUTInitialize(float **sinLUT, float **cosLUT)
{
    // old unsused code, we're already initialized (hence the fixed "true")
    static bool initialized = true;

    if(!initialized) {
        unsigned int i;
        for (i=0; i <= ERP_SINCOS_LUT_RES; ++i) {
            sinSamples[i] = sin(ERP_TWO_PI * i * ERP_SINCOS_LUT_RES_F_INV);
            cosSamples[i] = cos(ERP_TWO_PI * i * ERP_SINCOS_LUT_RES_F_INV);
        }
        initialized = true;

        /*
        // print fixed LUT values to used for sinSamples/cosSamples
        for (i=0; i <= ERP_SINCOS_LUT_RES; ++i) {
            printf("%ff, ", sinSamples[i]);
        }
        printf("\n");
        for (i=0; i <= ERP_SINCOS_LUT_RES; ++i) {
            printf("%ff, ", cosSamples[i]);
            }
        */
    }

    *sinLUT = sinSamples;
    *cosLUT = cosSamples;
}

extern bool sincosLUTLookup(float x, float *sinX, float *cosX)
{
    float xt;
    int i0;
    float d, d2;
    float ts, tc;
    float dummy;

    xt = modff(ERP_TWO_PI_INV * x, &dummy); // xt in (-1, 1)
    if ( xt < 0.0f ) {
        xt += 1.0f;         // xt in [0, 1 )
    }

     // sanity check
    if ( xt < 0.0f || xt > 1.0f ) {
        logMessage(error, true, "sincosLUTLookup failed: xt = %f not in [0,1)\n", xt);
        return false;
    }

    // determine LUT index
    i0 = (int) (xt * ERP_SINCOS_LUT_RES_F + 0.5f);
    d = d2 = ERP_TWO_PI * (xt - ERP_SINCOS_LUT_RES_F_INV * i0);
    d2 *= 0.5f * d;

    // fetch sin/cos samples
    ts = sinSamples[i0];
    tc = cosSamples[i0];

    //use Taylor-expansions for sin/cos around samples
    (*sinX) = ts + d * tc - d2 * ts;
    (*cosX) = tc - d * ts - d2 * tc;

    return true;
}

int resolveFilename(const char *logical, char *physical, int maxLength)
{
    return boinc_resolve_filename(logical, physical, maxLength);
}

int dumpFloatBufferToTextFile(const float *buffer,
                              const size_t size,
                              const char  *filename)
{
    FILE *output = fopen(filename, "w");
    if(NULL == output) {
        logMessage(error, true, "Error opening file \"%s\" for buffer dump!\n", filename);
        return(RADPUL_EFILE);
    }
    for(size_t i = 0; i < size; ++i) {
        fprintf(output, "%e\n", buffer[i]);
    }
    fclose(output);

    logMessage(debug, true, "Successfully wrote buffer to \"%s\"...\n", filename);

    return(0);
}

int findNextPowerofTwo(int value)
{
    // sanity check (no negative values allowed)
    if(value < 0) {
        logMessage(error, true, "findNextPowerofTwo(): no negative input values allowed!\n");
        return (-1);
    }

    int power = 0;
    int max = 0;

    // determine power and its value
    do {
        max = (int) pow(2, power);
        if(max >= INT_MAX) {
            logMessage(warn, true, "MAX_INT reached! Returning %u (power: %i)\n", max, power);
            return(max);
        }
        power++;
    }
    while(max < value);

    logMessage(debug, true, "Next power-of-2 for %u determined to be: %u (power: %i)\n", value, max, power-1);

    return(max);
}
