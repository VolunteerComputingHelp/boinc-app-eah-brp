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
 *   Demodulates dedispersed time series using a bank of orbital           *
 *   parameters. After this step, an FFT of the resampled time series is   *
 *   searched for pulsed, periodic signals by harmonic summing.            *
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


// includes
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <fftw3.h>
#include <zlib.h>
#include "rngmed.h"
#include "structs.h"
#include "erp_utilities.h"
#include "erp_git_version.h"

#ifdef BOINCIFIED
// BOINC-specific includes
#include <time.h>
#include "demod_binary.h"
#include "erp_boinc_ipc.h"
#include "erp_boinc_wrapper.h"
#include "svn_version.h"
#include "filesys.h"
// BOINC-specific function mapping
#define fopen boinc_fopen
#define gzopen boinc_gzopen
#define rename boinc_rename
#endif

#include "hs_common.h"

#ifdef USE_CPU_RESAMP
#include "demod_binary_resamp_cpu.h"
#endif

#ifdef USE_FFTW_FFT
#include "demod_binary_fft_fftw.h"
#endif

#if defined USE_CUDA
#include "cuda/app/cuda_utilities.h"
#include "cuda/app/demod_binary_cuda.h"
#include "cuda/app/demod_binary_hs_cuda.cuh"
#elif defined USE_OPENCL
#include "boinc_opencl.h"
#include "opencl/app/demod_binary_ocl.h"
#else
#include "demod_binary_hs_cpu.h"
#endif

#define N_CAND_5 100   // number of candidates for each harmonic = number of candidates reported back
#define N_CAND 500     // number of candidates stored = five times the number of candidates reported back
#define TIME_FORMAT "%Y-%m-%dT%H:%M:%S+00:00" // used for the result file header
#define TIME_LENGTH 30                        // used for the result file header

// user input
typedef struct
{
  float f0;                             // maximum fundamental signal frequency (Hz) to search for
  float padding;                        // factor of frequency overresolution achieved by mean-padding
  float fA;                             // overall false alarm probability
  unsigned int window;                   // window width for the running median in frequency bins
  unsigned short int white;              // switch for power spectrum whitening
  unsigned short int debug;              // switch for debug mode
  char *inputfile;                       // name of the input file
  char *outputfile;                      // pointer to name of the output file
  char *templatebank;                    // pointer to name of the template bank
  char *checkpointfile;                  // pointer to name of the checkpoint file
  char *zaplistfile;                     // pointer to name of the zaplist file
  char outputfile_tmp[FN_LENGTH + 4];    // name of the temporary output file
  char checkpointfile_tmp[FN_LENGTH + 4];// name of the temporary checkpoint file
} User_Variables;

// prototypes
int compare_structs_by_P(const void * const ptr1, const void * const ptr2);
int compare_structs_by_ifa(const void * const ptr1, const void * const ptr2);
int set_checkpoint(const User_Variables * const uvar,
                   const CP_Header * const cp_head,
                   const CP_cand * const candidates);


/*+++++++++++++++++++++++++++++
+    start main program       +
+++++++++++++++++++++++++++++*/
int MAIN (int argc, char *argv[])
{
  // structs
  User_Variables uvar;                      // parameters specified in the command line
  DD_Header data_head;                      // header of the dedispersed time series
  CP_Header cp_head;                        // header of the checkpoint file
  t_pulsar_search search_params_tmp;        // for communication with screensaver

  // candiate array
  CP_cand *candidates_all = NULL;           // array of structs for all candidates

  // file pointers
  FILE *output = NULL, *templatebank = NULL, *checkpoint = NULL;
  gzFile input;

  // pointers for time series variables and FFT plan.
  // at top level to avoid unnecessary allocation and destruction of FFT plan
  int t_series_4bit = 1;                     // boolean flag to indicate 4-bit data format (default)
  unsigned char * t_series_dd_comp4 = NULL;  // nibble time series
  signed char * t_series_dd_comp8 = NULL;    // single-byte time series
  DIfloatPtr t_series_dd;                    // dedispersed time series
  DIfloatPtr t_series_resamp;                // resampled timeseries

  float *sumspec[5];                        // spectra of summed harmonics
  int32_t *dirty[5];                        // flags for non-zero pages (per sumpspec array)
  unsigned int nr_dirty_pages;              // total number of pages for sparse cand. selection
  unsigned int dirty_page_count;            // counts sumspec pages marked dirty (for logging only)
  unsigned int fft_size;                    // size of the FFT
  DIfloatPtr powerspectrum ;                // power spectrum of the resampled time series
  float *sinLUTsamples = NULL;              // sin/cos LUT samples
  float *cosLUTsamples = NULL;              // sin/cos LUT samples

  unsigned int window_2;                    // half the size of the running median window
  unsigned int fundamental_idx_hi;          // frequency bin of the highest fundamental frequency searched for
  unsigned int harmonic_idx_hi;             // frequency bin of the highest harmonic frequency searched for


  float t_obs; // observation time in seconds
  float dt;
  float step_inv; // inverse of a time sample

#ifdef BOINCIFIED
  BOINC_STATUS boinc_status;
#endif

#if defined USE_CUDA || defined USE_OPENCL
  int coprocDeviceId = -1;                  // Co-processor (CUDA/OpenCL) device id
  int coprocDeviceIdGiven = 0;              // Did we get a device ID via command line (bool)?
#endif

  // book-keeping
  unsigned int template_total_amount = 0;   // total amount of templates
  unsigned int template_counter;            // count the templates already done
  unsigned int n_unpadded;                  // number of the time samples in the original time series
  unsigned int n_unpadded_format;           // modified number of samples based on input file format (4-bit/8-bit)
  unsigned int i;                           // loop and index vars

  int result;                               // return value storage
  char line[FN_LENGTH];                     // string for parsing lines from the templatebank

  // vars for sky position in screensaver
  float hrs;                                // hours in RA, degrees in DEC
  float min;                                // minutes in RA, arcminutes in DEC
  float sec;                                // seconds in RA, arcseconds in DEC

  // thresholds

  float thrA[5];			      // threshold for harmonics


  t_series_dd.host_ptr       = NULL;
  t_series_resamp.host_ptr   = NULL;
  powerspectrum.host_ptr     = NULL;

  // vars for result header
  char resultTimeISO[TIME_LENGTH + 1] = {0};
  time_t timeValue;
  struct tm* timeUTC;

  // scan format constants
  static const char* c_template_scan_format = "%lg %lg %lg\n";

  // determine actual "endianess"
  int big_endian = check_byte_order() == ERP_BIG_ENDIAN ? 1 : 0;

  // set user variables to defaults
  uvar.outputfile = NULL;
  uvar.templatebank = NULL;
  uvar.checkpointfile = NULL;
  uvar.inputfile = NULL;
  uvar.zaplistfile = NULL;
  sprintf(uvar.outputfile_tmp, "");
  sprintf(uvar.checkpointfile_tmp, "");
  uvar.white = 0;
  uvar.f0 = 250.0;
  uvar.window = 1000;
  uvar.padding = 1.0;
  uvar.fA = 0.04;
  uvar.debug = 0;

  // parse command line arguments
  i = 1;
  while (i < argc)
    {
      if ((strcmp(argv[i], "-W") == 0) || (strcmp(argv[i], "--whitening") == 0))
	{
	  uvar.white = 1;
	  i++;
	}
      else if ((strcmp(argv[i], "-P") == 0) || (strcmp(argv[i], "--padding") == 0))
	{
	  // sanity check
	  if(atof(argv[i+1]) < 1.0)
	    {
	      logMessage(error, true, "Nonsense value: padding factor %g < 1.0.\n", atof(argv[i+1]));
	      return(RADPUL_EVAL);
	    }
	  else if(atof(argv[i+1]) > 10.0)
	    {
	      logMessage(error, true, "Nonsense value: padding factor %g > 10.0.\n", atof(argv[i+1]));
	      return(RADPUL_EVAL);
	    }
	  else
	    {
	      uvar.padding =  atof(argv[i+1]);
	      i += 2;
	    }
	}
      else if ((strcmp(argv[i], "-B") == 0) || (strcmp(argv[i], "--box") == 0))
	{
	  // sanity check
	  if(atoi(argv[i+1]) < 0)
	    {
	      logMessage(error, true, "Nonsense value: window size for running median %d is negative.\n", atoi(argv[i+1]));
	      return(RADPUL_EVAL);
	    }
	  else if(atoi(argv[i+1]) > 250000)
	    {
	      logMessage(error, true, "Nonsense value: window size for running median too large: %d.\n", atoi(argv[i+1]));
	      return(RADPUL_EVAL);
	    }
	  else
	    {
	      uvar.window = atoi(argv[i+1]);
	      i += 2;
	    }
	}
      else if ((strcmp(argv[i], "-z") == 0) || (strcmp(argv[i], "--debug") == 0))
	{
	  uvar.debug = 1;
	  logMessage(debug, true, "Running program in debugging mode.\n");
	  i++;
	}
      else if ((strcmp(argv[i], "-f") == 0) || (strcmp(argv[i], "--f0") == 0))
	{
	  // sanity check
	  if(atof(argv[i+1]) < 0.0)
	    {
	      logMessage(error, true, "Nonsense value: upper limit for search frequency %g is negative.\n", atof(argv[i+1]));
	      return(RADPUL_EVAL);
	    }
	  else if(atof(argv[i+1]) > 16.0e3)
	    {
	      logMessage(error, true, "Nonsense value: upper limit for search frequency %g > 16 kHz.\n", atof(argv[i+1]));
	      return(RADPUL_EVAL);
	    }
	  else
	    {
	      uvar.f0 = atof(argv[i+1]);
	      i += 2;
	    }
	}
      else if ((strcmp(argv[i], "-A") == 0) || (strcmp(argv[i], "--false_alarm") == 0))
	{
	  // sanity check
	  if(atof(argv[i+1]) < 0.0)
	    {
	      logMessage(error, true, "Nonsense value: false alarm rate %g is negative.\n", atof(argv[i+1]));
	      return(RADPUL_EVAL);
	    }
	  else if(atof(argv[i+1]) > 1.0)
	    {
	      logMessage(error, true, "Nonsense value: false alarm rate %g > 1.0.\n", atof(argv[i+1]));
	      return(RADPUL_EVAL);
	    }
	  else
	    {
	      uvar.fA = atof(argv[i+1]);
	      i += 2;
	    }
	}
      else if ((strcmp(argv[i], "-i") == 0) || (strcmp(argv[i], "--input_file") == 0))
	{
	  uvar.inputfile = argv[i+1];

	  if(NULL == uvar.inputfile)
	    {
	      logMessage(error, true, "Couldn't prepare input file name: %s\n", argv[i+1]);
	      return(RADPUL_EFILE);
	    }

	  if(strstr(uvar.inputfile, ".binary")) {
	      t_series_4bit = 0;
	      logMessage(debug, true, "Using 8-bit format instead of 4-bit format for input file: %s\n", uvar.inputfile);
	  }
	  else if(!strstr(uvar.inputfile, ".bin4")) {
	      logMessage(error, true, "Unknown file format (extension) for input file: %s\n", uvar.inputfile);
	      return(RADPUL_EFILE);
	  }
	  i += 2;
	}
     else if ((strcmp(argv[i], "-o") == 0) || (strcmp(argv[i], "--output_file") == 0))
	{
	  uvar.outputfile = argv[i+1];

	  if(NULL == uvar.outputfile)
	    {
	      logMessage(error, true, "Couldn't prepare output file name: %s\n", argv[i+1]);
	      return(RADPUL_EFILE);
	    }

	  if(snprintf(uvar.outputfile_tmp, sizeof(uvar.outputfile_tmp), "%s.tmp", argv[i+1]) >= sizeof(uvar.outputfile_tmp))
	    {
	      logMessage(error, true, "Couldn't prepare temporary output file name: %s\n", argv[i+1]);
	      return(RADPUL_EFILE);
	    }
	  i += 2;
	}
     else if ((strcmp(argv[i], "-c") == 0) || (strcmp(argv[i], "--checkpoint_file") == 0))
	{
	  uvar.checkpointfile = argv[i+1];

	  if(NULL == uvar.checkpointfile)
	    {
	      logMessage(error, true, "Couldn't prepare checkpoint file name: %s\n", argv[i+1]);
	      return(RADPUL_EFILE);
	    }

	  if(snprintf(uvar.checkpointfile_tmp, sizeof(uvar.checkpointfile_tmp), "%s.tmp", argv[i+1]) >= sizeof(uvar.checkpointfile_tmp))
	    {
	      logMessage(error, true, "Couldn't prepare temporary checkpoint file name: %s\n", argv[i+1]);
	      return(RADPUL_EFILE);
	    }
	  i += 2;
	}
      else if ((strcmp(argv[i], "-t") == 0) || (strcmp(argv[i], "--template_bank") == 0))
	{
	  uvar.templatebank = argv[i+1];

	  if(NULL == uvar.templatebank)
	    {
	      logMessage(error, true, "Couldn't prepare template bank file name: %s\n", argv[i+1]);
	      return(RADPUL_EFILE);
	    }
	  i += 2;
	}
      else if ((strcmp(argv[i], "-l") == 0) || (strcmp(argv[i], "--zaplist_file") == 0))
	{
	  uvar.zaplistfile = argv[i+1];

	  if(NULL == uvar.zaplistfile)
	    {
	      logMessage(error, true, "Couldn't prepare zaplist file name: %s\n", argv[i+1]);
	      return(RADPUL_EFILE);
	    }
	  i += 2;
	}
#if defined USE_CUDA || defined USE_OPENCL
      else if ((strcmp(argv[i], "-D") == 0) || (strcmp(argv[i], "--device") == 0)) {
          long int temp = -1;

          // input conversion
          if(isdigit(*(argv[i+1])))
          {
              errno = 0;
              temp = strtol(argv[i+1], (char**)NULL, 10);
          }
          else
          {
              logMessage(error, true, "Invalid GPU device ID encountered: %s\n", argv[i+1]);
              return(RADPUL_EVAL);
          }
          if(errno != 0)
          {
              logMessage(error, true, "GPU device ID couldn't be parsed: %s\n", strerror(errno));
              return(RADPUL_EVAL);
          }

          // sanity check
          if(temp < 0)
          {
              logMessage(error, true, "Nonsense value: GPU device ID %i is negative.\n", atoi(argv[i+1]));
              return(RADPUL_EVAL);
          }
          else
          {
              coprocDeviceId = (int)temp;
              coprocDeviceIdGiven = 1;
              i += 2;
          }
      }
#endif
    else if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "--help") == 0))
	{
	  printf("\nUsage: %s [options], options are:\n\n", argv[0]);
	  printf(" -h, --help\t\t\tboolean\tPrint this message\n");
	  printf(" -i, --input_file\t\tstring\tThe name of the input file.\n");
	  printf(" -o, --output_file\t\tstring\tThe name of the candidate output file.\n");
	  printf(" -t, --template_bank\t\tstring\tThe name of the random template bank.\n");
	  printf(" -c, --checkpoint_file\t\tstring\tThe name of the checkpoint file.\n");
	  printf(" -l, --zaplist_file\t\tstring\tThe name of the zaplist file.\n");
	  printf(" -f, --f0\t\t\tfloat\tThe maximum signal frequency (in Hz)\n");
	  printf(" -A, --false_alarm\t\tfloat\tFalse alarm probability.\n");
	  printf(" -P, --padding\t\t\tfloat\tThe frequency over-resolution factor.\n");
	  printf(" -W, --whitening\t\tboolean\tSwitch for power spectrum whitening and line zapping.\n");
	  printf(" -B, --box\t\t\tint\tWindow width for the running median in frequeny bins.\n");
#if defined USE_CUDA || defined USE_OPENCL
      printf(" -D, --device\t\tinteger\tThe GPU device ID to be used.\n");
#endif
	  printf(" -z, --debug\t\t\tboolean\tRun program in debug mode.\n");
	  printf("\n");
	  return(RADPUL_EMISC);
	}
      else
	{
	  logMessage(error, true, "\nUnknown option \"%s\". Use '%s --help'.\n\n", argv[i], argv[0]);
	  return(RADPUL_EMISC);
	}
    }// end: while (i < argc)


  logMessage(info, true, "Starting data processing...\n");

#if defined(BOINCIFIED) && (defined(USE_CUDA) || defined(USE_OPENCL))
      boinc_begin_critical_section();
      logMessage(debug, true, "Entered critical section: CUDA/OpenCL initialization\n");
#endif

#if defined USE_CUDA
  // set up CUDA device
  result = initialize_cuda(coprocDeviceIdGiven, &coprocDeviceId);
  if(result != 0) return result;
#elif defined USE_OPENCL
  cl_platform_id boincPlatformId = NULL;
  cl_device_id boincDeviceId = NULL;
  // do we run under BOINC control?
  if(!boinc_is_standalone()) {
    result = boinc_get_opencl_ids(&boincDeviceId, &boincPlatformId);
    if(CL_SUCCESS == result) {
      // BOINC takes the lead, ignore values passed manually (just to make sure)
      coprocDeviceIdGiven = 0;
      coprocDeviceId = -1;
      // acquire OpenCL device determined by BOINC
      result = initialize_ocl(coprocDeviceIdGiven, &coprocDeviceId, boincPlatformId, boincDeviceId);
    }
    else {
      logMessage(error, true, "Failed to get OpenCL platform/device info from BOINC (error: %i)!\n", result);
    }
  }
  else {
    // set up OpenCL device manually or determine "best" of first platform
    logMessage(debug, true, "Running in standalone mode, so we take care of OpenCL platform/device management...\n");
    result = initialize_ocl(coprocDeviceIdGiven, &coprocDeviceId, boincPlatformId, boincDeviceId);
  }
  if(result != 0) return result;
#endif

#if defined(BOINCIFIED) && (defined(USE_CUDA) || defined(USE_OPENCL))
  boinc_end_critical_section();
  logMessage(debug, true, "Left critical section: CUDA/OpenCL initialization\n");
#endif

  // allocate memory for candidates array of structs
  candidates_all = (CP_cand *)calloc(N_CAND, sizeof(CP_cand));
  if(candidates_all == NULL)
    {
      logMessage(error, true, "Couldn't allocate %d bytes of memory for candidates_all.\n", N_CAND*sizeof(CP_cand));
      return(RADPUL_EMEM);
    }

  // open template bank
  logMessage(debug, true, "Opening template bank file: %s\n", uvar.templatebank);
  templatebank = fopen(uvar.templatebank, "r");
  if(templatebank == NULL)
    {
      logMessage(error, true, "Couldn't open template bank file: %s (%s).\n", uvar.templatebank, strerror(errno));
      return(RADPUL_EIO);
    }

  // determine total number of templates
  while(1)
    {
      double P_tmp, tau_tmp, psi_tmp;
      char line[FN_LENGTH];

      // check whether file is sane, fgets can read line and the EOF isn't reached by this scan
      if(!ferror(templatebank) && (NULL != fgets(line, FN_LENGTH, templatebank)) && !feof(templatebank))
	{
	  // parse line and check whether three values could be read
	  if(sscanf(line, c_template_scan_format, &P_tmp, &tau_tmp, &psi_tmp) == 3)
	    {
	      template_total_amount++;
	    }
	  else // if three values could not be read
	    {
	      logMessage(error, true, "Line %d in templatebank %s seems to be damaged.\n", template_total_amount + 1, uvar.templatebank);
	      return(RADPUL_EVAL);
	    }
	}
	else if(feof(templatebank)) // if EOF reached, break out of while loop
	  {
	    break;
	  }
	else // if line couldn't be read and EOF is not reached -> error has happened
	  {
	    logMessage(error, true, "Couldn't determine number of templates in %s (%s).\n", uvar.templatebank, strerror(errno));
	    return(RADPUL_EIO);
	  }
    }

  // reset file position
  if(fseek(templatebank, 0, SEEK_SET))
    {
      logMessage(error, true, "Couldn't reset template bank file: %s (%s).\n", uvar.templatebank, strerror(errno));
      return(RADPUL_EIO);
    }

  logMessage(debug, true, "Total amount of templates: %i\n", template_total_amount);

  // open checkpoint input stream
  logMessage(debug, true, "Opening checkpoint file: %s\n", uvar.checkpointfile);
  checkpoint = fopen(uvar.checkpointfile,"rb");
  if(checkpoint == NULL)
    {
      logMessage(info, true, "Checkpoint file unavailable: %s (%s).\n", uvar.checkpointfile, strerror(errno));
      logMessage(info, false, "Starting from scratch...\n");
      template_counter = 0;
      if(snprintf(cp_head.originalfile, sizeof(cp_head.originalfile), uvar.inputfile) >= sizeof(cp_head.originalfile))
	{
    	  logMessage(error, true, "Couldn't write input file %s name to checkpoint header.\n", uvar.inputfile);
	  return(RADPUL_EFILE);
	}
    }
  else
    {
      // read the header information from checkpoint file
      if(fread(&cp_head, sizeof(CP_Header), 1, checkpoint) != 1)
	{
	  // if header is broken: just die
          logMessage(error, true, "Premature end of data header in file: %s (%s)\n", uvar.checkpointfile, strerror(errno));
	  return(RADPUL_EFILE);
	}
      else
	{
	  logMessage(debug, true, "Header read from checkpoint file.\n");

    	  // determine if there's still work left
    	  if(cp_head.n_template == template_total_amount)
	    {
	      logMessage(info, true, "Thank you but this work unit has already been processed completely...\n");
	    }
	  else if(cp_head.n_template < template_total_amount)
	    {
	      logMessage(info, true, "Continuing work on %s at template no. %d\n", cp_head.originalfile, cp_head.n_template);
	    }
	  else if(cp_head.n_template > template_total_amount)
	    {
	      logMessage(error, true, "Header checkpoint file %s contains inconsistent information about number of templates done (%d > %d).\n", uvar.checkpointfile, cp_head.n_template, template_total_amount);
	      return(RADPUL_EFILE);
	    }

	  // check command line information equals header information
	  if(strcmp(uvar.inputfile, cp_head.originalfile))
	    {
	      logMessage(error, true, "Input file on command line %s doesn't agree with input file %s from checkpoint header.\n", uvar.inputfile, cp_head.originalfile);
	      return(RADPUL_EFILE);
	    }

	  // read in candidates from checkpoint file
	  if(fread(candidates_all, sizeof(CP_cand), N_CAND, checkpoint) != N_CAND)
	    {
	      // XXX checksum for checkpoint
	      logMessage(error, true, "Couldn't read all candidates from checkpoint (%s)!", strerror(errno));
	      return(RADPUL_EIO);
	    }

	  if(ferror(checkpoint))
	    {
	      logMessage(error, true, "Couldn't read checkpoint file: %s (%s)\n", uvar.checkpointfile, strerror(errno));
	      return(RADPUL_EIO);
	    }

	  if(fclose(checkpoint))
	    {
	      logMessage(error, true, "Couldn't close checkpoint file: %s (%s).\n", uvar.checkpointfile, strerror(errno));
	      return(RADPUL_EIO);
	    }

	  if(uvar.debug)
	    {
	      logMessage(debug, true, "Candidates found so far:\n");
	      for(i = 0; i < N_CAND; i++)
		{
		  logMessage(debug, false, "%d %6.12f %6.12f %6.12f %6.12f %d\n", candidates_all[i].f0, candidates_all[i].power, candidates_all[i].P_b, candidates_all[i].tau, candidates_all[i].Psi, candidates_all[i].n_harm);
		}
	    }

	  // go to the next unused template in the template bank if checkpoint file exists
	  for( i = 0; i < cp_head.n_template; i++)
	    {
	      float P_tmp, tau_tmp, psi_tmp;
	      char line[FN_LENGTH];

	      if(NULL == fgets(line, FN_LENGTH, templatebank))
		{
		  if(feof(templatebank))
		    {
		      logMessage(error, true, "Premature end of data in: %s\n", uvar.templatebank);
		      return(RADPUL_EIO);
		    }
		  else
		    {
		      logMessage(error, true, "Error while reading data from %s\n", uvar.templatebank);
		      return(RADPUL_EIO);
		    }
		}
	      else if(sscanf(line, c_template_scan_format, &P_tmp, &tau_tmp, &psi_tmp) != 3)
		{
		  logMessage(error, true, "Couldn't read complete line %d in %s\n", i, uvar.templatebank);
		  return(RADPUL_EIO);
		}
	    }
	  // set template_counter to value in checkpoint header
	  template_counter = cp_head.n_template;
	}
    }


  // open input stream
  logMessage(debug, true, "Opening input file: %s\n", uvar.inputfile);
  input = gzopen(uvar.inputfile,"rb");
  if(input == NULL)
    {
      logMessage(error, true, "Couldn't open input file: %s (%s)\n", uvar.inputfile, strerror(errno));
      return(RADPUL_EIO);
    }

  // read data header
  logMessage(debug, true, "Reading header from time series file: %s\n", uvar.inputfile);
  if(gzread(input, &data_head, sizeof(DD_Header)) != sizeof(DD_Header))
    {
      logMessage(error, true, "Premature end of data in file: %s (%s; %s)\n", uvar.inputfile, gzerror(input, NULL), strerror(errno));
      gzclose_r(input);
      return(RADPUL_EIO);
    }

  // convert little endian input file header if necessary
  if(big_endian)
    {
      // double
      endian_swap((uint8_t*) &data_head.tsample, sizeof(data_head.tsample), 1);
      endian_swap((uint8_t*) &data_head.tobs, sizeof(data_head.tobs), 1);
      endian_swap((uint8_t*) &data_head.timestamp, sizeof(data_head.timestamp), 1);
      endian_swap((uint8_t*) &data_head.fcenter, sizeof(data_head.fcenter), 1);
      endian_swap((uint8_t*) &data_head.fchan, sizeof(data_head.fchan), 1);
      endian_swap((uint8_t*) &data_head.RA, sizeof(data_head.RA), 1);
      endian_swap((uint8_t*) &data_head.DEC, sizeof(data_head.DEC), 1);
      endian_swap((uint8_t*) &data_head.gal_l, sizeof(data_head.gal_l), 1);
      endian_swap((uint8_t*) &data_head.gal_b, sizeof(data_head.gal_b), 1);
      endian_swap((uint8_t*) &data_head.AZstart, sizeof(data_head.AZstart), 1);
      endian_swap((uint8_t*) &data_head.ZAstart, sizeof(data_head.ZAstart), 1);
      endian_swap((uint8_t*) &data_head.ASTstart, sizeof(data_head.ASTstart), 1);
      endian_swap((uint8_t*) &data_head.LSTstart, sizeof(data_head.LSTstart), 1);
      endian_swap((uint8_t*) &data_head.DM, sizeof(data_head.DM), 1);
      endian_swap((uint8_t*) &data_head.scale, sizeof(data_head.scale), 1);
      // uint32
      endian_swap((uint8_t*) &data_head.filesize, sizeof(data_head.filesize), 1);
      endian_swap((uint8_t*) &data_head.datasize, sizeof(data_head.datasize), 1);
      endian_swap((uint8_t*) &data_head.nsamples, sizeof(data_head.nsamples), 1);
      // uint 16
      endian_swap((uint8_t*) &data_head.smprec, sizeof(data_head.smprec), 1);
      endian_swap((uint8_t*) &data_head.nchan, sizeof(data_head.nchan), 1);
      endian_swap((uint8_t*) &data_head.nifs, sizeof(data_head.nifs), 1);
      endian_swap((uint8_t*) &data_head.lagformat, sizeof(data_head.lagformat), 1);
      endian_swap((uint8_t*) &data_head.sum, sizeof(data_head.sum), 1);
      endian_swap((uint8_t*) &data_head.level, sizeof(data_head.level), 1);
    }

  // dump header information to stdout
  if(uvar.debug)
    {
      logMessage(info, true, "Header contents:\n");
      logMessage(info, false, "Original WAPP file: %s\n", data_head.originalfile);
      logMessage(info, false, "Sample time in microseconds: %g\n", data_head.tsample);
      logMessage(info, false, "Observation time in seconds: %.8g\n", data_head.tobs);
      logMessage(info, false, "Time stamp (MJD): %.17g\n", data_head.timestamp);
      logMessage(info, false, "Number of samples/record: %d\n", data_head.smprec);
      logMessage(info, false, "Center freq in MHz: %.10g\n", data_head.fcenter);
      logMessage(info, false, "Channel band in MHz: %.9g\n", data_head.fchan);
      logMessage(info, false, "Number of channels/record: %d\n", data_head.nchan);
      logMessage(info, false, "Nifs: %d\n", data_head.nifs);
      logMessage(info, false, "RA (J2000): %.12g\n", data_head.RA);
      logMessage(info, false, "DEC (J2000): %.12g\n", data_head.DEC);
      logMessage(info, false, "Galactic l: %.7g\n", data_head.gal_l);
      logMessage(info, false, "Galactic b: %.7g\n", data_head.gal_b);
      logMessage(info, false, "Name: %s\n", data_head.name);
      logMessage(info, false, "Lagformat: %d\n", data_head.lagformat);
      logMessage(info, false, "Sum: %d\n", data_head.sum);
      logMessage(info, false, "Level: %d\n", data_head.level);
      logMessage(info, false, "AZ at start: %.9g\n", data_head.AZstart);
      logMessage(info, false, "ZA at start: %.9g\n", data_head.ZAstart);
      logMessage(info, false, "AST at start: %.9g\n", data_head.ASTstart);
      logMessage(info, false, "LST at start: %.9g\n", data_head.LSTstart);
      logMessage(info, false, "Project ID: %s\n", data_head.proj_id);
      logMessage(info, false, "Observers: %s\n", data_head.observers);
      logMessage(info, false, "File size (bytes): %d\n", data_head.filesize);
      logMessage(info, false, "Data size (bytes): %d\n", data_head.datasize);
      logMessage(info, false, "Number of samples: %d\n", data_head.nsamples);
      logMessage(info, false, "Trial dispersion measure: %g cm^-3 pc\n", data_head.DM);
      logMessage(info, false, "Scale factor: %g\n", data_head.scale);
    }

  /*----------------------------------------------
    NO SANITY CHECK OF HEADER DATA HERE AS LONG AS
    THE FINAL HEADER FORMAT IS NOT EXACTLY DEFINED
    ----------------------------------------------*/

  // drop header information into temporary param array
  // convert RA to radian
  hrs = floor(data_head.RA/10000.0);
  min = floor((data_head.RA - 10000.0*hrs)/100.0);
  sec = data_head.RA - 10000.0*hrs - 100.0*min;

  // set sky position RA in temporary struct
  search_params_tmp.skypos_rac = M_PI*(hrs/12.0 + min/720.0 + sec/43200.0);

  // convert DEC to radian
  if(data_head.DEC < 0.0)
    {
      hrs = floor(-data_head.DEC/10000.0);
      min = floor(-(data_head.DEC + 10000.0*hrs)/100.0);
      sec = -(data_head.DEC + 10000.0*hrs + 100.0*min);

      // set sky position DEC in temporary struct
      search_params_tmp.skypos_dec = -M_PI*(hrs/180.0 + min/10800.0 + sec/648000.0);
    }
  else
    {
      hrs = floor(data_head.DEC/10000.0);
      min = floor((data_head.DEC - 10000.0*hrs)/100.0);
      sec = data_head.DEC - 10000.0*hrs - 100.0*min;

      // set sky position DEC in temporary struct
      search_params_tmp.skypos_dec = M_PI*(hrs/180.0 + min/10800.0 + sec/648000.0);
    }


  // set DM in temporary struct
  search_params_tmp.dispersion_measure = data_head.DM;

  // remember number of original time samples
  n_unpadded = (unsigned int)data_head.nsamples;
  n_unpadded_format = t_series_4bit ? n_unpadded * 0.5 : n_unpadded;

  // apply padding
  data_head.nsamples = (int)(uvar.padding*data_head.nsamples + 0.5);

  // allocate memory for compressed dedispersed time series
  logMessage(debug, true, "Allocating memory for dedispersed compressed time series...\n");
  if(t_series_4bit) {
      t_series_dd_comp4 = (unsigned char *) calloc(n_unpadded_format, sizeof(unsigned char));
      if(t_series_dd_comp4 == NULL)
        {
          logMessage(error, true, "Couldn't allocate %d bytes of memory for dedispersed 4-bit compressed time series.\n", n_unpadded_format*sizeof(unsigned char));
          return(RADPUL_EMEM);
        }
  }
  else {
      t_series_dd_comp8 = (signed char *) calloc(n_unpadded_format, sizeof(signed char));
      if(t_series_dd_comp8 == NULL)
        {
          logMessage(error, true, "Couldn't allocate %d bytes of memory for dedispersed 8-bit compressed time series.\n", n_unpadded_format*sizeof(signed char));
          return(RADPUL_EMEM);
        }
  }

  // read dedispersed timeseries
  logMessage(debug, true, "Reading dedispersed time series from file: %s\n", uvar.inputfile);
  if( ( t_series_4bit && gzread(input, t_series_dd_comp4, sizeof(unsigned char) * n_unpadded_format) != sizeof(unsigned char) * n_unpadded_format) ||
      (!t_series_4bit && gzread(input, t_series_dd_comp8, sizeof(signed char)   * n_unpadded_format) != sizeof(signed char)   * n_unpadded_format) )
    {
      logMessage(error, true, "Premature end of data in file: %s (%s; %s)\n", uvar.inputfile, gzerror(input, NULL), strerror(errno));
      gzclose_r(input);
      return(RADPUL_EIO);
    }

  // close input stream
  if(gzclose_r(input))
    {
      logMessage(error, true, "Couldn't close input file: %s (%s; %s)\n", uvar.inputfile, gzerror(input, NULL), strerror(errno));
      return(RADPUL_EIO);
    }

  // allocate memory for uncompressed dedispersed time series
  logMessage(debug, true, "Allocating memory for dedispersed uncompressed time series.\n");
  t_series_dd.host_ptr = (float *) calloc(n_unpadded, sizeof(float));

  if(t_series_dd.host_ptr == NULL)
    {
      logMessage(error, true, "Couldn't allocate %d bytes of memory for dedispersed uncompressed  time series.\n", n_unpadded*sizeof(float));
      return(RADPUL_EMEM);
    }

  // convert compressed single-byte data back into 4-byte floats
  for(i = 0; i < n_unpadded_format; i ++)
    {
      if(t_series_4bit) {
          // read in the two samples packed in one char and convert into floats
          t_series_dd.host_ptr[2*i + 1] = (float)(t_series_dd_comp4[i]%16)/data_head.scale;
          t_series_dd.host_ptr[2*i]     = (float)(t_series_dd_comp4[i]>>4)/data_head.scale;
      }
      else {
          // "unpack" single char sample
          t_series_dd.host_ptr[i] = t_series_dd_comp8[i]/data_head.scale;
      }
    }

  // free compressed time series
  if(t_series_4bit) {
      free(t_series_dd_comp4);
  }
  else {
      free(t_series_dd_comp8);
  }


  /*---------------------------------------------------------------------------
    WHITENING OF THE POWERSPECTRUM OF THE TIME SERIES AND ZAPPING OF KNOWN RFIS
    -------------------------------------------------------------------------*/

  logMessage(debug, true, "Starting whitening of the powerspectrum of the timeseries and zapping of known RFIs.\n");
  if(uvar.white)
    {
      FILE *zaplist = NULL;
      float norm_factor;
      float *time_series;
      float *powerspectrum;
      float *running_median;
      int32_t seed = 0;
      int white_size;
      int line_counter = 0;
      unsigned int fft_size;
      unsigned int window_2 = (int)(0.5*uvar.window + 0.5);
      fftwf_complex *fft;
      fftwf_plan fft_plan;
      gsl_rng *r;

      // if input is N real numbers, output has N/2 + 1 non-redundant entries
      fft_size = (unsigned int)(0.5*data_head.nsamples + 0.5) + 1;

      // sanity check for window size
      if(fft_size < uvar.window)
	{
	  logMessage(error, true, "Running median window (%d bins) is too wide for data set (%d bins)!\n", uvar.window, fft_size);
	  return(RADPUL_EVAL);
	}

      // use fftwf_malloc for allocation: this is recommended by FFTW (see manual: section 2.1, page 3)
      fft = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex)*fft_size);
      if(fft == NULL)
	{
	  logMessage(error, true, "Couldn't allocate %d bytes of memory for FFT.\n", fft_size*sizeof(fftwf_complex));
	  return(RADPUL_EMEM);
	}

      // allocate memory for time series and its FFT
#ifdef BRP_FFT_INPLACE
      // note that fft array is always at least as long as input array, so no problem here 
      // sharing both for inplace transform
      // BUT we have to take care explicitly about the zero-padding, as fft wasn't allocated with calloc.
      time_series = (float *) fft;
      for(i=n_unpadded; i < data_head.nsamples; i++) {
        time_series[i]=0.0f;
      }
#else      
      time_series = (float *) calloc(data_head.nsamples, sizeof(float));
      
      if(time_series == NULL)
      {
        logMessage(error, true, "Couldn't allocate %d bytes of memory for time series.\n", data_head.nsamples*sizeof(float));
        return(RADPUL_EMEM);
      }
#endif      


      // create zero-padded time series
      for(i = 0; i < n_unpadded; i++)
	time_series[i] = t_series_dd.host_ptr[i];

      // get seed for random number generator from the dedispersed time series itself
      seed = *((int32_t*)t_series_dd.host_ptr);
      logMessage(info, true, "Seed for random number generator is %d.\n", seed);
      
      // to conserve max working set size, free buffer here and reallocate it later when needed.
      free(t_series_dd.host_ptr);

      // compute FFT of the time series
      fft_plan = fftwf_plan_dft_r2c_1d(data_head.nsamples, time_series, fft, FFTW_ESTIMATE);
      fftwf_execute(fft_plan);
      fftwf_destroy_plan(fft_plan);
      
      
      // allocate memory for the array containing the periodogramm
      powerspectrum = (float *) calloc(fft_size, sizeof(float));
      if(powerspectrum == NULL)
	{
	  logMessage(error, true, "Couldn't allocate %d bytes of memory for power spectrum.\n", fft_size*sizeof(float));
	  return(RADPUL_EMEM);
	}

      // get the corresponding periodogramm, ignore DC element
      for(i = 1; i < fft_size; i++)
	powerspectrum[i] = gsl_pow_2(fft[i][0]) + gsl_pow_2(fft[i][1]);

      // size of the running median array
      white_size = fft_size - uvar.window + 1;

      // allocate memory for the running median array
      running_median = (float *) calloc(white_size, sizeof(float));
      if(running_median == NULL)
	{
	  logMessage(error, true, "Couldn't allocate %d bytes of memory for running median array.\n", white_size*sizeof(float));
	  return(RADPUL_EMEM);
	}

      // compute running median and store in running_median[]
      rngmed(powerspectrum, fft_size, uvar.window, running_median);

      // clean up
      free(powerspectrum);

      // periodogramm distribution of Gaussian should have
      // median M_LN2 and mean of 1, scale the amplitudes
      // so that the median is M_LN2
      // ATTENTION: don't make window size too small, will bias
      //            estimation of median
      // [0 ................ window_2 ................ white_size - 1 + window_2 ................ fft_size - 1]
      for(i = 0; i < white_size; i++)
	{
	  float factor = sqrt(M_LN2/running_median[i]);
	  fft[i + window_2][0] *= factor;
	  fft[i + window_2][1] *= factor;
	}

      // clean up
      free(running_median);


       /*------------------
	 ZAPPING KNOWN RFIS
	 ------------------*/

      logMessage(debug, true, "Start zapping known radio frequency interferences.\n");

      // open zaplist file
      zaplist = fopen(uvar.zaplistfile, "r");
      if(NULL == zaplist)
	{
	  logMessage(error, true, "Couldn't open zaplist file: %s (%s)\n", uvar.zaplistfile, strerror(errno));
	  return(RADPUL_EFILE);
	}

      // setup GSL random number generation
      r = gsl_rng_alloc(gsl_rng_taus2);
      gsl_rng_set(r, seed);

      while( (NULL != fgets(line, FN_LENGTH, zaplist) && !feof(zaplist)) )
	{
	  // var declarations
	  double t_obs = data_head.nsamples*data_head.tsample*MICROSEC;
	  unsigned int idx;
	  unsigned int idx_min;
	  unsigned int idx_max;
	  double fmin, fmax;

	  line_counter++;

	  // check if line could be read completely
	  if(sscanf(line, "%lg %lg", &fmin, &fmax) != 2)
	    {
	      logMessage(error, true, "Couldn't read complete line no. %d from zaplist file %s.\n", line_counter, uvar.zaplistfile);
	      return(RADPUL_EIO);
	    }

	  // get frequency range bin numbers
	  idx_min = (unsigned int)(fmin*t_obs + 0.5);
	  idx_max = (unsigned int)(fmax*t_obs + 0.5);

	  for(idx = idx_min; idx <= idx_max; idx++)
	    {
	      // fill FFT bins with Gaussian noise of sigma = sqrt(0.5)*sqrt(uvar.padding) = M_SQRT1_2*sqrt(uvar.padding)
	      double sigma = M_SQRT1_2*sqrt(uvar.padding);
	      fft[idx][0] = gsl_ran_gaussian_ziggurat(r, sigma);
	      fft[idx][1] = gsl_ran_gaussian_ziggurat(r, sigma);
	    }

	}

      // close zaplist file
      if(fclose(zaplist))
	{
	  logMessage(error, true, "Couldn't close zaplist file: %s (%s)\n", "/Users/benni/Desktop/zaplist_test.txt", strerror(errno));
	  return(RADPUL_EIO);
	}

      // free random number generator resources
      gsl_rng_free(r);

      logMessage(debug, true, "Zapped known radio frequency interferences.\n");

      // set the amplitudes not covered by running median to zero
      for(i = 0; i < window_2; i++)
	{
	  fft[i][0] = 0.0;
	  fft[i][1] = 0.0;
	  fft[fft_size - i - 1][0] = 0.0;
	  fft[fft_size - i - 1][1] = 0.0;
	}

      // back transformation to time domain
      fft_plan = fftwf_plan_dft_c2r_1d(data_head.nsamples, fft, time_series, FFTW_ESTIMATE);
      fftwf_execute(fft_plan);
      fftwf_destroy_plan(fft_plan);



      // allocate memory for uncompressed dedispersed time series
      logMessage(debug, true, "Allocating memory for dedispersed,whitenend and zapped uncompressed time series.\n");
      t_series_dd.host_ptr = (float *) calloc(n_unpadded, sizeof(float));

      
      if(t_series_dd.host_ptr == NULL)
      {
        logMessage(error, true, "Couldn't allocate %d bytes of memory for dedispersed uncompressed  time series (2).\n", n_unpadded*sizeof(float));
        return(RADPUL_EMEM);
      }
      
      
      // renormalize time series and copy only the values
      // that were initially nonzero
      norm_factor = 1.0/sqrt((float)data_head.nsamples);
      for(i = 0; i < n_unpadded; i++)
	t_series_dd.host_ptr[i] = norm_factor*time_series[i];

      
      // clean up
      fftwf_free(fft);
      
#ifndef BRP_FFT_INPLACE      
      // clean up
      free(time_series);
#endif
    }  // end of if(uvar.white == 1)


  /*----------------------------------------------------------------------------------
    MAIN PART OF THE CODE: RESAMPLING, FFT, HARMONIC SUMMING, CANDIDATE IDENTIFICATION
    ----------------------------------------------------------------------------------*/

  // observation time in seconds
  t_obs = data_head.nsamples*data_head.tsample*MICROSEC;
  dt = data_head.tsample*MICROSEC;
  step_inv = 1.0 / dt; // inverse of a time sample

  // if input is N real numbers, output has N/2 + 1 non-redundant entries
  fft_size = (unsigned int)(data_head.nsamples*0.5 + 0.5) + 1;

  // half of the window size
  window_2 = (unsigned int)(uvar.window*0.5 + 0.5);
  // frequency bin of the highest fundamental frequency searched for
  fundamental_idx_hi = (unsigned int)GSL_MIN_INT(fft_size - window_2, (int)(uvar.f0*t_obs + 0.5));
  // frequeny bin of the highes harmonic frequency searched for
  harmonic_idx_hi = (unsigned int)GSL_MIN_INT(fft_size - window_2, (int)(16.0*uvar.f0*t_obs + 0.5));

  // sanity check
  if(fft_size < uvar.window)
    {
      logMessage(error, true, "Running median window (%d bins) is too wide for data set (%d bins)!\n", uvar.window, fft_size);
      return(RADPUL_EVAL);
    }

  // initialize sin/cos lookup tables (required during resampling)
  sincosLUTInitialize(&sinLUTsamples, &cosLUTsamples);

  // prepare resampling
  RESAMP_PARAMS params;
  params.nsamples = data_head.nsamples;
  params.nsamples_unpadded = n_unpadded;
  params.fft_size = fft_size;


#if defined(BOINCIFIED) && (defined(USE_CUDA) || defined(USE_OPENCL))
  boinc_begin_critical_section();
  logMessage(debug, true, "Entered critical section: CUDA/OpenCL setup phase\n");
#endif

  result = set_up_resampling(t_series_dd, &t_series_resamp, &params, sinLUTsamples, cosLUTsamples);
  if (result != 0) return result;

#if defined(USE_CUDA) && !defined(NDEBUG)
    logMessage(debug, true, "CUDA global memory status (resampling set up):\n");
    printDeviceGlobalMemStatus(debug, true);
#endif

  // prepare FFT and powerspectrum
  result = set_up_fft(t_series_resamp, &powerspectrum, data_head.nsamples, fft_size);
  if (result != 0) return result;

#if defined(USE_CUDA) && !defined(NDEBUG)
    logMessage(debug, true, "CUDA global memory status (FFT/powerspectrum set up):\n");
    printDeviceGlobalMemStatus(debug, true);
#endif

  // prepare harmonic summing
  result = set_up_harmonic_summing(sumspec,dirty,&nr_dirty_pages,fundamental_idx_hi,harmonic_idx_hi);
  if (result != 0) return result;

#if defined(USE_CUDA) && !defined(NDEBUG)
    logMessage(debug, true, "CUDA global memory status (harmonic summing set up):\n");
    printDeviceGlobalMemStatus(debug, true);
#endif

#if defined(BOINCIFIED) && (defined(USE_CUDA) || defined(USE_OPENCL))
  boinc_end_critical_section();
  logMessage(debug, true, "Left critical section: CUDA/OpenCL setup phase\n");
#endif

  // if in debug mode, drop information about thresholds
  if(uvar.debug)
    {
      float prob = 1.0 - pow(1.0 - uvar.fA, 1.0/fft_size);
      logMessage(info, true, "Derived global search parameters:\n");
      logMessage(info, false, "f_A probability = %g\n", uvar.fA);
      logMessage(info, false, "single bin prob(P_noise > P_thr) = %g\n",prob);
      logMessage(info, false, "thr1 = %g\n", 0.5*gsl_cdf_chisq_Qinv(prob, 2.0));
      logMessage(info, false, "thr2 = %g\n", 0.5*gsl_cdf_chisq_Qinv(prob, 4.0));
      logMessage(info, false, "thr4 = %g\n", 0.5*gsl_cdf_chisq_Qinv(prob, 8.0));
      logMessage(info, false, "thr8 = %g\n", 0.5*gsl_cdf_chisq_Qinv(prob, 16.0));
      logMessage(info, false, "thr16 = %g\n", 0.5*gsl_cdf_chisq_Qinv(prob, 32.0));
    }


  /*--------------------------------
    MAIN LOOP OVER THE TEMPLATE BANK
    --------------------------------*/

#if defined(USE_CUDA)
    logMessage(info, true, "CUDA global memory status (GPU setup complete):\n");
    printDeviceGlobalMemStatus(info, true);
#endif

  dirty_page_count=0;

  while( (NULL != fgets(line, FN_LENGTH, templatebank) && !feof(templatebank)) )
    {
      // template values
      double P_tmp, tau_tmp, Psi0_tmp;
      float P;                                 // orbital period of the binary in seconds
      float Psi0;                              // initial orbital phase
      float tau;                               // lighttravel time for projected binary semi-major axis

      // variables for resampling
      float Omega;                             // Omega = 2*M_PI/P [rad/s] (angular orbital velocity)
      float S0;

      // variables for FFT, powerspectrum and the harmonic summing
      unsigned int harm_idx;                    // log2 of the number of harmonics to convert those to linear scale
      float norm_factor;                       // normalization factor


      // variables for the screensaver
      unsigned char *binned_spectrum;           // char array for downsampled power spectrum

      // check if line could be read completely
      if(sscanf(line, c_template_scan_format, &P_tmp, &tau_tmp, &Psi0_tmp) != 3)
	{
	  logMessage(error, true, "Couldn't read complete line no. %d from templatebank.\n", template_counter);
	  return(RADPUL_EFILE);
	}

      // cast template parameters into floats
      P = (float)P_tmp;
      tau = (float)tau_tmp;
      Psi0 = (float)Psi0_tmp;

      // store template information in screensaver struct
      search_params_tmp.orbital_radius = tau;
      search_params_tmp.orbital_period = P;
      search_params_tmp.orbital_phase = Psi0;

      // get angular frequency
      Omega = 2.0*M_PI/P;

#if defined(USE_CUDA) && !defined(NDEBUG)
      logMessage(debug, true, "CUDA global memory status (template iteration):\n");
      printDeviceGlobalMemStatus(debug, true);
#endif

      /*-----------------------------
	RESAMPLING OF THE TIME SERIES
	-----------------------------*/

      // first: compute zero time offset
      S0 = tau * sin(Psi0) * step_inv;

      // now: generate time series in modulated time
      params.tau = tau;
      params.Omega = Omega;
      params.Psi0 = Psi0;
      params.dt = dt;
      params.step_inv = step_inv;
      params.S0 = S0;

#if defined(BOINCIFIED) && (defined(USE_CUDA) || defined(USE_OPENCL))
      boinc_begin_critical_section();
      logMessage(debug, true, "Entered critical section: CUDA/OpenCL template iteration\n");
#endif

      result = run_resampling(t_series_dd, t_series_resamp, &params);
      if(result != 0) return result;


      /*---------------------------------
	FFT AND POWERSPECTRUM COMPUTATION
	---------------------------------*/

      // Compute powerspectrum of resampled time series
      logMessage(debug, true, "Computing powerspectrum.\n");
      norm_factor = 1.0/data_head.nsamples;   // normalization factor

      result = run_fft(t_series_resamp, powerspectrum, data_head.nsamples, fft_size, norm_factor);
      if (result != 0) return result;


      /*---------------------------------------------
	HARMONIC SUMMING AND CANDIDATE IDENTIFICATION
	---------------------------------------------*/

      logMessage(debug, true, "Harmonic summing and candidate identification.\n");
      // calculate thresholds for power

      for(harm_idx = 0; harm_idx < 5 ; harm_idx++) {
        // get number of harmonics from logarithmic scale harm_idx
        unsigned int N_h = (int)(pow(2.0, harm_idx) + 0.5);

        // uvar.fA is the false alarm rate for the complete FFT and
        // prob is the false alarm rate for a single bin in the FFT
        float prob = 1.0 - pow(1.0 - uvar.fA, 1.0/fft_size);

        int last_cand = (harm_idx + 1)*N_CAND_5 - 1;
        float power =candidates_all[last_cand].power;

        // 2*power is distributed in a chi-squared of 2n d.o.f. in sum over n harmonics,
        // corresponding thresholds on power
        thrA[harm_idx]  = fmaxf(power,0.5*gsl_cdf_chisq_Qinv(prob, 2.0*N_h));
      }

	/* if we want to see the summed spectrum in the screensaver, we have to set the threshold
	   for the highests harmonics to 0.0 . But maybe there is a way to make a screensaver
	   display from 1st harm powerspectrum which is not thresholded */
#if 0
thrA[4] = 0.0;
#endif


      result=run_harmonic_summing(sumspec, dirty, nr_dirty_pages, powerspectrum,window_2,fundamental_idx_hi,harmonic_idx_hi,thrA);
      if(result != 0) return result;

#if defined(BOINCIFIED) && (defined(USE_CUDA) || defined(USE_OPENCL))
      boinc_end_critical_section();
      logMessage(debug, true, "Left critical section: CUDA/OpenCL template iteration\n");
#endif

      // allocate memory for downsampled power spectrum
      binned_spectrum = (unsigned char *) calloc(N_BINS_SS, sizeof(unsigned char));
      if(binned_spectrum == NULL)
	{
	  logMessage(error, true, "Couldn't allocate %d bytes of memory for downsampled power spectrum.\n", N_BINS_SS*sizeof(unsigned char));
	  return(RADPUL_EMEM);
	}

      // check for candidates
      // use dirty page array to find those segments of sumspec arrays that have a chance to contribute a candidate
      for(harm_idx = 0; harm_idx < 5; harm_idx++)
	{
	  // indices for first and last candidate (for this number of harmonics) in candidates_all[]
	  int first_cand = harm_idx*N_CAND_5;
	  int last_cand = (harm_idx + 1)*N_CAND_5 - 1;

	  // get number of harmonics from logarithmic scale harm_idx
	  unsigned int N_h = (int)(pow(2.0, harm_idx) + 0.5);

	  // 2*power is distributed in a chi-squared of 2n d.o.f. in sum over n harmonics,
	  // corresponding thresholds on power
	  float thr  = thrA[harm_idx];

	  i=window_2;
	  while(i < fundamental_idx_hi)
	    {
	      int page_idx;
              int i_next_page;

	      // find next "dirty" page in sumspec beginning with page that covers bin i

	      // note: we cannot assume i is a multiple of Page size here,
	      //       for the very first iteration (i = window_2) it might not be

	      for(page_idx = (i>> LOG_PS_PAGE_SIZE) ; (page_idx < nr_dirty_pages) && (dirty[harm_idx][page_idx] ==0); page_idx++, i=page_idx * (1 << LOG_PS_PAGE_SIZE))
	        { /* sic */ ;}

	      // if at the end, break out of while loop
	      if(i >= fundamental_idx_hi) {
		break;
	      }

	      dirty_page_count++;

	      // now look for candidates on a "dirty" page of sumspec, but make sure not to go beyond fundamental_idx_hi
	      i_next_page = GSL_MIN_INT((page_idx+1) << LOG_PS_PAGE_SIZE, fundamental_idx_hi);
	      for(    ; i < i_next_page ;i++) {
	       float power = sumspec[harm_idx][i];

	      // store and sort top candidates for 2^harm_idx summed harmonics
	      if(power > thr && power > candidates_all[last_cand].power)
		{
		  int idx;
		  int store_idx = last_cand; // default place for storage in candidates_all is at the end
		  // check whether frequency bin already in candidates_all and
		  // whether candidate already stored at this frequency has less power
		  for(idx = first_cand; idx <= last_cand; idx++)
		    {
		      if(candidates_all[idx].f0 == i)
			{
			  store_idx = (candidates_all[idx].power < power) ? idx : -1;
			  break; // break out of loop if candidate at same frequency was found
			}
		    }

		  // Only store and resort if either no candidate at the same freq was found
		  // or if the candidate at the same freq has less power than the new one.
		  if(store_idx >= first_cand)
		    {
		      candidates_all[store_idx].f0 = i;
		      candidates_all[store_idx].P_b = P;
		      candidates_all[store_idx].tau = tau;
		      candidates_all[store_idx].Psi = Psi0;
		      candidates_all[store_idx].power = power;
		      candidates_all[store_idx].n_harm = N_h;

		      // most efficient way would be heap, at the moment for simplicity use qsort
		      qsort(candidates_all + first_cand, N_CAND_5, sizeof(CP_cand), compare_structs_by_P);
		    }

		}

	      // if using the sum of 4 harmonics, update screen saver as well
	      if(harm_idx == 2)
		{
		  float max_power_screensaver = 100.0;                              // maximum power displayed without truncation in the screensaver
		  float powerscale = max_power_screensaver/255.0;                   // power axis scale factor in the computation of downsampled spectrum
		  float stepscale = (float)N_BINS_SS/(float)fundamental_idx_hi;  // conversion factor between full spectrum and SS spectrum
		  int bin_ss;

		  bin_ss = (int)(stepscale*i);
		  if(sumspec[2][i] > powerscale*(float)binned_spectrum[bin_ss])
		    binned_spectrum[bin_ss] = (unsigned char)(GSL_MIN_DBL(sumspec[2][i]/powerscale, 255.0)); // avoid numbers > 255
		}
	     } // end of loop over current page
	    } // end of loop over frequency bins

	} // end of loop over harm_idx



#ifdef BOINCIFIED
      // copy spectrum data to search parameters
      strncpy((char*) search_params_tmp.power_spectrum, (const char*) binned_spectrum, POWERSPECTRUM_BINS);
      // copy search parameters to final destination
      erp_search_info = search_params_tmp;
      // update shared memory area
      erp_update_shmem();
#endif

      // clean up
      free(binned_spectrum);

      template_counter++;
      cp_head.n_template = template_counter;

      logMessage(debug, true, "Template done!\n");

#ifdef BOINCIFIED
      // update work unit fraction done (counter is 0-indexed)
      erp_fraction_done((template_counter + 1.0) / template_total_amount);

      // only commit checkpoints if BOINC says so
      if (boinc_time_to_checkpoint()) {
#endif
	// at the end of each template searched, write checkpoint file
	logMessage(debug, false, "Committing checkpoint.\n");
	result = set_checkpoint(&uvar, &cp_head, candidates_all);
	if(result != 0) return(result);

#ifdef BOINCIFIED
	  logMessage(info, true, "Checkpoint committed!\n");
	  boinc_checkpoint_completed();
      }

      // time to check what's going on
      boinc_get_status(&boinc_status);

      // we are about to be killed or we lost contact, so prepare to exit prematurely
      if (boinc_status.quit_request || boinc_status.abort_request || boinc_status.no_heartbeat) {
          break;
      }
#endif
    }// end: loop over templates

  // clean up
  free(t_series_dd.host_ptr);

  result = tear_down_resampling(t_series_resamp);
  if (result != 0) return result;

  result = tear_down_fft(powerspectrum);
  if (result != 0) return result;

  result = tear_down_harmonic_summing(sumspec,dirty);
  if (result != 0) return result;

#if defined(USE_CUDA) && !defined(NDEBUG)
    logMessage(debug, true, "CUDA global memory status (all torn down):\n");
    printDeviceGlobalMemStatus(debug, true);
#endif

#if defined(BOINCIFIED) && (defined(USE_CUDA) || defined(USE_OPENCL))
  boinc_begin_critical_section();
  logMessage(debug, true, "Entered critical section: CUDA/OpenCL shutdown\n");
#endif

#if defined(USE_CUDA)
  // free CUDA device
  shutdown_cuda();
#elif defined(USE_OPENCL)
  // free OpenCL device
  shutdown_ocl();
#endif

#if defined(BOINCIFIED) && (defined(USE_CUDA) || defined(USE_OPENCL))
  boinc_end_critical_section();
  logMessage(debug, true, "Left critical section: CUDA/OpenCL shutdown\n");
#endif

  // close template bank file
  if(fclose(templatebank))
    {
      logMessage(error, true, "Couldn't close template bank file: %s (%s)\n", uvar.templatebank, strerror(errno));
      return(RADPUL_EIO);
    }

#ifdef BOINCIFIED
  // we are about to be killed or we lost contact, so exit prematurely
  if (boinc_status.quit_request || boinc_status.abort_request || boinc_status.no_heartbeat) {
      logMessage(warn, true, "BOINC wants us to quit prematurely or we lost contact! Exiting...\n");
      exit(0);
  }
#endif

  // set final checkpoint to avoid recomputation (since last checkpoint) on application restart
  logMessage(debug, true, "Search done!\n");
  logMessage(debug, false, "Committing final checkpoint.\n");
  result = set_checkpoint(&uvar, &cp_head, candidates_all);
  if(result != 0) return(result);

  // compute the -log10() of the inverse false alarm rate for all candidates
  for(i = 0; i < N_CAND; i++)
    {
      // 2*power is distributed in a chi-squared of 2n d.o.f. in sum over n harmonics
      // sigma in chi-squared of 2n d.o.f. is sqrt(4n) => sigma_n = sqrt(n)
      float sigma1 = 1.0;
      float sigma2 = sqrt(2.0);
      float sigma4 = 2.0;
      float sigma8 = sqrt(8.0);
      float sigma16 = 4.0;

      double fA;

      // compute -log10 of the inverse false alarm rate and convert power into units of sigma
      if(candidates_all[i].n_harm == 1)
	{
	  fA = gsl_cdf_chisq_Q(2.0*candidates_all[i].power, 2.0);
	  // check if fA is non-zero, because otherwise the log10() will return inf
	  candidates_all[i].fA = (fA > 0.0) ? -log10(fA) : 320.0;
	  candidates_all[i].power /= sigma1;
	}
      else if(candidates_all[i].n_harm == 2)
	{
	  fA = gsl_cdf_chisq_Q(2.0*candidates_all[i].power, 4.0);
	  // check if fA is non-zero, because otherwise the log10() will return inf
	  candidates_all[i].fA = (fA > 0.0) ? -log10(fA) : 320.0;
	  candidates_all[i].power /= sigma2;
	}
      else if(candidates_all[i].n_harm == 4)
	{
	  fA = gsl_cdf_chisq_Q(2.0*candidates_all[i].power, 8.0);
	  // check if fA is non-zero, because otherwise the log10() will return inf
	  candidates_all[i].fA = (fA > 0.0) ? -log10(fA) : 320.0;
	  candidates_all[i].power /= sigma4;
	}
      else if(candidates_all[i].n_harm == 8)
	{
	  fA = gsl_cdf_chisq_Q(2.0*candidates_all[i].power, 16.0);
	  // check if fA is non-zero, because otherwise the log10() will return inf
	  candidates_all[i].fA = (fA > 0.0) ? -log10(fA) : 320.0;
	  candidates_all[i].power /= sigma8;
	}
      else if(candidates_all[i].n_harm == 16)
	{
	  fA = gsl_cdf_chisq_Q(2.0*candidates_all[i].power, 32.0);
	  // check if fA is non-zero, because otherwise the log10() will return inf
	  candidates_all[i].fA = (fA > 0.0) ? -log10(fA) : 320.0;
	  candidates_all[i].power /= sigma16;
	}
      else // set entries without candidates (which should not exist) to impossible low value
	candidates_all[i].fA = -10.0;
    }

  // sort candidate array by -log10(ifa)
  qsort(candidates_all, N_CAND, sizeof(CP_cand), compare_structs_by_ifa);

  // if all templates are done, write output file
  logMessage(debug, true, "Writing all candidates to output file.\n");

  // use temporary output file for atomic transaction
  output = fopen(uvar.outputfile_tmp, "w");
  if(output == NULL)
    {
      logMessage(error, true, "Couldn't open temporary output file: %s (%s)\n", uvar.outputfile_tmp, strerror(errno));
      return(RADPUL_EIO);
    }

#ifdef BOINCIFIED

#define LEN_SHEBANG 40

  int   userId   = 0;
  char *userName = NULL;
  int   hostId   = 0;
  char *hostCpId = NULL;
  char *execName = argv[0];
  char erp_git_version[41];
  char boinc_rev[41];
  char *pathSepExecname;

  strncpy(erp_git_version,ERP_GIT_VERSION,LEN_SHEBANG);
  erp_git_version[LEN_SHEBANG]='\0';
  strncpy(boinc_rev,SVN_VERSION,LEN_SHEBANG);
  boinc_rev[LEN_SHEBANG]='\0';

  while ((pathSepExecname = strpbrk(execName, "\\/"))) {
      execName=pathSepExecname+1;
  }

  // parse and forward BOINC's application information
  APP_INIT_DATA appInitData;
  if(boinc_parse_init_data_file() == 0) {
      boinc_get_init_data(appInitData);
      userId = appInitData.userid;
      if(strlen(appInitData.user_name) != 0) {
          userName = appInitData.user_name;
      }
      hostId = appInitData.hostid;
      if(strlen(appInitData.host_info.host_cpid) != 0) {
          hostCpId = appInitData.host_info.host_cpid;
      }
  }
  else {
      logMessage(warn, true, "User/host details unavailable...\n");
  }
  // retrieve current time
  timeValue = time(0);
  if(timeValue != (time_t)-1) {
      // convert time
      timeUTC = gmtime(&timeValue);
      if(timeUTC != NULL) {
          strftime(resultTimeISO, TIME_LENGTH, TIME_FORMAT, timeUTC);
      }
  }
  // write header
  if(!fprintf(output, "%% User: %i (%s)\n%% Host: %i (%s)\n%% Date: %s\n%% Exec: %s\n%% ERP git id: %s\n%% BOINC rev.: %s\n\n",
              userId, userName?userName:"unknown",
              hostId, hostCpId?hostCpId:"unknown",
              resultTimeISO,
              execName?execName:"unknown",
              erp_git_version?erp_git_version:"unknown",
              boinc_rev?boinc_rev:"unknown"))
  {
      logMessage(error, true, "Couldn't write header to temporary output file: %s (%s)\n", uvar.outputfile_tmp, strerror(errno));
      return(RADPUL_EIO);
  }

#endif

  // write all candidates with power > 0.0 to file
  int counter = 0;
  while(counter < N_CAND_5 && candidates_all[0].fA > 0.0)
    {
      unsigned int j;

      // observation time in seconds and frequency resolution
      double t_obs = data_head.nsamples*data_head.tsample*MICROSEC;
      double res_factor = 1.0/t_obs;

      if(fprintf(output, "%6.12f %6.12f %6.12f %6.12f %g %g %d\n",	\
		 candidates_all[0].f0*res_factor, candidates_all[0].P_b, candidates_all[0].tau, candidates_all[0].Psi, \
		 candidates_all[0].power, candidates_all[0].fA, candidates_all[0].n_harm) < 0)
	{
	  logMessage(error, true, "Couldn't write candidate data to temporary output file: %s (%s)\n", uvar.outputfile_tmp, strerror(errno));
	  return(RADPUL_EIO);
	}

      // increase counter of stored candidates
      counter++;

      // Loop over all candidates (including the one stored above) and reset to negative values of fA
      // if at same frequency so that resorting puts them at the end of the array.
      // This step ensures maximization over N_h. If there's a less significant candidate
      // at the same frequency bin (and with different N_h) it's dismissed in this step.
      for(j = 0; j < N_CAND; j++)
	{
	  if(candidates_all[j].f0 == candidates_all[0].f0)
	    candidates_all[j].fA = -10.0;
	}

      // resort
      qsort(candidates_all, N_CAND, sizeof(CP_cand), compare_structs_by_ifa);

    }// end of while loop

  // write %DONE% marker
  if(fprintf(output, "%%DONE%%\n") < strlen("%DONE%\n"))
    {
      logMessage(error, true, "Couldn't write %%DONE%% marker to temporary output file: %s (%s)\n", uvar.outputfile_tmp, strerror(errno));
      return(RADPUL_EIO);
    }

  // close temporary output file
  if(fclose(output))
    {
      logMessage(error, true, "Couldn't close temporary output file: %s (%s)\n", uvar.outputfile_tmp, strerror(errno));
      return(RADPUL_EIO);
    }

  // rename temp to final output file (atomic)
  if(rename(uvar.outputfile_tmp, uvar.outputfile))
    {
      logMessage(error, true, "Couldn't rename temporary output file (%s) to final output file: %s (%s)\n", uvar.outputfile_tmp, uvar.outputfile, strerror(errno));
      return(RADPUL_EFILE);
    }

  // clean up
  free(candidates_all);

  logMessage(info, true, "Statistics: count dirty SumSpec pages %u (not checkpointed), Page Size %d, fundamental_idx_hi-window_2: %u\n",dirty_page_count,1 << LOG_PS_PAGE_SIZE,fundamental_idx_hi-window_2);
  logMessage(info, true, "Data processing finished successfully!\n");

  return(0);

}// end: MAIN()


// Comparison function for qsort().
int compare_structs_by_P(const void * const ptr1, const void * const ptr2)
{
  const CP_cand *cand1 = (const CP_cand *) ptr1;
  const CP_cand *cand2 = (const CP_cand *) ptr2;

  if(cand1->power < cand2->power)
    return 1;
  else if(cand1->power > cand2->power)
    return -1;
  else
    return 0;
}

// Comparison function for qsort() to sort by false alarm rate.
int compare_structs_by_ifa(const void * const ptr1, const void * const ptr2)
{
  const CP_cand *cand1 = (const CP_cand *) ptr1;
  const CP_cand *cand2 = (const CP_cand *) ptr2;

  if(cand1->fA < cand2->fA)
    return 1;
  else if(cand1->fA > cand2->fA)
    return -1;
  else
    {
      // if candidates agree in fA, go for higher power
      if(cand1->power < cand2->power)
	return 1;
      else if(cand1->power > cand2->power)
	return -1;
      else
	{
	  // if candidates agree in fA and power, sort by f0
	  if(cand1->f0 < cand2->f0)
	    return 1;
	  else if(cand1->f0 > cand2->f0)
	    return -1;
	  else
	    return 0; // (this should actually never happen)
	}
    }
}

// Write checkpoint
int set_checkpoint(const User_Variables * const uvar,
                   const CP_Header * const cp_head,
                   const CP_cand * const candidates)
{
  FILE *checkpoint = NULL;


  // use temporary checkpoint file for atomic transaction
  checkpoint = fopen(uvar->checkpointfile_tmp, "wb");
  if(checkpoint == NULL) {
    logMessage(error, true, "Couldn't open temporary checkpoint file: %s (%s).\n", uvar->checkpointfile_tmp, strerror(errno));
    return(RADPUL_EIO);
  }

  // write header to temporary checkpoint file
  if(fwrite(cp_head, sizeof(CP_Header), 1, checkpoint) != 1) {
    logMessage(error, true, "Couldn't write header to temporary checkpoint file: %s (%s).\n", uvar->checkpointfile_tmp, strerror(errno));
    return(RADPUL_EIO);
  }

  // write data (candidates) to temporary checkpoint file
  if(fwrite(candidates, sizeof(CP_cand), N_CAND, checkpoint) != N_CAND) {
    logMessage(error, true, "Couldn't write candidate data to temporary checkpoint file: %s (%s).\n", uvar->checkpointfile_tmp, strerror(errno));
    return(RADPUL_EIO);
  }

  // close checkpoint file
  if(fclose(checkpoint)) {
    logMessage(error, true, "Couldn't close temporary checkpoint file: %s (%s).\n", uvar->checkpointfile_tmp, strerror(errno));
    return(RADPUL_EIO);
  }

  // rename temp to final checkpoint file (atomic)
  if(rename(uvar->checkpointfile_tmp, uvar->checkpointfile)) {
    logMessage(error, true, "Couldn't rename temporary checkpoint file (%s) to final checkpoint file: %s (%s).\n", uvar->checkpointfile_tmp, uvar->checkpointfile, strerror(errno));
    return(RADPUL_EFILE);
  }

  // success
  return(0);
}


#ifndef BOINCIFIED
/*++++++++++++++++++++++++++++++++++++++++
+    main program standalone wrapper     +
++++++++++++++++++++++++++++++++++++++++*/
int main (int argc, char *argv[])
{
	return MAIN(argc, argv);
}
#endif
