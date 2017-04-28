/***************************************************************************
 *   Copyright (C) 2008 by Oliver Bock, Bernd Machenschalk                 *
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

/* BOINC includes - need to be before the #defines in boinc_wrapper.h */
#include "boinc_api.h"
#include "diagnostics.h"
#include "svn_version.h"

/* probably already included by previous headers, but make sure they are included */
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <zlib.h>

/* needed to define backtrace() which is glibc specific */
/* get REG_EIP from ucontext.h, see http://www.linuxjournal.com/article/6391 */
#ifdef __GLIBC__
#include <signal.h>
#include <execinfo.h>
#include <ucontext.h>
#endif

/* re-route sleep for win32 */
#ifdef _WIN32
#include <windows.h>
#define sleep Sleep
#endif

#ifdef __MINGW32__
#include "exchndl.h"
#endif

/* headers of our own code */
#include "demod_binary.h"
#include "erp_boinc_wrapper.h"
#include "erp_boinc_ipc.h"
#include "erp_getopt.h"
#include "erp_utilities.h"
#include "erp_execinfo_plus.h"
#include "erp_git_version.h"

#define MAX_PATH_LENGTH 512
#define EINSTEINRADIO_EMEM 1
#define EINSTEINRADIO_EFILE 2
#define EINSTEINRADIO_EOPT 4
#define EINSTEINRADIO_EXIT 8
#define EINSTEINRADIO_ESIG 16

using namespace std;

static int global_argc;
static char **global_argv;
static unsigned int pass=0, passes=0;

typedef uint16_t fpuw_t;

/* constants in FPU status word and control word mask */
#define FPU_STATUS_INVALID      (1<<0)
#define FPU_STATUS_DENORMALIZED (1<<1)
#define FPU_STATUS_ZERO_DIVIDE  (1<<2)
#define FPU_STATUS_OVERFLOW     (1<<3)
#define FPU_STATUS_UNDERFLOW    (1<<4)
#define FPU_STATUS_PRECISION    (1<<5)
#define FPU_STATUS_STACK_FAULT  (1<<6)
#define FPU_STATUS_ERROR_SUMM   (1<<7)
#define FPU_STATUS_COND_0       (1<<8)
#define FPU_STATUS_COND_1       (1<<9)
#define FPU_STATUS_COND_2       (1<<10)
#define FPU_STATUS_COND_3       (1<<14)

/* write the FPU status flags / exception mask bits to stderr */
#define PRINT_FPU_EXCEPTION_MASK(fpstat) \
        if (fpstat & FPU_STATUS_PRECISION) \
        logMessage(error, false, "FPU exception: PRECISION\n"); \
        if (fpstat & FPU_STATUS_UNDERFLOW) \
        logMessage(error, false, "FPU exception: UNDERFLOW\n"); \
        if (fpstat & FPU_STATUS_OVERFLOW) \
        logMessage(error, false, "FPU exception: OVERFLOW\n"); \
        if (fpstat & FPU_STATUS_ZERO_DIVIDE) \
        logMessage(error, false, "FPU exception: ZERO_DIVIDE\n"); \
        if (fpstat & FPU_STATUS_DENORMALIZED) \
        logMessage(error, false, "FPU exception: DENORMALIZED\n"); \
        if (fpstat & FPU_STATUS_INVALID) \
        logMessage(error, false, "FPU exception: INVALID\n")
#define PRINT_FPU_STATUS_FLAGS(fpstat) \
        if (fpstat & FPU_STATUS_COND_3) \
        logMessage(error, false, "FPU status: COND_3\n"); \
        if (fpstat & FPU_STATUS_COND_2) \
        logMessage(error, false, "FPU status: COND_2\n"); \
        if (fpstat & FPU_STATUS_COND_1) \
        logMessage(error, false, "FPU status: COND_1\n"); \
        if (fpstat & FPU_STATUS_COND_0) \
        logMessage(error, false, "FPU status: COND_0\n"); \
        if (fpstat & FPU_STATUS_ERROR_SUMM) \
        logMessage(error, false, "FPU status: ERR_SUMM\n"); \
        if (fpstat & FPU_STATUS_STACK_FAULT) \
        logMessage(error, false, "FPU status: STACK_FAULT\n"); \
        PRINT_FPU_EXCEPTION_MASK(fpstat)


/* Signal handler */

#ifdef __GLIBC__
static void sighandler(int sig, siginfo_t *info, void *secret)
#else
static void sighandler(int sig)
#endif
{
    static int killcounter = 0;

#ifdef __GLIBC__
    /* for glibc stacktrace */
    static void *stackframes[64];
    static size_t nostackframes;
    static char **backtracesymbols = NULL;
    ucontext_t *uc = (ucontext_t *)secret;
#endif

    /* lets start by ignoring ANY further occurrences of this signal
         (hopefully just in THIS thread, if truly implementing POSIX threads */
    logMessage(error, true, "\nApplication caught signal %d.\n", sig);

    /* ignore TERM interrupts once  */
    if (sig == SIGTERM || sig == SIGINT) {
        killcounter++;
        if (killcounter >= 4) {
            logMessage(warn, true, "Got 4th kill-signal, guess you mean it. Exiting now!\n\n");
            boinc_finish(EINSTEINRADIO_EXIT);
        }
        else {
            return;
        }
    }

#ifdef __GLIBC__

#ifdef __i386__
    /* in case of a floating-point exception write out the FPU status */
    if ( sig == SIGFPE ) {
        fpuw_t fpstat = uc->uc_mcontext.fpregs->sw;
        logMessage(error, false, "\nFPU status word: %lx\n", uc->uc_mcontext.fpregs->sw);
        PRINT_FPU_STATUS_FLAGS(fpstat);
    }
#endif

    /* now get TRUE stacktrace */
    nostackframes = backtrace(stackframes, 64);
    logMessage(error, false, "\nObtained %zd stack frames for this thread.\n", nostackframes);

#ifdef __i386__
    /* overwrite sigaction with caller's address */
    stackframes[1] = (void *) uc->uc_mcontext.gregs[REG_EIP];
#endif

    /* print stacktrace to stderr */

#ifndef __arm__
    logMessage(error, false, "Backtrace:\n");
    backtracesymbols = backtrace_symbols(stackframes, nostackframes);
    if(backtracesymbols != NULL) {
        backtrace_symbols_fd_plus(backtracesymbols, nostackframes, fileno(stderr));
        free(backtracesymbols);
    }
    logMessage(error, false, "End of backtrace\n\n");
#endif 

#endif

    /* sleep a few seconds to let the OTHER thread(s) catch the signal too... */
    sleep(5);
    boinc_finish(sig);
    return;
}

gzFile boinc_gzopen(const char* path, const char* mode) {
  FILE* fp = boinc_fopen(path, mode);
  if(!fp) return NULL;
  return gzdopen(fileno(fp), mode);
}

void erp_fraction_done(double frac) {
  boinc_fraction_done((frac+pass)/passes);
}

static void handle_option(vector<char*>& options, int& option_index, const char* option)
{
    /* add option to forwarded option list */
    options.push_back((char*) calloc(strlen(option) + 1, sizeof(char)));
    if(!options.at(option_index)) {
        logMessage(error, true, "Out of memory\n");
        boinc_finish(EINSTEINRADIO_EMEM);
    }
    strncpy(options.at(option_index), option, strlen(option) + 1);
    option_index++;
}

static void handle_option_value(vector<char*>& options, int& option_index, const char* value)
{
    /* add value to forwarded option list */
    options.push_back((char*) calloc(strlen(value) + 1, sizeof(char)));
    if(!options.at(option_index)) {
        logMessage(error, true, "Out of memory\n");
        boinc_finish(EINSTEINRADIO_EMEM);
    }
    strncpy(options.at(option_index), value, strlen(value) + 1);
    option_index++;
}

static void handle_option_file_value(vector<char*>& options, int& option_index, const char* value)
{
    /* add filename to forwarded option list */
    options.push_back((char*) calloc(MAX_PATH_LENGTH, sizeof(char)));
    if(!options.at(option_index)) {
        logMessage(error, true, "Out of memory\n");
        boinc_finish(EINSTEINRADIO_EMEM);
    }
    /* resolve logical filename */
    //TODO: check error
    boinc_resolve_filename(value, options.at(option_index), MAX_PATH_LENGTH);
    option_index++;
}

static int worker(void)
{
    int argc = global_argc;
    char** argv = global_argv;
    int forward_argc = 0;
    vector<char*> forward_argv(1);
    int input_files = 0;
    vector<char*> input_file;
    int output_files = 0;
    vector<char*> output_file;
    char* checkpoint_file=NULL;

    int result = 0;

    /* keep the program name */
    forward_argv.at(forward_argc) = argv[0];

    /* move to the first option*/
    forward_argc++;

    while(true) {
        /* define known options */
        static struct option long_options[] = {
                { "input-file",         required_argument,  0, 'i' },
                { "template-bank-file", required_argument,  0, 't' },
                { "output-file",        required_argument,  0, 'o' },
                { "checkpoint-file",    required_argument,  0, 'c' },
                { "zaplist-file",       required_argument,  0, 'l' },
                { "f0",                 required_argument,  0, 'f' },
                { "false-alarm",        required_argument,  0, 'A' },
                { "kill-line",          no_argument,        0, 'K' },
                { "padding",            required_argument,  0, 'P' },
                { "whitening",          no_argument,        0, 'W' },
                { "box",                required_argument,  0, 'B' },
                { "device",             required_argument,  0, 'D' },
                { "debug",              no_argument,        0, 'z' },
                { "help",               no_argument,        0, 'h' },
                { "version",            no_argument,        0, 'v' },
                { 0, 0, 0, 0 } };

        /* next option index (after getopt_long) */
        int option_index = 0;

        result = getopt_long(argc, argv, "i:t:o:c:l:f:A:KP:WB:D:zhv", long_options, &option_index);

        /* no more options to parse, exit loop*/
        if (result == -1) {
            break;
        }

        switch (result) {
            case 0:
                /* we don't use any flags */
                logMessage(warn, true, "Unknown option flag encountered!\n");
                break;

            case 'i':
                input_file.push_back(optarg);
                input_files++;
                break;

            case 'o':
                output_file.push_back(optarg);
                output_files++;
                break;

            case 't':
                handle_option(forward_argv, forward_argc, "-t");
                handle_option_file_value(forward_argv, forward_argc, optarg);
                logMessage(debug, true, "Template bank file: '%s'\n", optarg);
                logMessage(debug, false, "Resolved template bank file: '%s'\n", forward_argv.at(forward_argc-1));
                break;

            case 'c':
                checkpoint_file=optarg;
                handle_option(forward_argv, forward_argc, "-c");
                handle_option_file_value(forward_argv, forward_argc, optarg);
                logMessage(debug, true, "Checkpoint file: '%s'\n", optarg);
                logMessage(debug, false, "Resolved checkpoint file: '%s'\n", forward_argv.at(forward_argc-1));
                break;

            case 'l':
                handle_option(forward_argv, forward_argc, "-l");
                handle_option_file_value(forward_argv, forward_argc, optarg);
                logMessage(debug, true, "Zaplist file: '%s'\n", optarg);
                logMessage(debug, false, "Resolved zaplist file: '%s'\n", forward_argv.at(forward_argc-1));
                break;

            case 'f':
                handle_option(forward_argv, forward_argc, "-f");
                handle_option_value(forward_argv, forward_argc, optarg);
                logMessage(debug, true, "Maximum signal frequency [Hz]: '%s'\n", optarg);
                break;

            case 'A':
                handle_option(forward_argv, forward_argc, "-A");
                handle_option_value(forward_argv, forward_argc, optarg);
                logMessage(debug, true, "False alarm probability: '%s'\n", optarg);
                break;

            case 'K':
                handle_option(forward_argv, forward_argc, "-K");
                logMessage(debug, true, "Kill the power line at 60 Hz: on\n");
                break;

            case 'P':
                handle_option(forward_argv, forward_argc, "-P");
                handle_option_value(forward_argv, forward_argc, optarg);
                logMessage(debug, true, "Frequency over-resolution factor: '%s'\n", optarg);
                break;

            case 'W':
                handle_option(forward_argv, forward_argc, "-W");
                logMessage(debug, true, "Power spectrum whitening: on\n");
                break;

            case 'B':
                handle_option(forward_argv, forward_argc, "-B");
                handle_option_value(forward_argv, forward_argc, optarg);
                logMessage(debug, true, "Box width for the running median [frequency bins]: '%s'\n", optarg);
                break;

            case 'D':
                handle_option(forward_argv, forward_argc, "-D");
                handle_option_value(forward_argv, forward_argc, optarg);
                logMessage(debug, true, "Requested GPU device: #%s\n", optarg);
                break;

            case 'z':
                handle_option(forward_argv, forward_argc, "-z");
                logMessage(debug, true, "Debug mode: on\n");
                break;

            case 'h':
                handle_option(forward_argv, forward_argc, "-h");
                logMessage(debug, true, "Show help: on\n");
                break;

            case 'v':
                logMessage(info, true, "Version information:\n");
                logMessage(info, false, "Binary Pulsar Search Revision: %s\n", ERP_GIT_VERSION);
                logMessage(info, false, "BOINC Revision: %s\n", SVN_VERSION);
                return 0;

            case '?':
                /* invalid option (warning already printed) */
                boinc_finish(EINSTEINRADIO_EOPT);
                break;

            default:
                /* we shouldn't get here as we handle all options */
                logMessage(warn, true, "Unhandled option encountered!\n");
                boinc_finish(EINSTEINRADIO_EOPT);
                break;
        }
    }

    /* show warning if we've been passed unknown arguments*/
    if (optind < argc) {
        logMessage(warn, true, "Non-option arguments encountered:\n");
        while (optind < argc) {
            logMessage(warn, false, "%s\n", argv[optind++]);
        }
    }

    /* everything ok */
    result=0;

    /* error if input and output files don't match */
    if (input_files != output_files) {
        logMessage(error, true, "number of input- and output files don't match\n");
        result=1;
    }
    passes = input_files;

    /* set up shared memory segment for data exchange */
    if (erp_setup_shmem()) {
        logMessage(warn, true, "Shared memory setup failed!\n");
    } else {
        logMessage(debug, true, "Shared memory setup completed...\n");
    }

    /*
      process each input file writing to the corresponding output file
      do this only if result==0 (i.e. input_files == output_files)
      do a pass only if the output file doesn't exist yet, else skip that pass
      delete the checkpoint file after each pass, so it doesn't get confused with the next
    */
    for(pass=0; pass<passes && !result; pass++) {
      handle_option(forward_argv, forward_argc, "-i");
      handle_option_file_value(forward_argv, forward_argc, input_file[pass]);
      logMessage(debug, true, "Input file: '%s'\n", input_file[pass]);
      logMessage(debug, false, "Resolved input file: '%s'\n", forward_argv.at(forward_argc-1));

      handle_option(forward_argv, forward_argc, "-o");
      handle_option_file_value(forward_argv, forward_argc, output_file[pass]);
      logMessage(debug, true, "Output file: '%s'\n", output_file[pass]);
      logMessage(debug, false, "Resolved output file: '%s'\n", forward_argv.at(forward_argc-1));

#ifndef NDEBUG
      fprintf(stderr,"command_line: ");
      for(unsigned int i = 0; i < forward_argv.size(); ++i) {
        fprintf(stderr," %s", forward_argv.at(i));
      }
      fprintf(stderr,"\n");
#endif

      /* if this output file already exists, don't process this input file again */
      if (FILE* fp=fopen(forward_argv.at(forward_argc-1),"r")) {
        logMessage(info, true, "Output file: '%s' already exists - skipping pass\n",
                   forward_argv.at(forward_argc-1));
        fclose(fp);
      } else {
        /* call worker's MAIN() */
        result = MAIN(forward_argv.size(), &forward_argv.at(0));
        if (result) {
          logMessage(error, true, "Demodulation failed (error: %i)!\n", result);
          break;
        } else {
          logMessage(debug, true, "Demodulation successful!\n");
          /* prepare for the next pass */
          if(checkpoint_file)
            unlink(checkpoint_file);
        }
      }

      /* prepare for the next pass */
      for(unsigned int arg=0; arg<4; arg++) {
        forward_argc--;
        free(forward_argv.at(forward_argc));
        forward_argv.pop_back();
      }
    }

    /* explicitly free command line resources (skip first entry) */
    for(int i = 1; i < forward_argv.size(); ++i) {
        if(forward_argv.at(i)) {
            free(forward_argv.at(i));
            forward_argv.at(i) = NULL;
        }
    }

    return(result);
}

int main(int argc, char**argv)
{
    int result = 0;

    logMessage(info, true, "Application startup - thank you for supporting Einstein@Home!\n");
    logMessage(debug, true, "Setting up diagnotics and exception handling...\n");

    /* init BOINC diagnostics */
    boinc_init_diagnostics(     BOINC_DIAG_DUMPCALLSTACKENABLED |
            BOINC_DIAG_HEAPCHECKENABLED |
            BOINC_DIAG_ARCHIVESTDERR |
            BOINC_DIAG_REDIRECTSTDERR |
            BOINC_DIAG_TRACETOSTDERR);

    /* pass argc/v to the worker via global vars */
    global_argc = argc;
    global_argv = argv;

    /* the previous boinc_init_diagnostics() call should have installed boinc_catch_signal() for
         SIGILL
         SIGABRT
         SIGBUS
         SIGSEGV
         SIGSYS
         SIGPIPE
         With the current debugging stuff now in boinc/diagnostic.C (for Windows & MacOS)
         it's probably best to leave it that way on everything else but Linux (glibc), where
         backtrace() would give messed up stackframes in the signal handler and we are
         interested in the FPU status word, too.

         NOTE: it is critical to catch SIGINT with our own handler, because a user
         pressing Ctrl-C under boinc should not directly kill the app (which is attached to the
         same terminal), but the app should wait for the client to send <quit/> and cleanly exit.
     */

#ifdef __MINGW32__
    ExchndlSetup();
#elif _WIN32
    signal(SIGTERM, sighandler);
    signal(SIGINT, sighandler);
    signal(SIGFPE, sighandler);
#elif __GLIBC__
    /* this uses unsupported features of the glibc, so don't
         use the (rather portable) boinc_set_signal_handler() here */
    struct sigaction sa;

    sa.sa_sigaction = (void (*)(int, siginfo_t*, void*)) sighandler;
    sigemptyset (&sa.sa_mask);
    sa.sa_flags = SA_RESTART | SA_SIGINFO;

    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGABRT, &sa, NULL);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGFPE, &sa, NULL);
    sigaction(SIGILL, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
#else
    /* install signal handler (generic unix) */
    boinc_set_signal_handler(SIGTERM, sighandler);
    boinc_set_signal_handler(SIGINT, sighandler);
    boinc_set_signal_handler(SIGFPE, boinc_catch_signal);
#endif

    /* boinc_init */
    logMessage(debug, true, "Initializing BOINC...\n");
    erp_set_boinc_options();
    erp_boinc_init();

    /* start program */
    logMessage(debug, true, "Calling worker, let's get started...\n");
    result = worker();

    if(result == RADPUL_CUDA_MEM_ALLOC_HOST ||
       result == RADPUL_CUDA_MEM_ALLOC_DEVICE ||
       result == RADPUL_CUDA_FFT_PLAN ||
       result == RADPUL_OCL_MEM_ALLOC_HOST ||
       result == RADPUL_OCL_MEM_ALLOC_DEVICE)
    {
        logMessage(warn, true, "Sorry, at the moment your system doesn't have enough free CPU/GPU memory to run this task!\n");
        logMessage(warn, false, "Returning control to BOINC, delaying next attempt for at least 15 minutes...\n");
        logMessage(warn, false, "If this problem persists you should consider aborting this task...\n");
        boinc_temporary_exit(900, "Not enough free CPU/GPU memory available! Delaying next attempt for at least 15 minutes...");
    }
    else {
        logMessage(debug, true, "Shutting down BOINC... Bye!\n");
        boinc_finish(result);
    }
    /* boinc_finish() ends the program, we never get here */

#ifdef __MINGW32__
    /* we rather keep exception handling enabled during boinc_finish() and sacrifice proper cleanup */
    ExchndlShutdown();
#endif

    /* just in case */
    return(result);
}
