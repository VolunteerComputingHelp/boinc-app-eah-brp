#!/bin/sh

# Copyright 2017 Steffen Moeller <moeller@debian.org>
# and Christian Dreihsig <christian.dreihsig@t-online.de>
#
# Distributed under terms of the GPL-2 or any later version.

set -e

if [ "-h" = "$1" -o "--help" = "$1" ]; then
cat <<EOHELP
Usage:  $(basename $0) -h|--help - show this help
        $(basename $0) [file]    - compute and write wisdomf to file

DESCRIPTION
        The FFWT library offers the concept of wisdom files to cache
        the best solutions for dissecting large series of data into
        serial invocations of FFWT algorithmic variants.

        This script is provided by the Einstein@Home community to
        support the efficiency of the search for Binary Radio Pulsars
        (BRP) and for sufficiently capable hardware also the Gamma-ray
        pulsar binary search (FGRP).  On a lower level, the wisdom file
        itself is created by the fftwf-wisdom tool as provided by the
        ffwt library which takes between 6 and 120 hours to compute,
        depending on the platform it is executed on.

ENV
        WISDOMFLAGS	If the '-n' or '-c' oprion shall be defined,
                        set the WISDOMFLAGS.

EXAMPLE
        sudo $(basename $0) /etc/fftw/wisdomf
	
        WISDOMFLAGS="-c" $(basename $0)

AUTHORS
        Steffen Moeller <moeller@debian.org>
        Christian Dreihsig <christian.dreihsig@t-online.de>

SEE ALSO
	https://github.com/VolunteerComputingHelp/boinc-app-eah-brp/blob/master/debian/extra/create_wisdomf_eah_brp.sh

EOHELP
exit
fi

DESTFILE="$1"
if [ -z "$1" ]; then
	DESTFILE=/tmp/brp4.$(fftw-wisdom --version | head -n 1 | tr " " "\n" |tail -n 1 )wisdomf
fi

if [ -r "$DESTFILE" ]; then
	echo "E: File '$DESTFILE' already exists."
	exit 1
fi

echo "I: Computing wisdom file for BRP4 and FGRP projects. This will take several hours if not days."
echo

if [ -r /etc/fftw/$(basename $DESTFILE) ]; then
	echo "W: Rename existing /etc/fftw/$(basenme $DESTFILE) if you have not generated it yourself"
	echo
fi

ARCH=$(arch)
if [ -z "$ARCH" ]; then
	echo "E: Cannot tell the platform you are working with, do you have the 'arch' tool installed?"
	exit -1
fi


echo "I: Generating wisdom for (2^22)*3 sample projects (BRP)"
TAGS=rof12582912

if echo "$ARCH" | egrep -q 'i386|x86_64|powerpc'; then
	echo "I: Also generating wisdom for 2^26 sample projects (FGRP)"
	#TAGS="$TAGS rob67108864"

else
	# some extra checks on CPU speeds should go here
	# pointless as of today since no other client than BRP are available for this platform
	echo "W: Not recognising platform '$ARCH' as sufficiently capable for 2^26 bit projects"
fi

if [ -n "$WISDOMFLAGS" ]; then
	echo "I: Adding flags '$WISDOMFLAGS' as defined by envvar WISDOMFLAGS"
fi

fftwf-wisdom $WISDOMFLAGS -v -o "$DESTFILE" $TAGS	# do not use quotes around $TAGS or $WISDOMFLAGS

# -n asks to ignore earlier wisdomf files (not set)
# -c adds a collection of additional plans (not set)
# rof12582912 supports BRP, identified by N30dG
# rob67108864 is meant for a "2^26 length, complex to real inverse FFT" for FGRP
#             as suggested by Bernd Machenschalk

echo "I: Wisdom file was computed successfully. To move the file to where the BRP app looks for it, do:"
echo "   sudo mkdir -p /etc/fftw"
echo "   sudo mv '$DESTFILE' /etc/fftw/"
if [ "$(basename $DESTFILE)" != "brp4.wisdomf" ]; then
	echo "   (cd /etc/fftw && sudo ln -fs $(basename $DESTFILE) brp4.wisdomf)"
fi
