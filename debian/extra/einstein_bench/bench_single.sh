#!/bin/sh

set -e

EHBENCHHOME="/tmp/einstein_bench/"; 
EINSTEIN_APP=$1
TESTDATADIR=/var/lib/boinc-app-eah-brp/testwu
ZAPFILE="$TESTDATADIR"/p2030.20151015.G187.41-00.88.N.b2s0g0.00000.zap
WU="$TESTDATADIR"/p2030.20151015.G187.41-00.88.N.b2s0g0.00000_1099.bin4

if [ "-h" = "$1" -o "--help" = "$1" ]; then
	cat <<EOHELP
$(basename $0) -h|--help - shows this help
$(basename $0) <path to Einstein@Home BRP search binary> - runs test+benchmark
EOHELP
exit 0
fi

if [ ! -x "$EINSTEIN_APP" ]; then
	echo "E: Expected executable at '$EINSTEIN_APP'"
	exit 1
fi

#cd ${EHBENCHHOME}
RESULTSDIR=$EHBENCHHOME/$(basename ${EINSTEIN_APP})
mkdir -p "$RESULTSDIR"

/usr/bin/time  --format="%C %e sec %U sec %S sec" ${EINSTEIN_APP} -i ${WU} -t "$TESTDATADIR"/stochastic_full.bank -l ${ZAPFILE} -o $RESULTSDIR/results.cand0 -c $RESULTSDIR/nocheckpoint.cpt -A 0.08 -P 3.0 -f 400.0 -W -z >> $RESULTSDIR/TIMEplusSTDOUT 2>&1;
