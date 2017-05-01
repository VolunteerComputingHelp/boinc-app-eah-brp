#!/bin/sh

set -e

BENCHMARKSCRIPT=$(dirname $0)/bench_single.sh 

if [ ! -x "$BENCHMARKSCRIPT" ]; then
	echo "E: Could not find benchmark script on '$BENCHMARKSCRIPT'."
	exit 1
fi

for APP in `ls apps | grep -v archive`; do
	echo "I: Starting test with $APP";
	$BENCHMARKSCRIPT ${APP} &
done

# wait for boinc_EinsteinRadio_0 creation
sleep 30;

while true; do
	for APP in `ls apps | grep -v archive`; do
		awk -F"<|>" '/fraction_done/ { print $3 }' ${APP}/boinc_EinsteinRadio_0 ;
	done | xargs;
	sleep 10;
done
