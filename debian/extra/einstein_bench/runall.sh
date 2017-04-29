for APP in `ls apps | grep -v archive`; do
	echo "Starting test with $APP";
	./bench_single.sh ${APP} &
done
# wait for boinc_EinsteinRadio_0 creation
sleep 30;
while true; do
	for APP in `ls apps | grep -v archive`; do
		awk -F"<|>" '/fraction_done/ { print $3 }' ${APP}/boinc_EinsteinRadio_0 ;
	done | xargs;
	sleep 10;
done
