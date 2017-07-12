#!/usr/bin/env bash
#hackcheck from  https://gist.github.com/shaiguitar/1032229
#check if there are running processes that are not reported in /proc
#referred to at http://serverfault.com/questions/2783/how-do-i-know-if-my-linux-server-has-been-hacked
#run with
#hackcheck
#which will complain on seeing problems, or,
#hackcheck --verbose
#make sure this file is executable by means of
#sudo chmod +x hackcheck.sh

usage()
{
  echo "./$0 [--verbose]"
}

VERBOSE=0
if [ "$1" == "--verbose" ]; then
  VERBOSE=1
fi

max_pid_num=$(cat /proc/sys/kernel/pid_max)
retval=0

for ((i=1;i<$max_pid_num;i++)) ; do
  kill -0 $i 2>/dev/null
  ret=$?
  if [ $ret -eq 0 ]; then
    is_reported_in_proc=$(ls -d /proc/$i)
    if [ "$is_reported_in_proc" = "" ] ; then
      #problem, check this PID.
      echo -e "\e[01;31mPROBLEM\e[0m"
      echo "kill reported process $i $(ps $i |awk '$1!="PID"') to be running, but it does not show in proc!"
      retval=-1
    else
      if [ $VERBOSE -eq 1 ]; then
        echo "Process $i $(ps $i |awk '$1!="PID"') running. exe:$( ls -l /proc/$i/exe 2>/dev/null | awk '{print $10}')"
      fi
    fi
  fi
done

if [ $retval -eq 0 ]; then
    echo "process check was ok"
else
    echo "bad process found"
fi

