#!/usr/bin/python3

from optparse import OptionParser
import sys, re, string, time, os
from subprocess import Popen
from datetime import datetime

def getcmd(cmd_list, simscript):
  try:
    script = open(simscript,'r')
  except IOError:
    print ("cannot open " + options.simscript)
    sys.exit()

  for line in script:
    line = re.sub('#.*', '', line)
    if len(line) > 2:
      line=line[:-1]
      cmd_list.append(line)

def check_va_randomize():
  va_flag = "/proc/sys/kernel/randomize_va_space"
  try:
    f = open(va_flag,'rt')
  except IOError:
    print ("cannot open " + va_flag)

  flag = re.split("\s*",f.readline())
  if flag[0] != "0":
    print ("Set %s 0"%(va_flag))
    print ("cmd echo 0 > %s"%(va_flag))
    print ("sudo sh -c \"echo 0 > /proc/sys/kernel/randomize_va_space\" ")
#    sys.exit(1)
    print ("for now, it is allowed to run yet")

usage = "usage: %prog [options]"
parser = OptionParser(usage)
parser.add_option("--process", action = "store", type="int", default=4, dest="process", help="maximum active process, default = 4")
parser.add_option("--script", action = "store", type="string", default="/dev/null", dest="simscript", help="cmd list")

(options, args) = parser.parse_args()

active_processes = []
cmd_list = []

check_va_randomize()
getcmd(cmd_list, options.simscript)


#create log dir if it is not exist
host = os.uname()[1]
now = str(datetime.now())
t = re.split("\s*",now)
t1 = re.split("\.",t[1])
t1 = re.split(":",t1[0])
if not os.path.exists("/scale/cal/home/jhpark/log/%s"%t[0]):
  os.makedirs("/scale/cal/home/jhpark/log/%s"%t[0])
#log = open(logfile, 'w')
script = os.path.abspath(options.simscript)
#log.write("Simulation Start: %s with %d processes\n"%(now, options.process))
#log.write("Script: %s\n\n"%(script))
#log.close()

cmd_list.reverse()
# first, check if we have enough processes
# for process in active_processes:
while True:
  try:
    for process in active_processes[:]:
     if process.poll() != None:
       active_processes.remove(process)
       finish = "Process [pid = " + str(process.pid) + "] finished"
       #log = open(logfile,'a')
       #log.write(finish + "\n")
       #log.close()
       print (finish)

    while len(active_processes) < options.process:
      if len(cmd_list) != 0:
        cmd=cmd_list.pop()
        #log = open(logfile, 'a')
        #log.write("started -- " + cmd + "\n")
        #log.close()
      else:
        break
      active_processes.append(Popen([cmd], shell=True))
      print ("Process [pid = "+str(active_processes[-1].pid)+"] started -- " + cmd)

    time.sleep(1)

    if len(active_processes) == 0:
      break

  except KeyboardInterrupt:
    os.killpg(os.getpgrp(), 9)
    for process in active_processes:
      kill = "Process [pid = " + str(process.pid) + "] killed"
      #log = open(logfile, 'a')
      #log.write(kill + "\n")
      #log.close()
      print (kill)
      os.kill(process.pid, 9)
    sys.exit()

#log = open(logfile, 'a')
end = str(datetime.now())
#log.write("\nSimulation End: %s\n"%(end))
#log.close()
sys.exit()
