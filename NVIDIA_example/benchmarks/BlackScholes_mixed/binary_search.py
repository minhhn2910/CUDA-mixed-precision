#!/usr/bin/python

""" Python version of binary search. Assume that the program outputs directly to stdout.
This version is a blackbox search for to support mixed-precision tuning for GPU application
Written by Minh Ho (minhhn2910(at)gmail.com)
(c) Copyright, All Rights Reserved. NO WARRANTY.
"""
import sys
import subprocess
import shutil
import os

def parse_output(output):
    a = output.split("\n")
    for item in a :
        if('tuning_error' in item):
            tokens = item.split(' ')
            for token in tokens:
                if('tuning_error' in token):
                    arr = token.split("=")
                    return float(arr[1])
    return 0.0
def run_program(program, percentage):
    output = os.popen(program + " "+str(percentage)).read()
    print("run program %d %%, error:%E"%(percentage, parse_output(output)))
    return parse_output(output)

def binary_search(program, error):
    stop_resolution = 1
    upper  = 100
    lower = 0
    while(upper - lower > stop_resolution):
        mid = (upper + lower)/2
        current_error = run_program(program, mid)
        if (current_error < error):
            lower = mid
        else:
            upper = mid
        #print (lower, mid ,upper)
    return lower


def main(argv):
    global SEED_NUMBER
    if (len(argv) != 2):
        print "\n ---------------------------------------------"
        print "Usage: ./search.py program error"
        print "---------------------------------------------\n"
        exit()
    program = './' + argv[0]
    error = float(argv[1])
    #curr_error = parse_output(" adbads\n gasgdae\n re er er tuning_error=0.0001 \n")
    #print run_program(program,80)
    result = binary_search(program, error)
    print ("In total, %d %% of blocks can be run in lower precision"%(result))

if __name__ == '__main__':
    main(sys.argv[1:])

