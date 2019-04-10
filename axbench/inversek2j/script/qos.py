#!/usr/bin/python

import sys
import math


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def printUsage():
	print "Usage: python qos.py <original file> <nn file>"
	exit(1)
pass;


def findTarget(theta1, theta2, theta3):
    xTgt = math.cos(theta1) + math.cos(theta1+theta2) + math.cos(theta1+theta2+theta3)
    yTgt = math.sin(theta1) + math.sin(theta1+theta2) + math.sin(theta1+theta2+theta3)
    return (xTgt, yTgt)
pass

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)*(x1 - x2) + (y1 - y2) * (y1 - y2))
pass

if(len(sys.argv) != 3):
	printUsage()

origFilename 	= sys.argv[1]
nnFilename		= sys.argv[2]

origLines 		= open(origFilename).readlines()
nnLines			= open(nnFilename).readlines()

total = 0.0

for i in range(len(origLines)):
    origLine 	= origLines[i].rstrip()
    nnLine 		= nnLines[i].rstrip()
    originalSplitted 	 = origLine.split(" ")
    nnSplitted 	         = nnLine.split(" ")

    (xOrig, yOrig)  = findTarget(float(originalSplitted[2]) * (math.pi / 180.0), float(originalSplitted[3])* (math.pi / 180.0), float(originalSplitted[4])* (math.pi / 180.0))
    (xNN,   yNN)    = findTarget(float(nnSplitted[2])* (math.pi / 180.0), float(nnSplitted[3])* (math.pi / 180.0), float(nnSplitted[4])* (math.pi / 180.0))

    total += (distance(xOrig, yOrig, xNN, yNN) / math.sqrt(xOrig*xOrig + yOrig*yOrig))

print bcolors.FAIL	+ "*** Error: %1.2f%%" % ((total/float(len(origLines))) * 100) + bcolors.ENDC
