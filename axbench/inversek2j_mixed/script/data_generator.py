#!/usr/bin/python

# Designed by: Amir Yazdanbakhsh
# Date: March 26th - 2015
# Alternative Computing Technologies Lab.
# Georgia Institute of Technology

import sys
import random
import math

def Usage():
	print "Usage: python data_generator.py <size> <output file>"
	exit(1)

if(len(sys.argv) != 3):
	Usage()

data_size 	= sys.argv[1] 
coord_out 	= open(sys.argv[2], 'w')

percent_div = 1 # Number between 0 and 100 (inclusive)

coord_out.write(str(data_size) + "\n")

for i in range(int(data_size)):
	# Generate x and y target coordinates as random floats from -NUM_JOINTS to NUM_JOINTS
	xtmp = (int)((1.5 * 3 * (random.uniform(0.0,1.0)) - (0.75 * 3)) * 1000) / 1000.0;
	ytmp = (int)((0.8 * 3 * (random.uniform(0.0,1.0)) + (0.1 * 3)) * 1000) / 1000.0;
	# Check to make sure joint arm can reach target
	len = math.sqrt(xtmp * xtmp + ytmp * ytmp);

	while(len >= 3):
		# Generate x and y target coordinates as random floats from -NUM_JOINTS to NUM_JOINTS
		xtmp = (int)((1.5 * 3 * (random.uniform(0.0,1.0)) - (0.75 * 3)) * 1000) / 1000.0;
		ytmp = (int)((0.8 * 3 * (random.uniform(0.0,1.0)) + (0.1 * 3)) * 1000) / 1000.0;
		# Check to make sure joint arm can reach target
		len = math.sqrt(xtmp * xtmp + ytmp * ytmp);

	coord_out.write("%f %f\n" % (xtmp, ytmp))
pass;

print "Thank you..."