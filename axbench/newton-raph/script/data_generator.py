#!/usr/bin/python

# Designed by: Amir Yazdanbakhsh
# Date: March 26th - 2015
# Alternative Computing Technologies Lab.
# Georgia Institute of Technology

import sys
import random

def Usage():
	print "Usage: python data_generator.py <size> <output file>"
	exit

if(len(sys.argv) != 3):
	Usage()

data_size 	= sys.argv[1]
coeff_out 	= open(sys.argv[2], 'w')

percent_div = 1 # Number between 0 and 100 (inclusive)

coeff_out.write(str(data_size) + "\n")

for i in range(int(data_size)):#16:

	A_tmp 	= random.randint(0, 20)
	B_tmp 	= random.randint(0, 20)
	C_tmp 	= random.randint(0, 20)
	D_tmp 	= random.randint(-6000, -2000)
	x0_tmp	= random.randint(0, 15000)

	if(D_tmp % 100 < percent_div):
		C_tmp 	= 0.0
		x0_tmp 	= 0.0
	pass;

	coeff_out.write("%f %f %f %f %f\n" % (A_tmp, B_tmp, C_tmp, D_tmp, x0_tmp))
pass;

print "Thank you..."
