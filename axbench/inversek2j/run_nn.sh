#!/bin/bash

# Regular Colors
Black='\e[0;30m'        # Black
Red='\e[0;31m'          # Red
Green='\e[0;32m'        # Green
Yellow='\e[0;33m'       # Yellow
Blue='\e[0;34m'         # Blue
Purple='\e[0;35m'       # Purple
Cyan='\e[0;36m'         # Cyan
White='\e[0;37m'        # White


echo -e "${Green} nrpoly3 Starting... ${White}"

if [ ! -d ./train.data/output/kernel.data ]; then
	mkdir ./train.data/output/kernel.data
fi

for f in test.data/input/*.txt
do
	filename=$(basename "$f")
	extension="${filename##*.}"
	filename="${filename%.*}"
	echo -e "-------------------------------------------------------"
	echo -e "${Green} Input Coefficient File:  $f"
	echo -e "${Green} Output Root File: ./test.data/output/${filename}_angles.txt ${White}"
	echo -e "${Green} Output Root File: ./test.data/output/${filename}_angles_nn.txt ${White}"
	echo -e "-------------------------------------------------------"
	./bin/invkin.out $f ./test.data/output/${filename}_angles.txt 0.001
	./bin/invkin_nn.out $f ./test.data/output/${filename}_angles_nn.txt 0.001
	python ./script/qos.py ./test.data/output/${filename}_angles.txt ./test.data/output/${filename}_angles_nn.txt
done
