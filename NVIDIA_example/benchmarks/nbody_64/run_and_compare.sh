./nbody -benchmark -i=1 -numbodies=327680 -fp64 ;
sleep 0.2;
python calculate_error.py output.txt ref_fp64.txt
