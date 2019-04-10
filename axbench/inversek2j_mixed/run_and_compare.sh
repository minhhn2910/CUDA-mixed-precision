./a.out coord2_20.txt testoutput.txt 0.1 $1;
python script/qos.py ref.txt testoutput.txt ;
python calculate_error.py testoutput.txt ref.txt
