PYTHONPATH=$PYTHONPATH:./parboil_common/python
./mri-q -i ./data/large/input/64_64_64_dataset.bin -o test_output.bin;
python compare.py test_output.bin ref.bin 
