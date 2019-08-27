CUDA_PATH=/usr/local/cuda
rm build_cuda/*.o
g++ -O2  -I./parboil_common/include -I$CUDA_PATH/include  -c ./cuda_64/file.cc -o build_cuda/file.o
$CUDA_PATH/bin/nvcc ./cuda_64/main.cu -O2 -I./parboil_common/include -I$CUDA_PATH/include  -c -o build_cuda/main.o
$CUDA_PATH/bin/nvcc -O2  -I./parboil_common/include -I$CUDA_PATH/include  -c ./parboil_common/src/parboil_cuda.c -o build_cuda/parboil_cuda.o
gcc -O2  -I./parboil_common/include -I$CUDA_PATH/include  -c ./parboil_common/src/args.c -o build_cuda/args.o
$CUDA_PATH/bin/nvcc -o mri-q -L$CUDA_PATH/lib64 -lm  build_cuda/*.o
