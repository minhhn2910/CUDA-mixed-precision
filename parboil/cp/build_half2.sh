CUDA_PATH=/usr/local/cuda-10.0
SRC_PATH=./src/cuda_half2
gcc  -I./parboil_common/include -I$CUDA_PATH/include  -c ./parboil_common/src/args.c -o build_cuda/args.o
gcc  -I./parboil_common/include -I$CUDA_PATH/include  -c ./parboil_common/src/parboil_cuda.c -o build_cuda/parboil_cuda.o
nvcc -I./parboil_common/include -I$CUDA_PATH/include  -c $SRC_PATH/main.cu -o build_cuda/main.o
nvcc -arch=sm_70 -I./parboil_common/include -I$CUDA_PATH/include  -c $SRC_PATH/cuenergy_pre8_coalesce.cu -o build_cuda/cp.o
nvcc -arch=sm_70 -o cp build_cuda/*.o
