CUDA_PATH=/usr/local/cuda-10.0
SRC_PATH=./src/cuda_base
gcc  -I./parboil_common/include -I$CUDA_PATH/include  -c ./parboil_common/src/args.c -o build_cuda/args.o
gcc  -I./parboil_common/include -I$CUDA_PATH/include  -c ./parboil_common/src/parboil_cuda.c -o build_cuda/parboil_cuda.o
nvcc -I./parboil_common/include -I$CUDA_PATH/include  -c $SRC_PATH/main.cu -o build_cuda/main.o
nvcc -I./parboil_common/include -I../../include -I$CUDA_PATH/include  -c $SRC_PATH/cuenergy_pre.cu -o build_cuda/cp.o
nvcc -o cp build_cuda/*.o
