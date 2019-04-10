CUDA_PATH=/usr/local/cuda-10.0
SRC_PATH=./src/cuda_mixed
ARCH_FLAG='-gencode arch=compute_70,code=sm_70 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_75,code=sm_75'
rm build_cuda/*.o
gcc  -I./parboil_common/include -I$CUDA_PATH/include  -c ./parboil_common/src/args.c -o build_cuda/args.o
gcc  -I./parboil_common/include -I$CUDA_PATH/include  -c ./parboil_common/src/parboil_cuda.c -o build_cuda/parboil_cuda.o
nvcc -I./parboil_common/include -I$CUDA_PATH/include  -c $SRC_PATH/main.cu -o build_cuda/main.o
nvcc $ARCH_FLAG -I./parboil_common/include -I../../include -I$CUDA_PATH/include  -c $SRC_PATH/cuenergy_pre8_coalesce.cu -o build_cuda/cp.o
nvcc $ARCH_FLAG -o cp build_cuda/*.o
