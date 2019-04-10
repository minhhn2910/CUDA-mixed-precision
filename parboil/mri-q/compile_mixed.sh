CUDA_PATH=/usr/local/cuda-10.0
SRC_PATH=cuda_mixed
ARCH_FLAG='-gencode arch=compute_70,code=sm_70 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_75,code=sm_75'
rm build_cuda/*.o
g++ -O2  -I./parboil_common/include -I$CUDA_PATH/include  -c ./$SRC_PATH/file.cc -o build_cuda/file.o
$CUDA_PATH/bin/nvcc $ARCH_FLAG ./$SRC_PATH/main.cu -O2 -I../../include -I./parboil_common/include -I$CUDA_PATH/include -c -o build_cuda/main.o
$CUDA_PATH/bin/nvcc -O2  -I./parboil_common/include -I$CUDA_PATH/include  -c ./parboil_common/src/parboil_cuda.c -o build_cuda/parboil_cuda.o
gcc -O2  -I./parboil_common/include -I$CUDA_PATH/include  -c ./parboil_common/src/args.c -o build_cuda/args.o
$CUDA_PATH/bin/nvcc -o mri-q -L$CUDA_PATH/lib64 -lm  build_cuda/*.o
