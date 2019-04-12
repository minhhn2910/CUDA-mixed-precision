## CUDA-mixed-precision
Mixed precision between FP32 and FP16x2 in CUDA programs
This repo contains all benchmarks for mixed precision tuning between FP16 and FP32 in CUDA program (and some extension to FP32-FP64, microbenchmarks). The below is the function of each subfolder (and how to run benchmarks in them). There will be further README file in each benchmark to guide the execution and performace measurement if they are not standard:
## Header files' folder
1. `include/` contains the most important part: the operator overload headers and the approximate math library. Without this, the benchmarks cannot compile. 
## FP16-FP32 mix: 
1. `Nvidia_example/benchmarks/`: Each subfolder with the postfix :`_mixed` is the mixed precision version. The makefiles are provided. The `run_and_compare.sh` script is used to measure performance + error with the respective input parameters. For this category, there are 3 benchmarks: 
    * `Nvidia_example/benchmarks/bilateralFilter_mixed`
    * `Nvidia_example/benchmarks/nbody_mixed` 
    * `Nvidia_example/benchmarks/matmul_sharedmem/matrixMul_mixed_final`
2. `axbench/`: There are two benchmarks. For this benchmarks suite, we use the compile.sh script inside each benchmark. If your GPU architecture is different from Volta, please change the `arch` flag for correct compiling. The `run_and_compare.sh` script now takes a speed parameter, which is the number of approximated blocks. It ranges from 0 to 2048. e.g. `./run_and_compare.sh 1024` 
    * `inversek2j_mixed`
    * `newton-raph_mixed`
3. `parboil/`. There are 2 benchmarks. To compile them, run `compile_mixed.sh` or `build_mixed.sh`. To change the approximation degree, please go to the subfolder `cuda_mixed` inside each benchmark to change the `SPEED` definition at the beginning of the code. In the code, there will be comment mentioning the range of the `SPEED` value. Again, `./run_and_compare.sh` is used to run and show the output error.
    * `mri-q`
    * `cp`
4. `rodinia/`. There are 2 benchmarks. To compile them, use the respective make file inside the subfolders. The method to change the approximation degree is similar to `parboil`. 
    * `kmeans_mixed`
    * `lavaMD_mixed` 
## FP32-FP64 mix and auxiliary folder: 
1. `Nvidia_example/benchmarks/BlackScholes_mixed`  contains the mixed precision method for blackscholes. The same compile and run method as `Nvidia_example/`in the previous section. This subfolder also contains the search script on python.
2. `lulesh/lulesh_new/lulesh_mixed` LULESH example. The `makefile` and `run.sh` are provided. The `SPEED` parameter need to be changed. 
3. `funarc/funarc_mixed_fp3264/` contains the arclength example. 
2. `microbenchmarkSFU/benchmarks `: This contains the microbenchmarks and compile scripts to benchmark the math functions in CUDA. There will be a README file explains how to rerun the microbenchmarks

## Input Data: 
The input data are provided in this repo, there is no need to generate input (except for `kmeans`, where the randomly generated input is too large). You can generate input for `kmeans` by going to the `rodinia/data/kmeans/inpuGen/`, type `make ; ./datagen 640000 -f`.

Todo: gradually update README files as I dont have enough time during deadline. 
