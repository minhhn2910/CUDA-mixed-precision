nvcc BlackScholes.cu BlackScholes_gold.cpp -I../../common/inc -gencode=arch=compute_60,code=compute_60 -gencode=arch=compute_70,code=compute_70
