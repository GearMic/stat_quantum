#pragma once

#include <stdio.h>
#include <curand_kernel.h>


// macros for error checking
#ifndef NDEBUG
#define CUDA_CALL(x) do { \
cudaError_t err = x; \
if(err != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    printf("%s\n", cudaGetErrorString(err)); \
}} while(0)
#define CURAND_CALL(x) do { \
cudaError_t err = x; \
if(err != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    printf("\t%s\n", cudaGetErrorString(err)); \
}} while(0)

#else
#define CUDA_CALL(x) do { \
    x; \
}} while(0)
#define CURAND_CALL(x) do { \
    x; \
}} while(0)

#endif


__global__ void setup_randomize(curandState_t* state, size_t len, unsigned long long seed);
__global__ void randomize_double_array(double* array, size_t len, double lower, double upper, curandState_t* state);
void printfl(double x);
void export_csv_double_1d(FILE* file, double* arr, size_t cols);
void export_csv_double_2d(FILE* file, double* arr, size_t pitch, size_t width, size_t height);
size_t cuda_block_amount(size_t kernels, size_t max_kernels);