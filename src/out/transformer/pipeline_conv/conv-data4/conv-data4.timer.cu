/*****************************************
Emitting C Generated Code
*******************************************/
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "cudnn_header.h"
#include "nccl_header.h"
#include <string.h>
#include <cblas.h>
#include <stdlib.h>
#include "cuda_header.h"
#include <stdio.h>
#include <stdint.h>
#include "cublas_header.h"
#include <stdbool.h>
#include "mpi_header.h"
#include "scanner_header.h"
/************* Functions **************/
__global__ void x10(float* x11, float x12, int x13) {
  // begin generating kernel function for FILL of type Float
  int x14 = gridDim.x * blockDim.x;
  int x15 = threadIdx.x + blockIdx.x * blockDim.x;
  while (x15 < x13) {
    x11[x15] = x12;
    x15 = x15 + x14;
  }
  // end generating kernel function for FILL of type Float
}
__global__ void x63(float* x64, float* x65, int x66) {
  // begin generating kernel function for RELU of type Float
  int x67 = gridDim.x * blockDim.x;
  int x68 = threadIdx.x + blockIdx.x * blockDim.x;
  while (x68 < x66) {
    int x69 = x68;
    x65[x69] = max(0.0, x64[x69]);
    x68 = x68 + x67;
  }
  // end generating kernel function for RELU of type Float
}
__global__ void x172(float* x173, float* x174, float* x175, int x176) {
  // begin generating kernel function for RELU_GRAD of type Float
  int x177 = gridDim.x * blockDim.x;
  int x178 = threadIdx.x + blockIdx.x * blockDim.x;
  while (x178 < x176) {
    int x179 = x178;
    x175[x179] = x174[x179] > 0.0 ? x173[x179] : 0.0;
    x178 = x178 + x177;
  }
  // end generating kernel function for RELU_GRAD of type Float
}
__global__ void x180(float* x181, float* x182, int x183) {
  // begin generating kernel function for ACCUM of type Float
  int x184 = gridDim.x * blockDim.x;
  int x185 = threadIdx.x + blockIdx.x * blockDim.x;
  while (x185 < x183) {
    int x186 = x185;
    x181[x186] = x181[x186] + x182[x186];
    x185 = x185 + x184;
  }
  // end generating kernel function for ACCUM of type Float
}
__global__ void x336(float* x337, float* x338, float* x339, int x340) {
  // begin generating kernel function for SGD of type Float
  int x341 = gridDim.x * blockDim.x;
  int x342 = threadIdx.x + blockIdx.x * blockDim.x;
  while (x342 < x340) {
    int x343 = x342;
    float x344 = x339[x343] * 0.5 + x338[x343];
    x337[x343] = x337[x343] - x344 * 1.0E-4;
    x339[x343] = x344;
    x342 = x342 + x341;
  }
  // end generating kernel function for SGD of type Float
}
/**************** Snippet ****************/
void Snippet(int x0) {
  // begin setting up the MPI/NCCL environment
  int x1 = 0;
  int x2 = 0;
  MPICHECK(MPI_Init(NULL, NULL));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &x2));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &x1));
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  CUDA_CALL(cudaSetDevice(x2));
  ncclUniqueId x3;
  NCCLCHECK(ncclGetUniqueId(&x3));
  MPICHECK(MPI_Bcast(&x3, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, MPI_COMM_WORLD));
  ncclComm_t x4;
  NCCLCHECK(ncclCommInitRank(&x4, x1, x3, x2));
  cudaStream_t x5;
  CUDA_CALL(cudaStreamCreateWithFlags(&x5, cudaStreamNonBlocking));
  int x6 = x2;
  // end setting up the MPI/NCCL environment
  // begin setting up the CUDNN environment
  cudnnHandle_t x7;
  CUDNNCHECK(cudnnCreate(&x7));
  // end setting up the CUDNN environment
  // begin initializing random GPU array of size 73984 and type Float at device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x8 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x8, (size_t)(73984 * sizeof(float))));
  // end initializing random GPU array of size 73984 and type Float at device (pre-rename) x39
  // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x9 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x9, (size_t)(73984 * sizeof(float))));
  x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x9, 0, 73984);
  // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x16 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x16, (size_t)(73984 * sizeof(float))));
  x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x16, 0, 73984);
  // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  // begin initializing random GPU array of size 73984 and type Float at device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x17 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x17, (size_t)(73984 * sizeof(float))));
  // end initializing random GPU array of size 73984 and type Float at device (pre-rename) x39
  // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x18 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x18, (size_t)(73984 * sizeof(float))));
  x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x18, 0, 73984);
  // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x19 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x19, (size_t)(73984 * sizeof(float))));
  x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x19, 0, 73984);
  // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  // begin initializing random GPU array of size 73984 and type Float at device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x20 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x20, (size_t)(73984 * sizeof(float))));
  // end initializing random GPU array of size 73984 and type Float at device (pre-rename) x39
  // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x21 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x21, (size_t)(73984 * sizeof(float))));
  x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x21, 0, 73984);
  // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x22 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x22, (size_t)(73984 * sizeof(float))));
  x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x22, 0, 73984);
  // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  // begin initializing random GPU array of size 73984 and type Float at device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x23 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x23, (size_t)(73984 * sizeof(float))));
  // end initializing random GPU array of size 73984 and type Float at device (pre-rename) x39
  // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x24 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x24, (size_t)(73984 * sizeof(float))));
  x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x24, 0, 73984);
  // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x25 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x25, (size_t)(73984 * sizeof(float))));
  x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x25, 0, 73984);
  // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  // begin initializing random GPU array of size 73984 and type Float at device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x26 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x26, (size_t)(73984 * sizeof(float))));
  // end initializing random GPU array of size 73984 and type Float at device (pre-rename) x39
  // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x27 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x27, (size_t)(73984 * sizeof(float))));
  x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x27, 0, 73984);
  // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x28 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x28, (size_t)(73984 * sizeof(float))));
  x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x28, 0, 73984);
  // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  // begin initializing random GPU array of size 73984 and type Float at device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x29 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x29, (size_t)(73984 * sizeof(float))));
  // end initializing random GPU array of size 73984 and type Float at device (pre-rename) x39
  // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x30 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x30, (size_t)(73984 * sizeof(float))));
  x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x30, 0, 73984);
  // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x31 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x31, (size_t)(73984 * sizeof(float))));
  x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x31, 0, 73984);
  // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  // begin initializing random GPU array of size 73984 and type Float at device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x32 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x32, (size_t)(73984 * sizeof(float))));
  // end initializing random GPU array of size 73984 and type Float at device (pre-rename) x39
  // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x33 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x33, (size_t)(73984 * sizeof(float))));
  x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x33, 0, 73984);
  // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x34 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x34, (size_t)(73984 * sizeof(float))));
  x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x34, 0, 73984);
  // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  // begin initializing random GPU array of size 73984 and type Float at device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x35 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x35, (size_t)(73984 * sizeof(float))));
  // end initializing random GPU array of size 73984 and type Float at device (pre-rename) x39
  // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x36 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x36, (size_t)(73984 * sizeof(float))));
  x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x36, 0, 73984);
  // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x37 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x37, (size_t)(73984 * sizeof(float))));
  x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x37, 0, 73984);
  // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  // begin initializing random GPU array of size 73984 and type Float at device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x38 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x38, (size_t)(73984 * sizeof(float))));
  // end initializing random GPU array of size 73984 and type Float at device (pre-rename) x39
  // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x39 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x39, (size_t)(73984 * sizeof(float))));
  x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x39, 0, 73984);
  // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x40 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x40, (size_t)(73984 * sizeof(float))));
  x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x40, 0, 73984);
  // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  // begin initializing random GPU array of size 73984 and type Float at device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x41 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x41, (size_t)(73984 * sizeof(float))));
  // end initializing random GPU array of size 73984 and type Float at device (pre-rename) x39
  // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x42 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x42, (size_t)(73984 * sizeof(float))));
  x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x42, 0, 73984);
  // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x43 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x43, (size_t)(73984 * sizeof(float))));
  x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x43, 0, 73984);
  // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  // begin initializing random GPU array of size 73984 and type Float at device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x44 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x44, (size_t)(73984 * sizeof(float))));
  // end initializing random GPU array of size 73984 and type Float at device (pre-rename) x39
  // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x45 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x45, (size_t)(73984 * sizeof(float))));
  x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x45, 0, 73984);
  // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x46 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x46, (size_t)(73984 * sizeof(float))));
  x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x46, 0, 73984);
  // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  // begin initializing random GPU array of size 73984 and type Float at device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x47 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x47, (size_t)(73984 * sizeof(float))));
  // end initializing random GPU array of size 73984 and type Float at device (pre-rename) x39
  // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x48 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x48, (size_t)(73984 * sizeof(float))));
  x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x48, 0, 73984);
  // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  CUDA_CALL(cudaSetDevice(x6));
  float* x49 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x49, (size_t)(73984 * sizeof(float))));
  x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x49, 0, 73984);
  // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x39
  int x50 = 0;
  float* x51 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x51, (size_t)(16384 * sizeof(float))));
  cudnnTensorDescriptor_t x52;
  cudnnCreateTensorDescriptor(&x52);
  cudnnSetTensor4dDescriptor(x52, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 128, 128);
  cudnnFilterDescriptor_t x53;
  cudnnCreateFilterDescriptor(&x53);
  cudnnSetFilter4dDescriptor(x53, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 256, 1, 17, 17);
  cudnnConvolutionDescriptor_t x54;
  cudnnCreateConvolutionDescriptor(&x54);
  cudnnSetConvolution2dDescriptor(x54, 8, 8, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
  cudnnTensorDescriptor_t x55;
  cudnnCreateTensorDescriptor(&x55);
  cudnnSetTensor4dDescriptor(x55, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 256, 128, 128);
  float* x56 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x56, (size_t)(4194304 * sizeof(float))));
  cudnnConvolutionFwdAlgoPerf_t x57;
  int x58 = 0;
  cudnnFindConvolutionForwardAlgorithm(x7, x52, x53, x54, x55, 1, &x58, &x57);
  cudnnConvolutionFwdAlgo_t x59 = x57.algo;
  size_t x60 = (size_t)0;
  cudnnGetConvolutionForwardWorkspaceSize(x7, x52, x53, x54, x55, x59, &x60);
  float* x61 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x61, (size_t)x60));
  float* x62 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x62, (size_t)(4194304 * sizeof(float))));
  float* x70 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x70, (size_t)(4194304 * sizeof(float))));
  cudnnConvolutionFwdAlgoPerf_t x71;
  int x72 = 0;
  cudnnFindConvolutionForwardAlgorithm(x7, x55, x53, x54, x55, 1, &x72, &x71);
  cudnnConvolutionFwdAlgo_t x73 = x71.algo;
  size_t x74 = (size_t)0;
  cudnnGetConvolutionForwardWorkspaceSize(x7, x55, x53, x54, x55, x73, &x74);
  float* x75 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x75, (size_t)x74));
  float* x76 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x76, (size_t)(4194304 * sizeof(float))));
  float* x77 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x77, (size_t)(4194304 * sizeof(float))));
  cudnnConvolutionFwdAlgoPerf_t x78;
  int x79 = 0;
  cudnnFindConvolutionForwardAlgorithm(x7, x55, x53, x54, x55, 1, &x79, &x78);
  cudnnConvolutionFwdAlgo_t x80 = x78.algo;
  size_t x81 = (size_t)0;
  cudnnGetConvolutionForwardWorkspaceSize(x7, x55, x53, x54, x55, x80, &x81);
  float* x82 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x82, (size_t)x81));
  float* x83 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x83, (size_t)(4194304 * sizeof(float))));
  float* x84 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x84, (size_t)(4194304 * sizeof(float))));
  cudnnConvolutionFwdAlgoPerf_t x85;
  int x86 = 0;
  cudnnFindConvolutionForwardAlgorithm(x7, x55, x53, x54, x55, 1, &x86, &x85);
  cudnnConvolutionFwdAlgo_t x87 = x85.algo;
  size_t x88 = (size_t)0;
  cudnnGetConvolutionForwardWorkspaceSize(x7, x55, x53, x54, x55, x87, &x88);
  float* x89 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x89, (size_t)x88));
  float* x90 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x90, (size_t)(4194304 * sizeof(float))));
  float* x91 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x91, (size_t)(4194304 * sizeof(float))));
  cudnnConvolutionFwdAlgoPerf_t x92;
  int x93 = 0;
  cudnnFindConvolutionForwardAlgorithm(x7, x55, x53, x54, x55, 1, &x93, &x92);
  cudnnConvolutionFwdAlgo_t x94 = x92.algo;
  size_t x95 = (size_t)0;
  cudnnGetConvolutionForwardWorkspaceSize(x7, x55, x53, x54, x55, x94, &x95);
  float* x96 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x96, (size_t)x95));
  float* x97 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x97, (size_t)(4194304 * sizeof(float))));
  cudnnFilterDescriptor_t x98;
  cudnnCreateFilterDescriptor(&x98);
  cudnnSetFilter4dDescriptor(x98, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 256, 17, 17);
  float* x99 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x99, (size_t)(16384 * sizeof(float))));
  cudnnConvolutionFwdAlgoPerf_t x100;
  int x101 = 0;
  cudnnFindConvolutionForwardAlgorithm(x7, x55, x98, x54, x52, 1, &x101, &x100);
  cudnnConvolutionFwdAlgo_t x102 = x100.algo;
  size_t x103 = (size_t)0;
  cudnnGetConvolutionForwardWorkspaceSize(x7, x55, x98, x54, x52, x102, &x103);
  float* x104 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x104, (size_t)x103));
  float* x105 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x105, (size_t)(16384 * sizeof(float))));
  float* x106 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x106, (size_t)(4194304 * sizeof(float))));
  cudnnConvolutionFwdAlgoPerf_t x107;
  int x108 = 0;
  cudnnFindConvolutionForwardAlgorithm(x7, x52, x53, x54, x55, 1, &x108, &x107);
  cudnnConvolutionFwdAlgo_t x109 = x107.algo;
  size_t x110 = (size_t)0;
  cudnnGetConvolutionForwardWorkspaceSize(x7, x52, x53, x54, x55, x109, &x110);
  float* x111 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x111, (size_t)x110));
  float* x112 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x112, (size_t)(4194304 * sizeof(float))));
  float* x113 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x113, (size_t)(4194304 * sizeof(float))));
  cudnnConvolutionFwdAlgoPerf_t x114;
  int x115 = 0;
  cudnnFindConvolutionForwardAlgorithm(x7, x55, x53, x54, x55, 1, &x115, &x114);
  cudnnConvolutionFwdAlgo_t x116 = x114.algo;
  size_t x117 = (size_t)0;
  cudnnGetConvolutionForwardWorkspaceSize(x7, x55, x53, x54, x55, x116, &x117);
  float* x118 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x118, (size_t)x117));
  float* x119 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x119, (size_t)(4194304 * sizeof(float))));
  float* x120 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x120, (size_t)(4194304 * sizeof(float))));
  cudnnConvolutionFwdAlgoPerf_t x121;
  int x122 = 0;
  cudnnFindConvolutionForwardAlgorithm(x7, x55, x53, x54, x55, 1, &x122, &x121);
  cudnnConvolutionFwdAlgo_t x123 = x121.algo;
  size_t x124 = (size_t)0;
  cudnnGetConvolutionForwardWorkspaceSize(x7, x55, x53, x54, x55, x123, &x124);
  float* x125 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x125, (size_t)x124));
  float* x126 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x126, (size_t)(4194304 * sizeof(float))));
  float* x127 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x127, (size_t)(4194304 * sizeof(float))));
  cudnnConvolutionFwdAlgoPerf_t x128;
  int x129 = 0;
  cudnnFindConvolutionForwardAlgorithm(x7, x55, x53, x54, x55, 1, &x129, &x128);
  cudnnConvolutionFwdAlgo_t x130 = x128.algo;
  size_t x131 = (size_t)0;
  cudnnGetConvolutionForwardWorkspaceSize(x7, x55, x53, x54, x55, x130, &x131);
  float* x132 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x132, (size_t)x131));
  float* x133 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x133, (size_t)(4194304 * sizeof(float))));
  float* x134 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x134, (size_t)(4194304 * sizeof(float))));
  cudnnConvolutionFwdAlgoPerf_t x135;
  int x136 = 0;
  cudnnFindConvolutionForwardAlgorithm(x7, x55, x53, x54, x55, 1, &x136, &x135);
  cudnnConvolutionFwdAlgo_t x137 = x135.algo;
  size_t x138 = (size_t)0;
  cudnnGetConvolutionForwardWorkspaceSize(x7, x55, x53, x54, x55, x137, &x138);
  float* x139 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x139, (size_t)x138));
  float* x140 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x140, (size_t)(4194304 * sizeof(float))));
  float* x141 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x141, (size_t)(16384 * sizeof(float))));
  cudnnConvolutionFwdAlgoPerf_t x142;
  int x143 = 0;
  cudnnFindConvolutionForwardAlgorithm(x7, x55, x98, x54, x52, 1, &x143, &x142);
  cudnnConvolutionFwdAlgo_t x144 = x142.algo;
  size_t x145 = (size_t)0;
  cudnnGetConvolutionForwardWorkspaceSize(x7, x55, x98, x54, x52, x144, &x145);
  float* x146 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x146, (size_t)x145));
  float* x147 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x147, (size_t)(4194304 * sizeof(float))));
  float* x148 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x148, (size_t)(4194304 * sizeof(float))));
  float* x149 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x149, (size_t)(4194304 * sizeof(float))));
  float* x150 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x150, (size_t)(4194304 * sizeof(float))));
  float* x151 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x151, (size_t)(4194304 * sizeof(float))));
  float* x152 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x152, (size_t)(4194304 * sizeof(float))));
  float* x153 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x153, (size_t)(4194304 * sizeof(float))));
  float* x154 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x154, (size_t)(4194304 * sizeof(float))));
  float* x155 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x155, (size_t)(4194304 * sizeof(float))));
  float* x156 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x156, (size_t)(4194304 * sizeof(float))));
  float* x157 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x157, (size_t)(16384 * sizeof(float))));
  float* x158 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x158, (size_t)(16384 * sizeof(float))));
  float* x159 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x159, (size_t)(4194304 * sizeof(float))));
  float* x160 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x160, (size_t)(4194304 * sizeof(float))));
  float* x161 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x161, (size_t)(4194304 * sizeof(float))));
  float* x162 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x162, (size_t)(4194304 * sizeof(float))));
  float* x163 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x163, (size_t)(4194304 * sizeof(float))));
  float* x164 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x164, (size_t)(4194304 * sizeof(float))));
  float* x165 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x165, (size_t)(4194304 * sizeof(float))));
  float* x166 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x166, (size_t)(4194304 * sizeof(float))));
  float* x167 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x167, (size_t)(4194304 * sizeof(float))));
  float* x168 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x168, (size_t)(4194304 * sizeof(float))));
  float* x169 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x169, (size_t)(16384 * sizeof(float))));
  float* x170 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x170, (size_t)(16384 * sizeof(float))));
  float* x171 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x171, (size_t)(16384 * sizeof(float))));
  float* x187 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x187, (size_t)(73984 * sizeof(float))));
  cudnnConvolutionBwdFilterAlgoPerf_t x188;
  int x189 = 0;
  cudnnFindConvolutionBackwardFilterAlgorithm(x7, x55, x52, x54, x98, 1, &x189, &x188);
  cudnnConvolutionBwdFilterAlgo_t x190 = x188.algo;
  size_t x191 = (size_t)0;
  cudnnGetConvolutionBackwardFilterWorkspaceSize(x7, x55, x52, x54, x98, x190, &x191);
  float* x192 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x192, (size_t)x191));
  float* x193 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x193, (size_t)(4194304 * sizeof(float))));
  cudnnConvolutionBwdDataAlgoPerf_t x194;
  int x195 = 0;
  cudnnFindConvolutionBackwardDataAlgorithm(x7, x98, x52, x54, x55, 1, &x195, &x194);
  cudnnConvolutionBwdDataAlgo_t x196 = x194.algo;
  size_t x197 = (size_t)0;
  cudnnGetConvolutionBackwardDataWorkspaceSize(x7, x98, x52, x54, x55, x196, &x197);
  float* x198 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x198, (size_t)x197));
  float* x199 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x199, (size_t)(4194304 * sizeof(float))));
  float* x200 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x200, (size_t)(73984 * sizeof(float))));
  cudnnConvolutionBwdFilterAlgoPerf_t x201;
  int x202 = 0;
  cudnnFindConvolutionBackwardFilterAlgorithm(x7, x55, x55, x54, x53, 1, &x202, &x201);
  cudnnConvolutionBwdFilterAlgo_t x203 = x201.algo;
  size_t x204 = (size_t)0;
  cudnnGetConvolutionBackwardFilterWorkspaceSize(x7, x55, x55, x54, x53, x203, &x204);
  float* x205 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x205, (size_t)x204));
  float* x206 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x206, (size_t)(4194304 * sizeof(float))));
  cudnnConvolutionBwdDataAlgoPerf_t x207;
  int x208 = 0;
  cudnnFindConvolutionBackwardDataAlgorithm(x7, x53, x55, x54, x55, 1, &x208, &x207);
  cudnnConvolutionBwdDataAlgo_t x209 = x207.algo;
  size_t x210 = (size_t)0;
  cudnnGetConvolutionBackwardDataWorkspaceSize(x7, x53, x55, x54, x55, x209, &x210);
  float* x211 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x211, (size_t)x210));
  float* x212 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x212, (size_t)(4194304 * sizeof(float))));
  float* x213 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x213, (size_t)(73984 * sizeof(float))));
  cudnnConvolutionBwdFilterAlgoPerf_t x214;
  int x215 = 0;
  cudnnFindConvolutionBackwardFilterAlgorithm(x7, x55, x55, x54, x53, 1, &x215, &x214);
  cudnnConvolutionBwdFilterAlgo_t x216 = x214.algo;
  size_t x217 = (size_t)0;
  cudnnGetConvolutionBackwardFilterWorkspaceSize(x7, x55, x55, x54, x53, x216, &x217);
  float* x218 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x218, (size_t)x217));
  float* x219 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x219, (size_t)(4194304 * sizeof(float))));
  cudnnConvolutionBwdDataAlgoPerf_t x220;
  int x221 = 0;
  cudnnFindConvolutionBackwardDataAlgorithm(x7, x53, x55, x54, x55, 1, &x221, &x220);
  cudnnConvolutionBwdDataAlgo_t x222 = x220.algo;
  size_t x223 = (size_t)0;
  cudnnGetConvolutionBackwardDataWorkspaceSize(x7, x53, x55, x54, x55, x222, &x223);
  float* x224 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x224, (size_t)x223));
  float* x225 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x225, (size_t)(4194304 * sizeof(float))));
  float* x226 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x226, (size_t)(73984 * sizeof(float))));
  cudnnConvolutionBwdFilterAlgoPerf_t x227;
  int x228 = 0;
  cudnnFindConvolutionBackwardFilterAlgorithm(x7, x55, x55, x54, x53, 1, &x228, &x227);
  cudnnConvolutionBwdFilterAlgo_t x229 = x227.algo;
  size_t x230 = (size_t)0;
  cudnnGetConvolutionBackwardFilterWorkspaceSize(x7, x55, x55, x54, x53, x229, &x230);
  float* x231 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x231, (size_t)x230));
  float* x232 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x232, (size_t)(4194304 * sizeof(float))));
  cudnnConvolutionBwdDataAlgoPerf_t x233;
  int x234 = 0;
  cudnnFindConvolutionBackwardDataAlgorithm(x7, x53, x55, x54, x55, 1, &x234, &x233);
  cudnnConvolutionBwdDataAlgo_t x235 = x233.algo;
  size_t x236 = (size_t)0;
  cudnnGetConvolutionBackwardDataWorkspaceSize(x7, x53, x55, x54, x55, x235, &x236);
  float* x237 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x237, (size_t)x236));
  float* x238 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x238, (size_t)(4194304 * sizeof(float))));
  float* x239 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x239, (size_t)(73984 * sizeof(float))));
  cudnnConvolutionBwdFilterAlgoPerf_t x240;
  int x241 = 0;
  cudnnFindConvolutionBackwardFilterAlgorithm(x7, x55, x55, x54, x53, 1, &x241, &x240);
  cudnnConvolutionBwdFilterAlgo_t x242 = x240.algo;
  size_t x243 = (size_t)0;
  cudnnGetConvolutionBackwardFilterWorkspaceSize(x7, x55, x55, x54, x53, x242, &x243);
  float* x244 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x244, (size_t)x243));
  float* x245 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x245, (size_t)(4194304 * sizeof(float))));
  cudnnConvolutionBwdDataAlgoPerf_t x246;
  int x247 = 0;
  cudnnFindConvolutionBackwardDataAlgorithm(x7, x53, x55, x54, x55, 1, &x247, &x246);
  cudnnConvolutionBwdDataAlgo_t x248 = x246.algo;
  size_t x249 = (size_t)0;
  cudnnGetConvolutionBackwardDataWorkspaceSize(x7, x53, x55, x54, x55, x248, &x249);
  float* x250 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x250, (size_t)x249));
  float* x251 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x251, (size_t)(4194304 * sizeof(float))));
  float* x252 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x252, (size_t)(73984 * sizeof(float))));
  cudnnConvolutionBwdFilterAlgoPerf_t x253;
  int x254 = 0;
  cudnnFindConvolutionBackwardFilterAlgorithm(x7, x52, x55, x54, x53, 1, &x254, &x253);
  cudnnConvolutionBwdFilterAlgo_t x255 = x253.algo;
  size_t x256 = (size_t)0;
  cudnnGetConvolutionBackwardFilterWorkspaceSize(x7, x52, x55, x54, x53, x255, &x256);
  float* x257 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x257, (size_t)x256));
  float* x258 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x258, (size_t)(16384 * sizeof(float))));
  cudnnConvolutionBwdDataAlgoPerf_t x259;
  int x260 = 0;
  cudnnFindConvolutionBackwardDataAlgorithm(x7, x53, x55, x54, x52, 1, &x260, &x259);
  cudnnConvolutionBwdDataAlgo_t x261 = x259.algo;
  size_t x262 = (size_t)0;
  cudnnGetConvolutionBackwardDataWorkspaceSize(x7, x53, x55, x54, x52, x261, &x262);
  float* x263 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x263, (size_t)x262));
  float* x264 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x264, (size_t)(16384 * sizeof(float))));
  float* x265 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x265, (size_t)(73984 * sizeof(float))));
  cudnnConvolutionBwdFilterAlgoPerf_t x266;
  int x267 = 0;
  cudnnFindConvolutionBackwardFilterAlgorithm(x7, x55, x52, x54, x98, 1, &x267, &x266);
  cudnnConvolutionBwdFilterAlgo_t x268 = x266.algo;
  size_t x269 = (size_t)0;
  cudnnGetConvolutionBackwardFilterWorkspaceSize(x7, x55, x52, x54, x98, x268, &x269);
  float* x270 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x270, (size_t)x269));
  float* x271 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x271, (size_t)(4194304 * sizeof(float))));
  cudnnConvolutionBwdDataAlgoPerf_t x272;
  int x273 = 0;
  cudnnFindConvolutionBackwardDataAlgorithm(x7, x98, x52, x54, x55, 1, &x273, &x272);
  cudnnConvolutionBwdDataAlgo_t x274 = x272.algo;
  size_t x275 = (size_t)0;
  cudnnGetConvolutionBackwardDataWorkspaceSize(x7, x98, x52, x54, x55, x274, &x275);
  float* x276 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x276, (size_t)x275));
  float* x277 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x277, (size_t)(4194304 * sizeof(float))));
  float* x278 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x278, (size_t)(73984 * sizeof(float))));
  cudnnConvolutionBwdFilterAlgoPerf_t x279;
  int x280 = 0;
  cudnnFindConvolutionBackwardFilterAlgorithm(x7, x55, x55, x54, x53, 1, &x280, &x279);
  cudnnConvolutionBwdFilterAlgo_t x281 = x279.algo;
  size_t x282 = (size_t)0;
  cudnnGetConvolutionBackwardFilterWorkspaceSize(x7, x55, x55, x54, x53, x281, &x282);
  float* x283 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x283, (size_t)x282));
  float* x284 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x284, (size_t)(4194304 * sizeof(float))));
  cudnnConvolutionBwdDataAlgoPerf_t x285;
  int x286 = 0;
  cudnnFindConvolutionBackwardDataAlgorithm(x7, x53, x55, x54, x55, 1, &x286, &x285);
  cudnnConvolutionBwdDataAlgo_t x287 = x285.algo;
  size_t x288 = (size_t)0;
  cudnnGetConvolutionBackwardDataWorkspaceSize(x7, x53, x55, x54, x55, x287, &x288);
  float* x289 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x289, (size_t)x288));
  float* x290 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x290, (size_t)(4194304 * sizeof(float))));
  float* x291 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x291, (size_t)(73984 * sizeof(float))));
  cudnnConvolutionBwdFilterAlgoPerf_t x292;
  int x293 = 0;
  cudnnFindConvolutionBackwardFilterAlgorithm(x7, x55, x55, x54, x53, 1, &x293, &x292);
  cudnnConvolutionBwdFilterAlgo_t x294 = x292.algo;
  size_t x295 = (size_t)0;
  cudnnGetConvolutionBackwardFilterWorkspaceSize(x7, x55, x55, x54, x53, x294, &x295);
  float* x296 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x296, (size_t)x295));
  float* x297 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x297, (size_t)(4194304 * sizeof(float))));
  cudnnConvolutionBwdDataAlgoPerf_t x298;
  int x299 = 0;
  cudnnFindConvolutionBackwardDataAlgorithm(x7, x53, x55, x54, x55, 1, &x299, &x298);
  cudnnConvolutionBwdDataAlgo_t x300 = x298.algo;
  size_t x301 = (size_t)0;
  cudnnGetConvolutionBackwardDataWorkspaceSize(x7, x53, x55, x54, x55, x300, &x301);
  float* x302 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x302, (size_t)x301));
  float* x303 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x303, (size_t)(4194304 * sizeof(float))));
  float* x304 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x304, (size_t)(73984 * sizeof(float))));
  cudnnConvolutionBwdFilterAlgoPerf_t x305;
  int x306 = 0;
  cudnnFindConvolutionBackwardFilterAlgorithm(x7, x55, x55, x54, x53, 1, &x306, &x305);
  cudnnConvolutionBwdFilterAlgo_t x307 = x305.algo;
  size_t x308 = (size_t)0;
  cudnnGetConvolutionBackwardFilterWorkspaceSize(x7, x55, x55, x54, x53, x307, &x308);
  float* x309 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x309, (size_t)x308));
  float* x310 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x310, (size_t)(4194304 * sizeof(float))));
  cudnnConvolutionBwdDataAlgoPerf_t x311;
  int x312 = 0;
  cudnnFindConvolutionBackwardDataAlgorithm(x7, x53, x55, x54, x55, 1, &x312, &x311);
  cudnnConvolutionBwdDataAlgo_t x313 = x311.algo;
  size_t x314 = (size_t)0;
  cudnnGetConvolutionBackwardDataWorkspaceSize(x7, x53, x55, x54, x55, x313, &x314);
  float* x315 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x315, (size_t)x314));
  float* x316 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x316, (size_t)(4194304 * sizeof(float))));
  float* x317 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x317, (size_t)(73984 * sizeof(float))));
  cudnnConvolutionBwdFilterAlgoPerf_t x318;
  int x319 = 0;
  cudnnFindConvolutionBackwardFilterAlgorithm(x7, x55, x55, x54, x53, 1, &x319, &x318);
  cudnnConvolutionBwdFilterAlgo_t x320 = x318.algo;
  size_t x321 = (size_t)0;
  cudnnGetConvolutionBackwardFilterWorkspaceSize(x7, x55, x55, x54, x53, x320, &x321);
  float* x322 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x322, (size_t)x321));
  float* x323 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x323, (size_t)(4194304 * sizeof(float))));
  cudnnConvolutionBwdDataAlgoPerf_t x324;
  int x325 = 0;
  cudnnFindConvolutionBackwardDataAlgorithm(x7, x53, x55, x54, x55, 1, &x325, &x324);
  cudnnConvolutionBwdDataAlgo_t x326 = x324.algo;
  size_t x327 = (size_t)0;
  cudnnGetConvolutionBackwardDataWorkspaceSize(x7, x53, x55, x54, x55, x326, &x327);
  float* x328 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x328, (size_t)x327));
  float* x329 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x329, (size_t)(4194304 * sizeof(float))));
  float* x330 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x330, (size_t)(73984 * sizeof(float))));
  cudnnConvolutionBwdFilterAlgoPerf_t x331;
  int x332 = 0;
  cudnnFindConvolutionBackwardFilterAlgorithm(x7, x52, x55, x54, x53, 1, &x332, &x331);
  cudnnConvolutionBwdFilterAlgo_t x333 = x331.algo;
  size_t x334 = (size_t)0;
  cudnnGetConvolutionBackwardFilterWorkspaceSize(x7, x52, x55, x54, x53, x333, &x334);
  float* x335 = (float*)malloc(0 * sizeof(float));
  CUDA_CALL(cudaMalloc(&x335, (size_t)x334));

  cudaEvent_t start_event;
  cudaEvent_t finish_event;
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  CUDA_CALL(cudaEventCreate(&start_event));
  CUDA_CALL(cudaEventCreate(&finish_event));
  CUDA_CALL(cudaEventRecord(start_event));
  while (x50 != 40) {
    float x345 = 1.0;
    float x346 = 0.0;
    cudnnConvolutionForward(x7, &x345, x52, x51, x53, x8, x54, x59, x61, x60, &x346, x55, x56);
    x63<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x56, x62, 4194304);
    float x347 = 1.0;
    float x348 = 0.0;
    cudnnConvolutionForward(x7, &x347, x55, x62, x53, x17, x54, x73, x75, x74, &x348, x55, x70);
    x63<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x70, x76, 4194304);
    float x349 = 1.0;
    float x350 = 0.0;
    cudnnConvolutionForward(x7, &x349, x55, x76, x53, x20, x54, x80, x82, x81, &x350, x55, x77);
    x63<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x77, x83, 4194304);
    float x351 = 1.0;
    float x352 = 0.0;
    cudnnConvolutionForward(x7, &x351, x55, x83, x53, x23, x54, x87, x89, x88, &x352, x55, x84);
    x63<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x84, x90, 4194304);
    float x353 = 1.0;
    float x354 = 0.0;
    cudnnConvolutionForward(x7, &x353, x55, x90, x53, x26, x54, x94, x96, x95, &x354, x55, x91);
    x63<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x91, x97, 4194304);
    float x355 = 1.0;
    float x356 = 0.0;
    cudnnConvolutionForward(x7, &x355, x55, x97, x98, x29, x54, x102, x104, x103, &x356, x52, x99);
    x63<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x99, x105, 16384);
    float x357 = 1.0;
    float x358 = 0.0;
    cudnnConvolutionForward(x7, &x357, x52, x105, x53, x32, x54, x109, x111, x110, &x358, x55, x106);
    x63<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x106, x112, 4194304);
    float x359 = 1.0;
    float x360 = 0.0;
    cudnnConvolutionForward(x7, &x359, x55, x112, x53, x35, x54, x116, x118, x117, &x360, x55, x113);
    x63<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x113, x119, 4194304);
    float x361 = 1.0;
    float x362 = 0.0;
    cudnnConvolutionForward(x7, &x361, x55, x119, x53, x38, x54, x123, x125, x124, &x362, x55, x120);
    x63<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x120, x126, 4194304);
    float x363 = 1.0;
    float x364 = 0.0;
    cudnnConvolutionForward(x7, &x363, x55, x126, x53, x41, x54, x130, x132, x131, &x364, x55, x127);
    x63<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x127, x133, 4194304);
    float x365 = 1.0;
    float x366 = 0.0;
    cudnnConvolutionForward(x7, &x365, x55, x133, x53, x44, x54, x137, x139, x138, &x366, x55, x134);
    x63<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x134, x140, 4194304);
    float x367 = 1.0;
    float x368 = 0.0;
    cudnnConvolutionForward(x7, &x367, x55, x140, x98, x47, x54, x144, x146, x145, &x368, x52, x141);
    x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x147, 0, 4194304);
    x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x148, 0, 4194304);
    x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x149, 0, 4194304);
    x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x150, 0, 4194304);
    x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x151, 0, 4194304);
    x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x152, 0, 4194304);
    x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x153, 0, 4194304);
    x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x154, 0, 4194304);
    x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x155, 0, 4194304);
    x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x156, 0, 4194304);
    x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x157, 0, 16384);
    x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x158, 0, 16384);
    x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x159, 0, 4194304);
    x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x160, 0, 4194304);
    x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x161, 0, 4194304);
    x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x162, 0, 4194304);
    x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x163, 0, 4194304);
    x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x164, 0, 4194304);
    x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x165, 0, 4194304);
    x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x166, 0, 4194304);
    x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x167, 0, 4194304);
    x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x168, 0, 4194304);
    x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x169, 0, 16384);
    x10<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x170, 1, 16384);
    x172<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x170, x141, x171, 16384);
    // end computing RELU_GRAD on GPU for size 16384 and type Float at device (pre-rename) x39 with left_operand x1495 and right_operand x1158
    // begin computing ACCUM on GPU for size 16384 and type Float at device (pre-rename) x39 with base_operand x1482 and addition_operand x1508
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x169, x171, 16384);
    // end allocating gpu array for convolution backward filter workspace
    // begin convolution backward filter pass
    float x369 = 1.0;
    float x370 = 0.0;
    CUDNNCHECK(cudnnConvolutionBackwardFilter(x7, &x369, x55, x140, x52, x169, x54, x190, x192, x191, &x370, x98, x187));
    // end convolution backward filter pass
    cudaStreamSynchronize(0);
    ncclAllReduce(x187, x187, (size_t)73984, ncclFloat32, ncclSum, x4, x5);
    CUDA_CALL(cudaStreamSynchronize(x5));
    // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x39 with base_operand x482 and addition_operand x1592
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x48, x187, 73984);
    // end allocating gpu array for convolution backward data workspace
    // begin convolution backward data pass
    float x371 = 1.0;
    float x372 = 0.0;
    CUDNNCHECK(cudnnConvolutionBackwardData(x7, &x371, x98, x47, x52, x169, x54, x196, x198, x197, &x372, x55, x193));
    // end convolution backward data pass
    // begin computing ACCUM on GPU for size 4194304 and type Float at device (pre-rename) x39 with base_operand x1469 and addition_operand x1644
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x168, x193, 4194304);
    x172<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x168, x134, x199, 4194304);
    // end computing RELU_GRAD on GPU for size 4194304 and type Float at device (pre-rename) x39 with left_operand x1469 and right_operand x1107
    // begin computing ACCUM on GPU for size 4194304 and type Float at device (pre-rename) x39 with base_operand x1456 and addition_operand x1689
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x167, x199, 4194304);
    // end allocating gpu array for convolution backward filter workspace
    // begin convolution backward filter pass
    float x373 = 1.0;
    float x374 = 0.0;
    CUDNNCHECK(cudnnConvolutionBackwardFilter(x7, &x373, x55, x133, x55, x167, x54, x203, x205, x204, &x374, x53, x200));
    // end convolution backward filter pass
    cudaStreamSynchronize(0);
    ncclAllReduce(x200, x200, (size_t)73984, ncclFloat32, ncclSum, x4, x5);
    CUDA_CALL(cudaStreamSynchronize(x5));
    // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x39 with base_operand x446 and addition_operand x1709
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x45, x200, 73984);
    // end allocating gpu array for convolution backward data workspace
    // begin convolution backward data pass
    float x375 = 1.0;
    float x376 = 0.0;
    CUDNNCHECK(cudnnConvolutionBackwardData(x7, &x375, x53, x44, x55, x167, x54, x209, x211, x210, &x376, x55, x206));
    // end convolution backward data pass
    // begin computing ACCUM on GPU for size 4194304 and type Float at device (pre-rename) x39 with base_operand x1443 and addition_operand x1761
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x166, x206, 4194304);
    x172<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x166, x127, x212, 4194304);
    // end computing RELU_GRAD on GPU for size 4194304 and type Float at device (pre-rename) x39 with left_operand x1443 and right_operand x1056
    // begin computing ACCUM on GPU for size 4194304 and type Float at device (pre-rename) x39 with base_operand x1430 and addition_operand x1806
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x165, x212, 4194304);
    // end allocating gpu array for convolution backward filter workspace
    // begin convolution backward filter pass
    float x377 = 1.0;
    float x378 = 0.0;
    CUDNNCHECK(cudnnConvolutionBackwardFilter(x7, &x377, x55, x126, x55, x165, x54, x216, x218, x217, &x378, x53, x213));
    // end convolution backward filter pass
    cudaStreamSynchronize(0);
    ncclAllReduce(x213, x213, (size_t)73984, ncclFloat32, ncclSum, x4, x5);
    CUDA_CALL(cudaStreamSynchronize(x5));
    // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x39 with base_operand x410 and addition_operand x1826
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x42, x213, 73984);
    // end allocating gpu array for convolution backward data workspace
    // begin convolution backward data pass
    float x379 = 1.0;
    float x380 = 0.0;
    CUDNNCHECK(cudnnConvolutionBackwardData(x7, &x379, x53, x41, x55, x165, x54, x222, x224, x223, &x380, x55, x219));
    // end convolution backward data pass
    // begin computing ACCUM on GPU for size 4194304 and type Float at device (pre-rename) x39 with base_operand x1417 and addition_operand x1878
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x164, x219, 4194304);
    x172<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x164, x120, x225, 4194304);
    // end computing RELU_GRAD on GPU for size 4194304 and type Float at device (pre-rename) x39 with left_operand x1417 and right_operand x1005
    // begin computing ACCUM on GPU for size 4194304 and type Float at device (pre-rename) x39 with base_operand x1404 and addition_operand x1923
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x163, x225, 4194304);
    // end allocating gpu array for convolution backward filter workspace
    // begin convolution backward filter pass
    float x381 = 1.0;
    float x382 = 0.0;
    CUDNNCHECK(cudnnConvolutionBackwardFilter(x7, &x381, x55, x119, x55, x163, x54, x229, x231, x230, &x382, x53, x226));
    // end convolution backward filter pass
    cudaStreamSynchronize(0);
    ncclAllReduce(x226, x226, (size_t)73984, ncclFloat32, ncclSum, x4, x5);
    CUDA_CALL(cudaStreamSynchronize(x5));
    // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x39 with base_operand x374 and addition_operand x1943
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x39, x226, 73984);
    // end allocating gpu array for convolution backward data workspace
    // begin convolution backward data pass
    float x383 = 1.0;
    float x384 = 0.0;
    CUDNNCHECK(cudnnConvolutionBackwardData(x7, &x383, x53, x38, x55, x163, x54, x235, x237, x236, &x384, x55, x232));
    // end convolution backward data pass
    // begin computing ACCUM on GPU for size 4194304 and type Float at device (pre-rename) x39 with base_operand x1391 and addition_operand x1995
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x162, x232, 4194304);
    x172<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x162, x113, x238, 4194304);
    // end computing RELU_GRAD on GPU for size 4194304 and type Float at device (pre-rename) x39 with left_operand x1391 and right_operand x954
    // begin computing ACCUM on GPU for size 4194304 and type Float at device (pre-rename) x39 with base_operand x1378 and addition_operand x2040
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x161, x238, 4194304);
    // end allocating gpu array for convolution backward filter workspace
    // begin convolution backward filter pass
    float x385 = 1.0;
    float x386 = 0.0;
    CUDNNCHECK(cudnnConvolutionBackwardFilter(x7, &x385, x55, x112, x55, x161, x54, x242, x244, x243, &x386, x53, x239));
    // end convolution backward filter pass
    cudaStreamSynchronize(0);
    ncclAllReduce(x239, x239, (size_t)73984, ncclFloat32, ncclSum, x4, x5);
    CUDA_CALL(cudaStreamSynchronize(x5));
    // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x39 with base_operand x338 and addition_operand x2060
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x36, x239, 73984);
    // end allocating gpu array for convolution backward data workspace
    // begin convolution backward data pass
    float x387 = 1.0;
    float x388 = 0.0;
    CUDNNCHECK(cudnnConvolutionBackwardData(x7, &x387, x53, x35, x55, x161, x54, x248, x250, x249, &x388, x55, x245));
    // end convolution backward data pass
    // begin computing ACCUM on GPU for size 4194304 and type Float at device (pre-rename) x39 with base_operand x1365 and addition_operand x2112
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x160, x245, 4194304);
    x172<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x160, x106, x251, 4194304);
    // end computing RELU_GRAD on GPU for size 4194304 and type Float at device (pre-rename) x39 with left_operand x1365 and right_operand x903
    // begin computing ACCUM on GPU for size 4194304 and type Float at device (pre-rename) x39 with base_operand x1352 and addition_operand x2157
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x159, x251, 4194304);
    // end allocating gpu array for convolution backward filter workspace
    // begin convolution backward filter pass
    float x389 = 1.0;
    float x390 = 0.0;
    CUDNNCHECK(cudnnConvolutionBackwardFilter(x7, &x389, x52, x105, x55, x159, x54, x255, x257, x256, &x390, x53, x252));
    // end convolution backward filter pass
    cudaStreamSynchronize(0);
    ncclAllReduce(x252, x252, (size_t)73984, ncclFloat32, ncclSum, x4, x5);
    CUDA_CALL(cudaStreamSynchronize(x5));
    // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x39 with base_operand x302 and addition_operand x2177
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x33, x252, 73984);
    // end allocating gpu array for convolution backward data workspace
    // begin convolution backward data pass
    float x391 = 1.0;
    float x392 = 0.0;
    CUDNNCHECK(cudnnConvolutionBackwardData(x7, &x391, x53, x32, x55, x159, x54, x261, x263, x262, &x392, x52, x258));
    // end convolution backward data pass
    // begin computing ACCUM on GPU for size 16384 and type Float at device (pre-rename) x39 with base_operand x1339 and addition_operand x2229
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x158, x258, 16384);
    x172<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x158, x99, x264, 16384);
    // end computing RELU_GRAD on GPU for size 16384 and type Float at device (pre-rename) x39 with left_operand x1339 and right_operand x852
    // begin computing ACCUM on GPU for size 16384 and type Float at device (pre-rename) x39 with base_operand x1326 and addition_operand x2274
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x157, x264, 16384);
    // end allocating gpu array for convolution backward filter workspace
    // begin convolution backward filter pass
    float x393 = 1.0;
    float x394 = 0.0;
    CUDNNCHECK(cudnnConvolutionBackwardFilter(x7, &x393, x55, x97, x52, x157, x54, x268, x270, x269, &x394, x98, x265));
    // end convolution backward filter pass
    cudaStreamSynchronize(0);
    ncclAllReduce(x265, x265, (size_t)73984, ncclFloat32, ncclSum, x4, x5);
    CUDA_CALL(cudaStreamSynchronize(x5));
    // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x39 with base_operand x266 and addition_operand x2294
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x30, x265, 73984);
    // end allocating gpu array for convolution backward data workspace
    // begin convolution backward data pass
    float x395 = 1.0;
    float x396 = 0.0;
    CUDNNCHECK(cudnnConvolutionBackwardData(x7, &x395, x98, x29, x52, x157, x54, x274, x276, x275, &x396, x55, x271));
    // end convolution backward data pass
    // begin computing ACCUM on GPU for size 4194304 and type Float at device (pre-rename) x39 with base_operand x1313 and addition_operand x2346
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x156, x271, 4194304);
    x172<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x156, x91, x277, 4194304);
    // end computing RELU_GRAD on GPU for size 4194304 and type Float at device (pre-rename) x39 with left_operand x1313 and right_operand x792
    // begin computing ACCUM on GPU for size 4194304 and type Float at device (pre-rename) x39 with base_operand x1300 and addition_operand x2391
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x155, x277, 4194304);
    // end allocating gpu array for convolution backward filter workspace
    // begin convolution backward filter pass
    float x397 = 1.0;
    float x398 = 0.0;
    CUDNNCHECK(cudnnConvolutionBackwardFilter(x7, &x397, x55, x90, x55, x155, x54, x281, x283, x282, &x398, x53, x278));
    // end convolution backward filter pass
    cudaStreamSynchronize(0);
    ncclAllReduce(x278, x278, (size_t)73984, ncclFloat32, ncclSum, x4, x5);
    CUDA_CALL(cudaStreamSynchronize(x5));
    // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x39 with base_operand x230 and addition_operand x2411
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x27, x278, 73984);
    // end allocating gpu array for convolution backward data workspace
    // begin convolution backward data pass
    float x399 = 1.0;
    float x400 = 0.0;
    CUDNNCHECK(cudnnConvolutionBackwardData(x7, &x399, x53, x26, x55, x155, x54, x287, x289, x288, &x400, x55, x284));
    // end convolution backward data pass
    // begin computing ACCUM on GPU for size 4194304 and type Float at device (pre-rename) x39 with base_operand x1287 and addition_operand x2463
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x154, x284, 4194304);
    x172<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x154, x84, x290, 4194304);
    // end computing RELU_GRAD on GPU for size 4194304 and type Float at device (pre-rename) x39 with left_operand x1287 and right_operand x741
    // begin computing ACCUM on GPU for size 4194304 and type Float at device (pre-rename) x39 with base_operand x1274 and addition_operand x2508
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x153, x290, 4194304);
    // end allocating gpu array for convolution backward filter workspace
    // begin convolution backward filter pass
    float x401 = 1.0;
    float x402 = 0.0;
    CUDNNCHECK(cudnnConvolutionBackwardFilter(x7, &x401, x55, x83, x55, x153, x54, x294, x296, x295, &x402, x53, x291));
    // end convolution backward filter pass
    cudaStreamSynchronize(0);
    ncclAllReduce(x291, x291, (size_t)73984, ncclFloat32, ncclSum, x4, x5);
    CUDA_CALL(cudaStreamSynchronize(x5));
    // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x39 with base_operand x194 and addition_operand x2528
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x24, x291, 73984);
    // end allocating gpu array for convolution backward data workspace
    // begin convolution backward data pass
    float x403 = 1.0;
    float x404 = 0.0;
    CUDNNCHECK(cudnnConvolutionBackwardData(x7, &x403, x53, x23, x55, x153, x54, x300, x302, x301, &x404, x55, x297));
    // end convolution backward data pass
    // begin computing ACCUM on GPU for size 4194304 and type Float at device (pre-rename) x39 with base_operand x1261 and addition_operand x2580
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x152, x297, 4194304);
    x172<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x152, x77, x303, 4194304);
    // end computing RELU_GRAD on GPU for size 4194304 and type Float at device (pre-rename) x39 with left_operand x1261 and right_operand x690
    // begin computing ACCUM on GPU for size 4194304 and type Float at device (pre-rename) x39 with base_operand x1248 and addition_operand x2625
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x151, x303, 4194304);
    // end allocating gpu array for convolution backward filter workspace
    // begin convolution backward filter pass
    float x405 = 1.0;
    float x406 = 0.0;
    CUDNNCHECK(cudnnConvolutionBackwardFilter(x7, &x405, x55, x76, x55, x151, x54, x307, x309, x308, &x406, x53, x304));
    // end convolution backward filter pass
    cudaStreamSynchronize(0);
    ncclAllReduce(x304, x304, (size_t)73984, ncclFloat32, ncclSum, x4, x5);
    CUDA_CALL(cudaStreamSynchronize(x5));
    // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x39 with base_operand x158 and addition_operand x2645
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x21, x304, 73984);
    // end allocating gpu array for convolution backward data workspace
    // begin convolution backward data pass
    float x407 = 1.0;
    float x408 = 0.0;
    CUDNNCHECK(cudnnConvolutionBackwardData(x7, &x407, x53, x20, x55, x151, x54, x313, x315, x314, &x408, x55, x310));
    // end convolution backward data pass
    // begin computing ACCUM on GPU for size 4194304 and type Float at device (pre-rename) x39 with base_operand x1235 and addition_operand x2697
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x150, x310, 4194304);
    x172<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x150, x70, x316, 4194304);
    // end computing RELU_GRAD on GPU for size 4194304 and type Float at device (pre-rename) x39 with left_operand x1235 and right_operand x639
    // begin computing ACCUM on GPU for size 4194304 and type Float at device (pre-rename) x39 with base_operand x1222 and addition_operand x2742
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x149, x316, 4194304);
    // end allocating gpu array for convolution backward filter workspace
    // begin convolution backward filter pass
    float x409 = 1.0;
    float x410 = 0.0;
    CUDNNCHECK(cudnnConvolutionBackwardFilter(x7, &x409, x55, x62, x55, x149, x54, x320, x322, x321, &x410, x53, x317));
    // end convolution backward filter pass
    cudaStreamSynchronize(0);
    ncclAllReduce(x317, x317, (size_t)73984, ncclFloat32, ncclSum, x4, x5);
    CUDA_CALL(cudaStreamSynchronize(x5));
    // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x39 with base_operand x122 and addition_operand x2762
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x18, x317, 73984);
    // end allocating gpu array for convolution backward data workspace
    // begin convolution backward data pass
    float x411 = 1.0;
    float x412 = 0.0;
    CUDNNCHECK(cudnnConvolutionBackwardData(x7, &x411, x53, x17, x55, x149, x54, x326, x328, x327, &x412, x55, x323));
    // end convolution backward data pass
    // begin computing ACCUM on GPU for size 4194304 and type Float at device (pre-rename) x39 with base_operand x1209 and addition_operand x2814
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x148, x323, 4194304);
    x172<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x148, x56, x329, 4194304);
    // end computing RELU_GRAD on GPU for size 4194304 and type Float at device (pre-rename) x39 with left_operand x1209 and right_operand x559
    // begin computing ACCUM on GPU for size 4194304 and type Float at device (pre-rename) x39 with base_operand x1196 and addition_operand x2859
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x147, x329, 4194304);
    // end allocating gpu array for convolution backward filter workspace
    // begin convolution backward filter pass
    float x413 = 1.0;
    float x414 = 0.0;
    CUDNNCHECK(cudnnConvolutionBackwardFilter(x7, &x413, x52, x51, x55, x147, x54, x333, x335, x334, &x414, x53, x330));
    // end convolution backward filter pass
    cudaStreamSynchronize(0);
    ncclAllReduce(x330, x330, (size_t)73984, ncclFloat32, ncclSum, x4, x5);
    CUDA_CALL(cudaStreamSynchronize(x5));
    // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x39 with base_operand x59 and addition_operand x2879
    CUDA_CALL(cudaSetDevice(x6));
    x180<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x9, x330, 73984);
    // end computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x39 with base_operand x59 and addition_operand x2879
    // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x39 with weight x49, grad x59, and momentum x99
    CUDA_CALL(cudaSetDevice(x6));
    x336<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x8, x9, x16, 73984);
    // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x39 with weight x49, grad x59, and momentum x99
    // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x39 with weight x112, grad x122, and momentum x135
    CUDA_CALL(cudaSetDevice(x6));
    x336<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x17, x18, x19, 73984);
    // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x39 with weight x112, grad x122, and momentum x135
    // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x39 with weight x148, grad x158, and momentum x171
    CUDA_CALL(cudaSetDevice(x6));
    x336<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x20, x21, x22, 73984);
    // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x39 with weight x148, grad x158, and momentum x171
    // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x39 with weight x184, grad x194, and momentum x207
    CUDA_CALL(cudaSetDevice(x6));
    x336<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x23, x24, x25, 73984);
    // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x39 with weight x184, grad x194, and momentum x207
    // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x39 with weight x220, grad x230, and momentum x243
    CUDA_CALL(cudaSetDevice(x6));
    x336<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x26, x27, x28, 73984);
    // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x39 with weight x220, grad x230, and momentum x243
    // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x39 with weight x256, grad x266, and momentum x279
    CUDA_CALL(cudaSetDevice(x6));
    x336<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x29, x30, x31, 73984);
    // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x39 with weight x256, grad x266, and momentum x279
    // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x39 with weight x292, grad x302, and momentum x315
    CUDA_CALL(cudaSetDevice(x6));
    x336<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x32, x33, x34, 73984);
    // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x39 with weight x292, grad x302, and momentum x315
    // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x39 with weight x328, grad x338, and momentum x351
    CUDA_CALL(cudaSetDevice(x6));
    x336<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x35, x36, x37, 73984);
    // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x39 with weight x328, grad x338, and momentum x351
    // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x39 with weight x364, grad x374, and momentum x387
    CUDA_CALL(cudaSetDevice(x6));
    x336<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x38, x39, x40, 73984);
    // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x39 with weight x364, grad x374, and momentum x387
    // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x39 with weight x400, grad x410, and momentum x423
    CUDA_CALL(cudaSetDevice(x6));
    x336<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x41, x42, x43, 73984);
    // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x39 with weight x400, grad x410, and momentum x423
    // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x39 with weight x436, grad x446, and momentum x459
    CUDA_CALL(cudaSetDevice(x6));
    x336<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x44, x45, x46, 73984);
    // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x39 with weight x436, grad x446, and momentum x459
    // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x39 with weight x472, grad x482, and momentum x495
    CUDA_CALL(cudaSetDevice(x6));
    x336<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x47, x48, x49, 73984);
    // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x39 with weight x472, grad x482, and momentum x495
    x50 = x50 + 1;
  }
  CUDA_CALL(cudaEventRecord(finish_event));
  CUDA_CALL(cudaEventSynchronize(finish_event));
  float runningtime = 0.0;
  CUDA_CALL(cudaEventElapsedTime(&runningtime, start_event, finish_event));
  float avgtime = 0.0;
  MPI_Allreduce(&runningtime, &avgtime, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  avgtime = (avgtime/x1)/1000;
  if (0 == x2) {
    printf("%f", avgtime);
  }
  NCCLCHECK(ncclCommDestroy(x4));
  CUDNNCHECK(cudnnDestroy(x7));
  MPICHECK(MPI_Finalize());
}
/*****************************************
End of C Generated Code
*******************************************/
int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("usage: %s <arg>\n", argv[0]);
    return 0;
  }
  Snippet(atoi(argv[1]));
  return 0;
}
