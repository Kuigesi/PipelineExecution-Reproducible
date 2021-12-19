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
__global__ void x16(float* x17, float x18, int x19) {
  // begin generating kernel function for FILL of type Float
  int x20 = gridDim.x * blockDim.x;
  int x21 = threadIdx.x + blockIdx.x * blockDim.x;
  while (x21 < x19) {
    x17[x21] = x18;
    x21 = x21 + x20;
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
__global__ void x120(float* x121, float* x122, float* x123, int x124) {
  // begin generating kernel function for RELU_GRAD of type Float
  int x125 = gridDim.x * blockDim.x;
  int x126 = threadIdx.x + blockIdx.x * blockDim.x;
  while (x126 < x124) {
    int x127 = x126;
    x123[x127] = x122[x127] > 0.0 ? x121[x127] : 0.0;
    x126 = x126 + x125;
  }
  // end generating kernel function for RELU_GRAD of type Float
}
__global__ void x128(float* x129, float* x130, int x131) {
  // begin generating kernel function for ACCUM of type Float
  int x132 = gridDim.x * blockDim.x;
  int x133 = threadIdx.x + blockIdx.x * blockDim.x;
  while (x133 < x131) {
    int x134 = x133;
    x129[x134] = x129[x134] + x130[x134];
    x133 = x133 + x132;
  }
  // end generating kernel function for ACCUM of type Float
}
__global__ void x211(float* x212, float* x213, float* x214, int x215) {
  // begin generating kernel function for SGD of type Float
  int x216 = gridDim.x * blockDim.x;
  int x217 = threadIdx.x + blockIdx.x * blockDim.x;
  while (x217 < x215) {
    int x218 = x217;
    float x219 = x214[x218] * 0.5 + x213[x218];
    x212[x218] = x212[x218] - x219 * 1.0E-4;
    x214[x218] = x219;
    x217 = x217 + x216;
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
  // begin setting up the local MPI/NCCL environment
  MPI_Comm x6;
  int x7 = x2;
  MPICHECK(MPI_Comm_split(MPI_COMM_WORLD, x7 / 2, x7, &x6));
  int x8 = 0;
  int x9 = 0;
  MPICHECK(MPI_Comm_rank(x6, &x9));
  MPICHECK(MPI_Comm_size(x6, &x8));
  ncclUniqueId x10;
  NCCLCHECK(ncclGetUniqueId(&x10));
  MPICHECK(MPI_Bcast(&x10, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, x6));
  ncclComm_t x11;
  NCCLCHECK(ncclCommInitRank(&x11, x8, x10, x9));
  int x12 = x2;
  // end setting up the local MPI/NCCL environment
  // end setting up the MPI/NCCL environment
  // begin setting up the CUDNN environment
  cudnnHandle_t x13;
  CUDNNCHECK(cudnnCreate(&x13));
  // end setting up the CUDNN environment

  cudaEvent_t start_event;
  cudaEvent_t finish_event;
  if (x12 >= 0 && x12 < 2) {
    // begin initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x14 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x14, (size_t)(73984 * sizeof(float))));
    // end initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x15 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x15, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x15, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x22 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x22, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x22, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x23 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x23, (size_t)(73984 * sizeof(float))));
    // end initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x24 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x24, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x24, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x25 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x25, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x25, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x26 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x26, (size_t)(73984 * sizeof(float))));
    // end initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x27 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x27, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x27, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x28 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x28, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x28, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x29 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x29, (size_t)(73984 * sizeof(float))));
    // end initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x30 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x30, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x30, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x31 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x31, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x31, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x32 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x32, (size_t)(73984 * sizeof(float))));
    // end initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x33 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x33, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x33, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x34 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x34, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x34, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x35 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x35, (size_t)(73984 * sizeof(float))));
    // end initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x36 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x36, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x36, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x37 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x37, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x37, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x38 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x38, (size_t)(67108864 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x38, 0, 67108864);
    // end initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 262144 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x39 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x39, (size_t)(262144 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x39, 0, 262144);
    // end initializing fixed GPU array of size 262144 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x40 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x40, (size_t)(67108864 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x40, 0, 67108864);
    // end initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x41 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x41, (size_t)(67108864 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x41, 0, 67108864);
    // end initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x42 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x42, (size_t)(67108864 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x42, 0, 67108864);
    // end initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x43 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x43, (size_t)(67108864 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x43, 0, 67108864);
    // end initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x44 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x44, (size_t)(67108864 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x44, 0, 67108864);
    // end initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x45 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x45, (size_t)(67108864 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x45, 0, 67108864);
    // end initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 262144 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x46 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x46, (size_t)(262144 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x46, 0, 262144);
    // end initializing fixed GPU array of size 262144 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x47 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x47, (size_t)(67108864 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x47, 0, 67108864);
    // end initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x48 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x48, (size_t)(67108864 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x48, 0, 67108864);
    // end initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x49 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x49, (size_t)(67108864 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x49, 0, 67108864);
    // end initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    int x50 = 0;
    float* x51 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x51, (size_t)(32768 * sizeof(float))));
    cudnnTensorDescriptor_t x52;
    cudnnCreateTensorDescriptor(&x52);
    cudnnSetTensor4dDescriptor(x52, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2, 1, 128, 128);
    cudnnFilterDescriptor_t x53;
    cudnnCreateFilterDescriptor(&x53);
    cudnnSetFilter4dDescriptor(x53, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 256, 1, 17, 17);
    cudnnConvolutionDescriptor_t x54;
    cudnnCreateConvolutionDescriptor(&x54);
    cudnnSetConvolution2dDescriptor(x54, 8, 8, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnTensorDescriptor_t x55;
    cudnnCreateTensorDescriptor(&x55);
    cudnnSetTensor4dDescriptor(x55, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2, 256, 128, 128);
    float* x56 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x56, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionFwdAlgoPerf_t x57;
    int x58 = 0;
    cudnnFindConvolutionForwardAlgorithm(x13, x52, x53, x54, x55, 1, &x58, &x57);
    cudnnConvolutionFwdAlgo_t x59 = x57.algo;
    size_t x60 = (size_t)0;
    cudnnGetConvolutionForwardWorkspaceSize(x13, x52, x53, x54, x55, x59, &x60);
    float* x61 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x61, (size_t)x60));
    float* x62 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x62, (size_t)(8388608 * sizeof(float))));
    float* x70 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x70, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionFwdAlgoPerf_t x71;
    int x72 = 0;
    cudnnFindConvolutionForwardAlgorithm(x13, x55, x53, x54, x55, 1, &x72, &x71);
    cudnnConvolutionFwdAlgo_t x73 = x71.algo;
    size_t x74 = (size_t)0;
    cudnnGetConvolutionForwardWorkspaceSize(x13, x55, x53, x54, x55, x73, &x74);
    float* x75 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x75, (size_t)x74));
    float* x76 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x76, (size_t)(8388608 * sizeof(float))));
    float* x77 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x77, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionFwdAlgoPerf_t x78;
    int x79 = 0;
    cudnnFindConvolutionForwardAlgorithm(x13, x55, x53, x54, x55, 1, &x79, &x78);
    cudnnConvolutionFwdAlgo_t x80 = x78.algo;
    size_t x81 = (size_t)0;
    cudnnGetConvolutionForwardWorkspaceSize(x13, x55, x53, x54, x55, x80, &x81);
    float* x82 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x82, (size_t)x81));
    float* x83 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x83, (size_t)(8388608 * sizeof(float))));
    float* x84 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x84, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionFwdAlgoPerf_t x85;
    int x86 = 0;
    cudnnFindConvolutionForwardAlgorithm(x13, x55, x53, x54, x55, 1, &x86, &x85);
    cudnnConvolutionFwdAlgo_t x87 = x85.algo;
    size_t x88 = (size_t)0;
    cudnnGetConvolutionForwardWorkspaceSize(x13, x55, x53, x54, x55, x87, &x88);
    float* x89 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x89, (size_t)x88));
    float* x90 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x90, (size_t)(8388608 * sizeof(float))));
    float* x91 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x91, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionFwdAlgoPerf_t x92;
    int x93 = 0;
    cudnnFindConvolutionForwardAlgorithm(x13, x55, x53, x54, x55, 1, &x93, &x92);
    cudnnConvolutionFwdAlgo_t x94 = x92.algo;
    size_t x95 = (size_t)0;
    cudnnGetConvolutionForwardWorkspaceSize(x13, x55, x53, x54, x55, x94, &x95);
    float* x96 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x96, (size_t)x95));
    float* x97 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x97, (size_t)(8388608 * sizeof(float))));
    cudnnFilterDescriptor_t x98;
    cudnnCreateFilterDescriptor(&x98);
    cudnnSetFilter4dDescriptor(x98, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 256, 17, 17);
    float* x99 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x99, (size_t)(32768 * sizeof(float))));
    cudnnConvolutionFwdAlgoPerf_t x100;
    int x101 = 0;
    cudnnFindConvolutionForwardAlgorithm(x13, x55, x98, x54, x52, 1, &x101, &x100);
    cudnnConvolutionFwdAlgo_t x102 = x100.algo;
    size_t x103 = (size_t)0;
    cudnnGetConvolutionForwardWorkspaceSize(x13, x55, x98, x54, x52, x102, &x103);
    float* x104 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x104, (size_t)x103));
    float* x105 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x105, (size_t)(32768 * sizeof(float))));
    int x106 = x12 + 2;
    float* x107 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x107, (size_t)(8388608 * sizeof(float))));
    float* x108 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x108, (size_t)(8388608 * sizeof(float))));
    float* x109 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x109, (size_t)(8388608 * sizeof(float))));
    float* x110 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x110, (size_t)(8388608 * sizeof(float))));
    float* x111 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x111, (size_t)(8388608 * sizeof(float))));
    float* x112 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x112, (size_t)(8388608 * sizeof(float))));
    float* x113 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x113, (size_t)(8388608 * sizeof(float))));
    float* x114 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x114, (size_t)(8388608 * sizeof(float))));
    float* x115 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x115, (size_t)(8388608 * sizeof(float))));
    float* x116 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x116, (size_t)(8388608 * sizeof(float))));
    float* x117 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x117, (size_t)(32768 * sizeof(float))));
    float* x118 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x118, (size_t)(32768 * sizeof(float))));
    float* x119 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x119, (size_t)(32768 * sizeof(float))));
    cudnnTensorDescriptor_t x135;
    cudnnCreateTensorDescriptor(&x135);
    cudnnSetTensor4dDescriptor(x135, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2, 256, 128, 128);
    cudnnTensorDescriptor_t x136;
    cudnnCreateTensorDescriptor(&x136);
    cudnnSetTensor4dDescriptor(x136, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2, 1, 128, 128);
    cudnnFilterDescriptor_t x137;
    cudnnCreateFilterDescriptor(&x137);
    cudnnSetFilter4dDescriptor(x137, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 256, 17, 17);
    cudnnConvolutionDescriptor_t x138;
    cudnnCreateConvolutionDescriptor(&x138);
    cudnnSetConvolution2dDescriptor(x138, 8, 8, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    float* x139 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x139, (size_t)(73984 * sizeof(float))));
    cudnnConvolutionBwdFilterAlgoPerf_t x140;
    int x141 = 0;
    cudnnFindConvolutionBackwardFilterAlgorithm(x13, x135, x136, x138, x137, 1, &x141, &x140);
    cudnnConvolutionBwdFilterAlgo_t x142 = x140.algo;
    size_t x143 = (size_t)0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(x13, x135, x136, x138, x137, x142, &x143);
    float* x144 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x144, (size_t)x143));
    float* x145 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x145, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionBwdDataAlgoPerf_t x146;
    int x147 = 0;
    cudnnFindConvolutionBackwardDataAlgorithm(x13, x137, x136, x138, x135, 1, &x147, &x146);
    cudnnConvolutionBwdDataAlgo_t x148 = x146.algo;
    size_t x149 = (size_t)0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(x13, x137, x136, x138, x135, x148, &x149);
    float* x150 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x150, (size_t)x149));
    float* x151 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x151, (size_t)(8388608 * sizeof(float))));
    cudnnFilterDescriptor_t x152;
    cudnnCreateFilterDescriptor(&x152);
    cudnnSetFilter4dDescriptor(x152, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 256, 1, 17, 17);
    float* x153 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x153, (size_t)(73984 * sizeof(float))));
    cudnnConvolutionBwdFilterAlgoPerf_t x154;
    int x155 = 0;
    cudnnFindConvolutionBackwardFilterAlgorithm(x13, x135, x135, x138, x152, 1, &x155, &x154);
    cudnnConvolutionBwdFilterAlgo_t x156 = x154.algo;
    size_t x157 = (size_t)0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(x13, x135, x135, x138, x152, x156, &x157);
    float* x158 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x158, (size_t)x157));
    float* x159 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x159, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionBwdDataAlgoPerf_t x160;
    int x161 = 0;
    cudnnFindConvolutionBackwardDataAlgorithm(x13, x152, x135, x138, x135, 1, &x161, &x160);
    cudnnConvolutionBwdDataAlgo_t x162 = x160.algo;
    size_t x163 = (size_t)0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(x13, x152, x135, x138, x135, x162, &x163);
    float* x164 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x164, (size_t)x163));
    float* x165 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x165, (size_t)(8388608 * sizeof(float))));
    float* x166 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x166, (size_t)(73984 * sizeof(float))));
    cudnnConvolutionBwdFilterAlgoPerf_t x167;
    int x168 = 0;
    cudnnFindConvolutionBackwardFilterAlgorithm(x13, x135, x135, x138, x152, 1, &x168, &x167);
    cudnnConvolutionBwdFilterAlgo_t x169 = x167.algo;
    size_t x170 = (size_t)0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(x13, x135, x135, x138, x152, x169, &x170);
    float* x171 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x171, (size_t)x170));
    float* x172 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x172, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionBwdDataAlgoPerf_t x173;
    int x174 = 0;
    cudnnFindConvolutionBackwardDataAlgorithm(x13, x152, x135, x138, x135, 1, &x174, &x173);
    cudnnConvolutionBwdDataAlgo_t x175 = x173.algo;
    size_t x176 = (size_t)0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(x13, x152, x135, x138, x135, x175, &x176);
    float* x177 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x177, (size_t)x176));
    float* x178 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x178, (size_t)(8388608 * sizeof(float))));
    float* x179 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x179, (size_t)(73984 * sizeof(float))));
    cudnnConvolutionBwdFilterAlgoPerf_t x180;
    int x181 = 0;
    cudnnFindConvolutionBackwardFilterAlgorithm(x13, x135, x135, x138, x152, 1, &x181, &x180);
    cudnnConvolutionBwdFilterAlgo_t x182 = x180.algo;
    size_t x183 = (size_t)0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(x13, x135, x135, x138, x152, x182, &x183);
    float* x184 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x184, (size_t)x183));
    float* x185 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x185, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionBwdDataAlgoPerf_t x186;
    int x187 = 0;
    cudnnFindConvolutionBackwardDataAlgorithm(x13, x152, x135, x138, x135, 1, &x187, &x186);
    cudnnConvolutionBwdDataAlgo_t x188 = x186.algo;
    size_t x189 = (size_t)0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(x13, x152, x135, x138, x135, x188, &x189);
    float* x190 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x190, (size_t)x189));
    float* x191 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x191, (size_t)(8388608 * sizeof(float))));
    float* x192 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x192, (size_t)(73984 * sizeof(float))));
    cudnnConvolutionBwdFilterAlgoPerf_t x193;
    int x194 = 0;
    cudnnFindConvolutionBackwardFilterAlgorithm(x13, x135, x135, x138, x152, 1, &x194, &x193);
    cudnnConvolutionBwdFilterAlgo_t x195 = x193.algo;
    size_t x196 = (size_t)0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(x13, x135, x135, x138, x152, x195, &x196);
    float* x197 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x197, (size_t)x196));
    float* x198 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x198, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionBwdDataAlgoPerf_t x199;
    int x200 = 0;
    cudnnFindConvolutionBackwardDataAlgorithm(x13, x152, x135, x138, x135, 1, &x200, &x199);
    cudnnConvolutionBwdDataAlgo_t x201 = x199.algo;
    size_t x202 = (size_t)0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(x13, x152, x135, x138, x135, x201, &x202);
    float* x203 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x203, (size_t)x202));
    float* x204 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x204, (size_t)(8388608 * sizeof(float))));
    float* x205 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x205, (size_t)(73984 * sizeof(float))));
    cudnnConvolutionBwdFilterAlgoPerf_t x206;
    int x207 = 0;
    cudnnFindConvolutionBackwardFilterAlgorithm(x13, x136, x135, x138, x152, 1, &x207, &x206);
    cudnnConvolutionBwdFilterAlgo_t x208 = x206.algo;
    size_t x209 = (size_t)0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(x13, x136, x135, x138, x152, x208, &x209);
    float* x210 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x210, (size_t)x209));

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    CUDA_CALL(cudaEventCreate(&start_event));
    CUDA_CALL(cudaEventCreate(&finish_event));
    CUDA_CALL(cudaEventRecord(start_event));
    while (x50 != 5) {
      int x220 = 0;
      while (x220 != 8) {
        int x221 = x220;
        int x222 = 32768 * x221;
        cudaMemcpy(x39 + x222, x51, (size_t)(32768 * sizeof(float)), cudaMemcpyDeviceToDevice);
        float x223 = 1.0;
        float x224 = 0.0;
        cudnnConvolutionForward(x13, &x223, x52, x51, x53, x14, x54, x59, x61, x60, &x224, x55, x56);
        int x225 = 8388608 * x221;
        cudaMemcpy(x45 + x225, x56, (size_t)(8388608 * sizeof(float)), cudaMemcpyDeviceToDevice);
        x63<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x56, x62, 8388608);
        cudaMemcpy(x44 + x225, x62, (size_t)(8388608 * sizeof(float)), cudaMemcpyDeviceToDevice);
        float x226 = 1.0;
        float x227 = 0.0;
        cudnnConvolutionForward(x13, &x226, x55, x62, x53, x23, x54, x73, x75, x74, &x227, x55, x70);
        cudaMemcpy(x49 + x225, x70, (size_t)(8388608 * sizeof(float)), cudaMemcpyDeviceToDevice);
        x63<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x70, x76, 8388608);
        cudaMemcpy(x43 + x225, x76, (size_t)(8388608 * sizeof(float)), cudaMemcpyDeviceToDevice);
        float x228 = 1.0;
        float x229 = 0.0;
        cudnnConvolutionForward(x13, &x228, x55, x76, x53, x26, x54, x80, x82, x81, &x229, x55, x77);
        cudaMemcpy(x38 + x225, x77, (size_t)(8388608 * sizeof(float)), cudaMemcpyDeviceToDevice);
        x63<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x77, x83, 8388608);
        cudaMemcpy(x40 + x225, x83, (size_t)(8388608 * sizeof(float)), cudaMemcpyDeviceToDevice);
        float x230 = 1.0;
        float x231 = 0.0;
        cudnnConvolutionForward(x13, &x230, x55, x83, x53, x29, x54, x87, x89, x88, &x231, x55, x84);
        cudaMemcpy(x41 + x225, x84, (size_t)(8388608 * sizeof(float)), cudaMemcpyDeviceToDevice);
        x63<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x84, x90, 8388608);
        cudaMemcpy(x47 + x225, x90, (size_t)(8388608 * sizeof(float)), cudaMemcpyDeviceToDevice);
        float x232 = 1.0;
        float x233 = 0.0;
        cudnnConvolutionForward(x13, &x232, x55, x90, x53, x32, x54, x94, x96, x95, &x233, x55, x91);
        cudaMemcpy(x42 + x225, x91, (size_t)(8388608 * sizeof(float)), cudaMemcpyDeviceToDevice);
        x63<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x91, x97, 8388608);
        cudaMemcpy(x48 + x225, x97, (size_t)(8388608 * sizeof(float)), cudaMemcpyDeviceToDevice);
        float x234 = 1.0;
        float x235 = 0.0;
        cudnnConvolutionForward(x13, &x234, x55, x97, x98, x35, x54, x102, x104, x103, &x235, x52, x99);
        cudaMemcpy(x46 + x222, x99, (size_t)(32768 * sizeof(float)), cudaMemcpyDeviceToDevice);
        x63<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x99, x105, 32768);
        // end computing RELU on GPU for size 32768 and type Float at device (pre-rename) x66 with operand x909
        cudaStreamSynchronize(0);
        NCCLCHECK(ncclSend(x105, (size_t)32768, ncclFloat32, x106, x4, x5));
        x220 = x220 + 1;
      }
      int x236 = 0;
      while (x236 != 8) {
        int x237 = x236;
        x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x107, 0, 8388608);
        x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x108, 0, 8388608);
        x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x109, 0, 8388608);
        x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x110, 0, 8388608);
        x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x111, 0, 8388608);
        x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x112, 0, 8388608);
        x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x113, 0, 8388608);
        x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x114, 0, 8388608);
        x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x115, 0, 8388608);
        x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x116, 0, 8388608);
        x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x117, 0, 32768);
        int x238 = 8388608 * x237;
        int x239 = 32768 * x237;
        x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x118, 0, 32768);
        ncclRecv(x118, (size_t)32768, ncclFloat32, x106, x4, x5);
        cudaStreamSynchronize(x5);
        x120<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x118, x46 + x239, x119, 32768);
        // end computing RELU_GRAD on GPU for size 32768 and type Float at device (pre-rename) x66 with left_operand x1140 and right_operand x1133
        // begin computing ACCUM on GPU for size 32768 and type Float at device (pre-rename) x66 with base_operand x1113 and addition_operand x1159
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x117, x119, 32768);
        // end allocating gpu array for convolution backward filter workspace
        // begin convolution backward filter pass
        float x240 = 1.0;
        float x241 = 0.0;
        CUDNNCHECK(cudnnConvolutionBackwardFilter(x13, &x240, x135, x48 + x238, x136, x117, x138, x142, x144, x143, &x241, x137, x139));
        // end convolution backward filter pass
        cudaStreamSynchronize(0);
        ncclAllReduce(x139, x139, (size_t)73984, ncclFloat32, ncclSum, x11, x5);
        CUDA_CALL(cudaStreamSynchronize(x5));
        // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x66 with base_operand x298 and addition_operand x1279
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x36, x139, 73984);
        // end allocating gpu array for convolution backward data workspace
        // begin convolution backward data pass
        float x242 = 1.0;
        float x243 = 0.0;
        CUDNNCHECK(cudnnConvolutionBackwardData(x13, &x242, x137, x35, x136, x117, x138, x148, x150, x149, &x243, x135, x145));
        // end convolution backward data pass
        // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x1100 and addition_operand x1331
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x116, x145, 8388608);
        x120<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x116, x42 + x238, x151, 8388608);
        // end computing RELU_GRAD on GPU for size 8388608 and type Float at device (pre-rename) x66 with left_operand x1100 and right_operand x1129
        // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x1087 and addition_operand x1376
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x115, x151, 8388608);
        // end allocating gpu array for convolution backward filter workspace
        // begin convolution backward filter pass
        float x244 = 1.0;
        float x245 = 0.0;
        CUDNNCHECK(cudnnConvolutionBackwardFilter(x13, &x244, x135, x47 + x238, x135, x115, x138, x156, x158, x157, &x245, x152, x153));
        // end convolution backward filter pass
        cudaStreamSynchronize(0);
        ncclAllReduce(x153, x153, (size_t)73984, ncclFloat32, ncclSum, x11, x5);
        CUDA_CALL(cudaStreamSynchronize(x5));
        // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x66 with base_operand x262 and addition_operand x1405
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x33, x153, 73984);
        // end allocating gpu array for convolution backward data workspace
        // begin convolution backward data pass
        float x246 = 1.0;
        float x247 = 0.0;
        CUDNNCHECK(cudnnConvolutionBackwardData(x13, &x246, x152, x32, x135, x115, x138, x162, x164, x163, &x247, x135, x159));
        // end convolution backward data pass
        // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x1074 and addition_operand x1457
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x114, x159, 8388608);
        x120<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x114, x41 + x238, x165, 8388608);
        // end computing RELU_GRAD on GPU for size 8388608 and type Float at device (pre-rename) x66 with left_operand x1074 and right_operand x1128
        // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x1061 and addition_operand x1502
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x113, x165, 8388608);
        // end allocating gpu array for convolution backward filter workspace
        // begin convolution backward filter pass
        float x248 = 1.0;
        float x249 = 0.0;
        CUDNNCHECK(cudnnConvolutionBackwardFilter(x13, &x248, x135, x40 + x238, x135, x113, x138, x169, x171, x170, &x249, x152, x166));
        // end convolution backward filter pass
        cudaStreamSynchronize(0);
        ncclAllReduce(x166, x166, (size_t)73984, ncclFloat32, ncclSum, x11, x5);
        CUDA_CALL(cudaStreamSynchronize(x5));
        // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x66 with base_operand x226 and addition_operand x1522
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x30, x166, 73984);
        // end allocating gpu array for convolution backward data workspace
        // begin convolution backward data pass
        float x250 = 1.0;
        float x251 = 0.0;
        CUDNNCHECK(cudnnConvolutionBackwardData(x13, &x250, x152, x29, x135, x113, x138, x175, x177, x176, &x251, x135, x172));
        // end convolution backward data pass
        // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x1048 and addition_operand x1574
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x112, x172, 8388608);
        x120<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x112, x38 + x238, x178, 8388608);
        // end computing RELU_GRAD on GPU for size 8388608 and type Float at device (pre-rename) x66 with left_operand x1048 and right_operand x1124
        // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x1035 and addition_operand x1619
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x111, x178, 8388608);
        // end allocating gpu array for convolution backward filter workspace
        // begin convolution backward filter pass
        float x252 = 1.0;
        float x253 = 0.0;
        CUDNNCHECK(cudnnConvolutionBackwardFilter(x13, &x252, x135, x43 + x238, x135, x111, x138, x182, x184, x183, &x253, x152, x179));
        // end convolution backward filter pass
        cudaStreamSynchronize(0);
        ncclAllReduce(x179, x179, (size_t)73984, ncclFloat32, ncclSum, x11, x5);
        CUDA_CALL(cudaStreamSynchronize(x5));
        // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x66 with base_operand x190 and addition_operand x1639
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x27, x179, 73984);
        // end allocating gpu array for convolution backward data workspace
        // begin convolution backward data pass
        float x254 = 1.0;
        float x255 = 0.0;
        CUDNNCHECK(cudnnConvolutionBackwardData(x13, &x254, x152, x26, x135, x111, x138, x188, x190, x189, &x255, x135, x185));
        // end convolution backward data pass
        // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x1022 and addition_operand x1691
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x110, x185, 8388608);
        x120<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x110, x49 + x238, x191, 8388608);
        // end computing RELU_GRAD on GPU for size 8388608 and type Float at device (pre-rename) x66 with left_operand x1022 and right_operand x1136
        // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x1009 and addition_operand x1736
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x109, x191, 8388608);
        // end allocating gpu array for convolution backward filter workspace
        // begin convolution backward filter pass
        float x256 = 1.0;
        float x257 = 0.0;
        CUDNNCHECK(cudnnConvolutionBackwardFilter(x13, &x256, x135, x44 + x238, x135, x109, x138, x195, x197, x196, &x257, x152, x192));
        // end convolution backward filter pass
        cudaStreamSynchronize(0);
        ncclAllReduce(x192, x192, (size_t)73984, ncclFloat32, ncclSum, x11, x5);
        CUDA_CALL(cudaStreamSynchronize(x5));
        // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x66 with base_operand x154 and addition_operand x1756
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x24, x192, 73984);
        // end allocating gpu array for convolution backward data workspace
        // begin convolution backward data pass
        float x258 = 1.0;
        float x259 = 0.0;
        CUDNNCHECK(cudnnConvolutionBackwardData(x13, &x258, x152, x23, x135, x109, x138, x201, x203, x202, &x259, x135, x198));
        // end convolution backward data pass
        // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x996 and addition_operand x1808
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x108, x198, 8388608);
        x120<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x108, x45 + x238, x204, 8388608);
        // end computing RELU_GRAD on GPU for size 8388608 and type Float at device (pre-rename) x66 with left_operand x996 and right_operand x1132
        // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x983 and addition_operand x1853
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x107, x204, 8388608);
        // end allocating gpu array for convolution backward filter workspace
        // begin convolution backward filter pass
        float x260 = 1.0;
        float x261 = 0.0;
        CUDNNCHECK(cudnnConvolutionBackwardFilter(x13, &x260, x136, x39 + x239, x135, x107, x138, x208, x210, x209, &x261, x152, x205));
        // end convolution backward filter pass
        cudaStreamSynchronize(0);
        ncclAllReduce(x205, x205, (size_t)73984, ncclFloat32, ncclSum, x11, x5);
        CUDA_CALL(cudaStreamSynchronize(x5));
        // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x66 with base_operand x91 and addition_operand x1873
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x15, x205, 73984);
        // end computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x66 with base_operand x91 and addition_operand x1873
        x236 = x236 + 1;
      }
      // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x81, grad x91, and momentum x131
      CUDA_CALL(cudaSetDevice(x12));
      x211<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x14, x15, x22, 73984);
      // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x81, grad x91, and momentum x131
      // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x144, grad x154, and momentum x167
      CUDA_CALL(cudaSetDevice(x12));
      x211<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x23, x24, x25, 73984);
      // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x144, grad x154, and momentum x167
      // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x180, grad x190, and momentum x203
      CUDA_CALL(cudaSetDevice(x12));
      x211<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x26, x27, x28, 73984);
      // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x180, grad x190, and momentum x203
      // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x216, grad x226, and momentum x239
      CUDA_CALL(cudaSetDevice(x12));
      x211<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x29, x30, x31, 73984);
      // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x216, grad x226, and momentum x239
      // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x252, grad x262, and momentum x275
      CUDA_CALL(cudaSetDevice(x12));
      x211<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x32, x33, x34, 73984);
      // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x252, grad x262, and momentum x275
      // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x288, grad x298, and momentum x311
      CUDA_CALL(cudaSetDevice(x12));
      x211<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x35, x36, x37, 73984);
      // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x288, grad x298, and momentum x311
      x50 = x50 + 1;
    }
  }
  if (x12 >= 2 && x12 < 4) {
    // begin initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x262 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x262, (size_t)(73984 * sizeof(float))));
    // end initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x263 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x263, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x263, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x264 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x264, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x264, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x265 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x265, (size_t)(73984 * sizeof(float))));
    // end initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x266 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x266, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x266, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x267 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x267, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x267, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x268 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x268, (size_t)(73984 * sizeof(float))));
    // end initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x269 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x269, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x269, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x270 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x270, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x270, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x271 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x271, (size_t)(73984 * sizeof(float))));
    // end initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x272 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x272, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x272, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x273 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x273, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x273, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x274 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x274, (size_t)(73984 * sizeof(float))));
    // end initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x275 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x275, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x275, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x276 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x276, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x276, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x277 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x277, (size_t)(73984 * sizeof(float))));
    // end initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x278 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x278, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x278, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x279 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x279, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x279, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x280 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x280, (size_t)(67108864 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x280, 0, 67108864);
    // end initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 262144 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x281 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x281, (size_t)(262144 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x281, 0, 262144);
    // end initializing fixed GPU array of size 262144 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x282 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x282, (size_t)(67108864 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x282, 0, 67108864);
    // end initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x283 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x283, (size_t)(67108864 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x283, 0, 67108864);
    // end initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x284 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x284, (size_t)(67108864 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x284, 0, 67108864);
    // end initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x285 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x285, (size_t)(67108864 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x285, 0, 67108864);
    // end initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x286 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x286, (size_t)(67108864 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x286, 0, 67108864);
    // end initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x287 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x287, (size_t)(67108864 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x287, 0, 67108864);
    // end initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x288 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x288, (size_t)(67108864 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x288, 0, 67108864);
    // end initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x289 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x289, (size_t)(67108864 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x289, 0, 67108864);
    // end initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x290 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x290, (size_t)(67108864 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x290, 0, 67108864);
    // end initializing fixed GPU array of size 67108864 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 262144 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x291 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x291, (size_t)(262144 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x291, 0, 262144);
    // end initializing fixed GPU array of size 262144 and type Float and device (pre-rename) x66
    int x292 = 0;
    float* x293 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x293, (size_t)(32768 * sizeof(float))));
    int x294 = x12 - 2;
    cudnnTensorDescriptor_t x295;
    cudnnCreateTensorDescriptor(&x295);
    cudnnSetTensor4dDescriptor(x295, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2, 1, 128, 128);
    cudnnFilterDescriptor_t x296;
    cudnnCreateFilterDescriptor(&x296);
    cudnnSetFilter4dDescriptor(x296, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 256, 1, 17, 17);
    cudnnConvolutionDescriptor_t x297;
    cudnnCreateConvolutionDescriptor(&x297);
    cudnnSetConvolution2dDescriptor(x297, 8, 8, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnTensorDescriptor_t x298;
    cudnnCreateTensorDescriptor(&x298);
    cudnnSetTensor4dDescriptor(x298, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2, 256, 128, 128);
    float* x299 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x299, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionFwdAlgoPerf_t x300;
    int x301 = 0;
    cudnnFindConvolutionForwardAlgorithm(x13, x295, x296, x297, x298, 1, &x301, &x300);
    cudnnConvolutionFwdAlgo_t x302 = x300.algo;
    size_t x303 = (size_t)0;
    cudnnGetConvolutionForwardWorkspaceSize(x13, x295, x296, x297, x298, x302, &x303);
    float* x304 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x304, (size_t)x303));
    float* x305 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x305, (size_t)(8388608 * sizeof(float))));
    float* x306 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x306, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionFwdAlgoPerf_t x307;
    int x308 = 0;
    cudnnFindConvolutionForwardAlgorithm(x13, x298, x296, x297, x298, 1, &x308, &x307);
    cudnnConvolutionFwdAlgo_t x309 = x307.algo;
    size_t x310 = (size_t)0;
    cudnnGetConvolutionForwardWorkspaceSize(x13, x298, x296, x297, x298, x309, &x310);
    float* x311 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x311, (size_t)x310));
    float* x312 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x312, (size_t)(8388608 * sizeof(float))));
    float* x313 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x313, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionFwdAlgoPerf_t x314;
    int x315 = 0;
    cudnnFindConvolutionForwardAlgorithm(x13, x298, x296, x297, x298, 1, &x315, &x314);
    cudnnConvolutionFwdAlgo_t x316 = x314.algo;
    size_t x317 = (size_t)0;
    cudnnGetConvolutionForwardWorkspaceSize(x13, x298, x296, x297, x298, x316, &x317);
    float* x318 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x318, (size_t)x317));
    float* x319 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x319, (size_t)(8388608 * sizeof(float))));
    float* x320 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x320, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionFwdAlgoPerf_t x321;
    int x322 = 0;
    cudnnFindConvolutionForwardAlgorithm(x13, x298, x296, x297, x298, 1, &x322, &x321);
    cudnnConvolutionFwdAlgo_t x323 = x321.algo;
    size_t x324 = (size_t)0;
    cudnnGetConvolutionForwardWorkspaceSize(x13, x298, x296, x297, x298, x323, &x324);
    float* x325 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x325, (size_t)x324));
    float* x326 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x326, (size_t)(8388608 * sizeof(float))));
    float* x327 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x327, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionFwdAlgoPerf_t x328;
    int x329 = 0;
    cudnnFindConvolutionForwardAlgorithm(x13, x298, x296, x297, x298, 1, &x329, &x328);
    cudnnConvolutionFwdAlgo_t x330 = x328.algo;
    size_t x331 = (size_t)0;
    cudnnGetConvolutionForwardWorkspaceSize(x13, x298, x296, x297, x298, x330, &x331);
    float* x332 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x332, (size_t)x331));
    float* x333 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x333, (size_t)(8388608 * sizeof(float))));
    cudnnFilterDescriptor_t x334;
    cudnnCreateFilterDescriptor(&x334);
    cudnnSetFilter4dDescriptor(x334, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 256, 17, 17);
    float* x335 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x335, (size_t)(32768 * sizeof(float))));
    cudnnConvolutionFwdAlgoPerf_t x336;
    int x337 = 0;
    cudnnFindConvolutionForwardAlgorithm(x13, x298, x334, x297, x295, 1, &x337, &x336);
    cudnnConvolutionFwdAlgo_t x338 = x336.algo;
    size_t x339 = (size_t)0;
    cudnnGetConvolutionForwardWorkspaceSize(x13, x298, x334, x297, x295, x338, &x339);
    float* x340 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x340, (size_t)x339));
    float* x341 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x341, (size_t)(32768 * sizeof(float))));
    float* x342 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x342, (size_t)(8388608 * sizeof(float))));
    float* x343 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x343, (size_t)(8388608 * sizeof(float))));
    float* x344 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x344, (size_t)(8388608 * sizeof(float))));
    float* x345 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x345, (size_t)(8388608 * sizeof(float))));
    float* x346 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x346, (size_t)(8388608 * sizeof(float))));
    float* x347 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x347, (size_t)(8388608 * sizeof(float))));
    float* x348 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x348, (size_t)(8388608 * sizeof(float))));
    float* x349 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x349, (size_t)(8388608 * sizeof(float))));
    float* x350 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x350, (size_t)(8388608 * sizeof(float))));
    float* x351 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x351, (size_t)(8388608 * sizeof(float))));
    float* x352 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x352, (size_t)(32768 * sizeof(float))));
    float* x353 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x353, (size_t)(32768 * sizeof(float))));
    float* x354 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x354, (size_t)(32768 * sizeof(float))));
    cudnnTensorDescriptor_t x355;
    cudnnCreateTensorDescriptor(&x355);
    cudnnSetTensor4dDescriptor(x355, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2, 256, 128, 128);
    cudnnTensorDescriptor_t x356;
    cudnnCreateTensorDescriptor(&x356);
    cudnnSetTensor4dDescriptor(x356, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2, 1, 128, 128);
    cudnnFilterDescriptor_t x357;
    cudnnCreateFilterDescriptor(&x357);
    cudnnSetFilter4dDescriptor(x357, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 256, 17, 17);
    cudnnConvolutionDescriptor_t x358;
    cudnnCreateConvolutionDescriptor(&x358);
    cudnnSetConvolution2dDescriptor(x358, 8, 8, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    float* x359 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x359, (size_t)(73984 * sizeof(float))));
    cudnnConvolutionBwdFilterAlgoPerf_t x360;
    int x361 = 0;
    cudnnFindConvolutionBackwardFilterAlgorithm(x13, x355, x356, x358, x357, 1, &x361, &x360);
    cudnnConvolutionBwdFilterAlgo_t x362 = x360.algo;
    size_t x363 = (size_t)0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(x13, x355, x356, x358, x357, x362, &x363);
    float* x364 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x364, (size_t)x363));
    float* x365 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x365, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionBwdDataAlgoPerf_t x366;
    int x367 = 0;
    cudnnFindConvolutionBackwardDataAlgorithm(x13, x357, x356, x358, x355, 1, &x367, &x366);
    cudnnConvolutionBwdDataAlgo_t x368 = x366.algo;
    size_t x369 = (size_t)0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(x13, x357, x356, x358, x355, x368, &x369);
    float* x370 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x370, (size_t)x369));
    float* x371 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x371, (size_t)(8388608 * sizeof(float))));
    cudnnFilterDescriptor_t x372;
    cudnnCreateFilterDescriptor(&x372);
    cudnnSetFilter4dDescriptor(x372, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 256, 1, 17, 17);
    float* x373 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x373, (size_t)(73984 * sizeof(float))));
    cudnnConvolutionBwdFilterAlgoPerf_t x374;
    int x375 = 0;
    cudnnFindConvolutionBackwardFilterAlgorithm(x13, x355, x355, x358, x372, 1, &x375, &x374);
    cudnnConvolutionBwdFilterAlgo_t x376 = x374.algo;
    size_t x377 = (size_t)0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(x13, x355, x355, x358, x372, x376, &x377);
    float* x378 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x378, (size_t)x377));
    float* x379 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x379, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionBwdDataAlgoPerf_t x380;
    int x381 = 0;
    cudnnFindConvolutionBackwardDataAlgorithm(x13, x372, x355, x358, x355, 1, &x381, &x380);
    cudnnConvolutionBwdDataAlgo_t x382 = x380.algo;
    size_t x383 = (size_t)0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(x13, x372, x355, x358, x355, x382, &x383);
    float* x384 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x384, (size_t)x383));
    float* x385 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x385, (size_t)(8388608 * sizeof(float))));
    float* x386 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x386, (size_t)(73984 * sizeof(float))));
    cudnnConvolutionBwdFilterAlgoPerf_t x387;
    int x388 = 0;
    cudnnFindConvolutionBackwardFilterAlgorithm(x13, x355, x355, x358, x372, 1, &x388, &x387);
    cudnnConvolutionBwdFilterAlgo_t x389 = x387.algo;
    size_t x390 = (size_t)0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(x13, x355, x355, x358, x372, x389, &x390);
    float* x391 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x391, (size_t)x390));
    float* x392 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x392, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionBwdDataAlgoPerf_t x393;
    int x394 = 0;
    cudnnFindConvolutionBackwardDataAlgorithm(x13, x372, x355, x358, x355, 1, &x394, &x393);
    cudnnConvolutionBwdDataAlgo_t x395 = x393.algo;
    size_t x396 = (size_t)0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(x13, x372, x355, x358, x355, x395, &x396);
    float* x397 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x397, (size_t)x396));
    float* x398 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x398, (size_t)(8388608 * sizeof(float))));
    float* x399 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x399, (size_t)(73984 * sizeof(float))));
    cudnnConvolutionBwdFilterAlgoPerf_t x400;
    int x401 = 0;
    cudnnFindConvolutionBackwardFilterAlgorithm(x13, x355, x355, x358, x372, 1, &x401, &x400);
    cudnnConvolutionBwdFilterAlgo_t x402 = x400.algo;
    size_t x403 = (size_t)0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(x13, x355, x355, x358, x372, x402, &x403);
    float* x404 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x404, (size_t)x403));
    float* x405 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x405, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionBwdDataAlgoPerf_t x406;
    int x407 = 0;
    cudnnFindConvolutionBackwardDataAlgorithm(x13, x372, x355, x358, x355, 1, &x407, &x406);
    cudnnConvolutionBwdDataAlgo_t x408 = x406.algo;
    size_t x409 = (size_t)0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(x13, x372, x355, x358, x355, x408, &x409);
    float* x410 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x410, (size_t)x409));
    float* x411 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x411, (size_t)(8388608 * sizeof(float))));
    float* x412 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x412, (size_t)(73984 * sizeof(float))));
    cudnnConvolutionBwdFilterAlgoPerf_t x413;
    int x414 = 0;
    cudnnFindConvolutionBackwardFilterAlgorithm(x13, x355, x355, x358, x372, 1, &x414, &x413);
    cudnnConvolutionBwdFilterAlgo_t x415 = x413.algo;
    size_t x416 = (size_t)0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(x13, x355, x355, x358, x372, x415, &x416);
    float* x417 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x417, (size_t)x416));
    float* x418 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x418, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionBwdDataAlgoPerf_t x419;
    int x420 = 0;
    cudnnFindConvolutionBackwardDataAlgorithm(x13, x372, x355, x358, x355, 1, &x420, &x419);
    cudnnConvolutionBwdDataAlgo_t x421 = x419.algo;
    size_t x422 = (size_t)0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(x13, x372, x355, x358, x355, x421, &x422);
    float* x423 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x423, (size_t)x422));
    float* x424 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x424, (size_t)(8388608 * sizeof(float))));
    float* x425 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x425, (size_t)(73984 * sizeof(float))));
    cudnnConvolutionBwdFilterAlgoPerf_t x426;
    int x427 = 0;
    cudnnFindConvolutionBackwardFilterAlgorithm(x13, x356, x355, x358, x372, 1, &x427, &x426);
    cudnnConvolutionBwdFilterAlgo_t x428 = x426.algo;
    size_t x429 = (size_t)0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(x13, x356, x355, x358, x372, x428, &x429);
    float* x430 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x430, (size_t)x429));
    float* x431 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x431, (size_t)(32768 * sizeof(float))));
    cudnnConvolutionBwdDataAlgoPerf_t x432;
    int x433 = 0;
    cudnnFindConvolutionBackwardDataAlgorithm(x13, x372, x355, x358, x356, 1, &x433, &x432);
    cudnnConvolutionBwdDataAlgo_t x434 = x432.algo;
    size_t x435 = (size_t)0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(x13, x372, x355, x358, x356, x434, &x435);
    float* x436 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x436, (size_t)x435));

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    CUDA_CALL(cudaEventCreate(&start_event));
    CUDA_CALL(cudaEventCreate(&finish_event));
    CUDA_CALL(cudaEventRecord(start_event));
    while (x292 != 5) {
      int x437 = 0;
      while (x437 != 8) {
        int x438 = x437;
        x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x293, 0, 32768);
        ncclRecv(x293, (size_t)32768, ncclFloat32, x294, x4, x5);
        cudaStreamSynchronize(x5);
        int x439 = 32768 * x438;
        cudaMemcpy(x291 + x439, x293, (size_t)(32768 * sizeof(float)), cudaMemcpyDeviceToDevice);
        float x440 = 1.0;
        float x441 = 0.0;
        cudnnConvolutionForward(x13, &x440, x295, x293, x296, x262, x297, x302, x304, x303, &x441, x298, x299);
        int x442 = 8388608 * x438;
        cudaMemcpy(x282 + x442, x299, (size_t)(8388608 * sizeof(float)), cudaMemcpyDeviceToDevice);
        x63<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x299, x305, 8388608);
        cudaMemcpy(x288 + x442, x305, (size_t)(8388608 * sizeof(float)), cudaMemcpyDeviceToDevice);
        float x443 = 1.0;
        float x444 = 0.0;
        cudnnConvolutionForward(x13, &x443, x298, x305, x296, x265, x297, x309, x311, x310, &x444, x298, x306);
        cudaMemcpy(x283 + x442, x306, (size_t)(8388608 * sizeof(float)), cudaMemcpyDeviceToDevice);
        x63<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x306, x312, 8388608);
        cudaMemcpy(x280 + x442, x312, (size_t)(8388608 * sizeof(float)), cudaMemcpyDeviceToDevice);
        float x445 = 1.0;
        float x446 = 0.0;
        cudnnConvolutionForward(x13, &x445, x298, x312, x296, x268, x297, x316, x318, x317, &x446, x298, x313);
        cudaMemcpy(x289 + x442, x313, (size_t)(8388608 * sizeof(float)), cudaMemcpyDeviceToDevice);
        x63<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x313, x319, 8388608);
        cudaMemcpy(x287 + x442, x319, (size_t)(8388608 * sizeof(float)), cudaMemcpyDeviceToDevice);
        float x447 = 1.0;
        float x448 = 0.0;
        cudnnConvolutionForward(x13, &x447, x298, x319, x296, x271, x297, x323, x325, x324, &x448, x298, x320);
        cudaMemcpy(x290 + x442, x320, (size_t)(8388608 * sizeof(float)), cudaMemcpyDeviceToDevice);
        x63<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x320, x326, 8388608);
        cudaMemcpy(x284 + x442, x326, (size_t)(8388608 * sizeof(float)), cudaMemcpyDeviceToDevice);
        float x449 = 1.0;
        float x450 = 0.0;
        cudnnConvolutionForward(x13, &x449, x298, x326, x296, x274, x297, x330, x332, x331, &x450, x298, x327);
        cudaMemcpy(x285 + x442, x327, (size_t)(8388608 * sizeof(float)), cudaMemcpyDeviceToDevice);
        x63<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x327, x333, 8388608);
        cudaMemcpy(x286 + x442, x333, (size_t)(8388608 * sizeof(float)), cudaMemcpyDeviceToDevice);
        // end allocating gpu array for convolution forward workspace
        // begin convolution forward pass
        float x451 = 1.0;
        float x452 = 0.0;
        CUDNNCHECK(cudnnConvolutionForward(x13, &x451, x298, x333, x334, x277, x297, x338, x340, x339, &x452, x295, x335));
        // end convolution forward pass
        CUDA_CALL(cudaMemcpy(x281 + x439, x335, (size_t)(32768 * sizeof(float)), cudaMemcpyDeviceToDevice));
        x437 = x437 + 1;
      }
      int x453 = 0;
      while (x453 != 8) {
        int x454 = x453;
        x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x341, 0, 32768);
        x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x342, 0, 8388608);
        x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x343, 0, 8388608);
        x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x344, 0, 8388608);
        x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x345, 0, 8388608);
        x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x346, 0, 8388608);
        x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x347, 0, 8388608);
        x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x348, 0, 8388608);
        x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x349, 0, 8388608);
        x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x350, 0, 8388608);
        x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x351, 0, 8388608);
        x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x352, 0, 32768);
        int x455 = 8388608 * x454;
        int x456 = 32768 * x454;
        x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x353, 1, 32768);
        x120<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x353, x281 + x456, x354, 32768);
        // end computing RELU_GRAD on GPU for size 32768 and type Float at device (pre-rename) x66 with left_operand x3024 and right_operand x3010
        // begin computing ACCUM on GPU for size 32768 and type Float at device (pre-rename) x66 with base_operand x2997 and addition_operand x3037
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x352, x354, 32768);
        // end allocating gpu array for convolution backward filter workspace
        // begin convolution backward filter pass
        float x457 = 1.0;
        float x458 = 0.0;
        CUDNNCHECK(cudnnConvolutionBackwardFilter(x13, &x457, x355, x286 + x455, x356, x352, x358, x362, x364, x363, &x458, x357, x359));
        // end convolution backward filter pass
        cudaStreamSynchronize(0);
        ncclAllReduce(x359, x359, (size_t)73984, ncclFloat32, ncclSum, x11, x5);
        CUDA_CALL(cudaStreamSynchronize(x5));
        // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x66 with base_operand x2207 and addition_operand x3093
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x278, x359, 73984);
        // end allocating gpu array for convolution backward data workspace
        // begin convolution backward data pass
        float x459 = 1.0;
        float x460 = 0.0;
        CUDNNCHECK(cudnnConvolutionBackwardData(x13, &x459, x357, x277, x356, x352, x358, x368, x370, x369, &x460, x355, x365));
        // end convolution backward data pass
        // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x2984 and addition_operand x3145
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x351, x365, 8388608);
        x120<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x351, x285 + x455, x371, 8388608);
        // end computing RELU_GRAD on GPU for size 8388608 and type Float at device (pre-rename) x66 with left_operand x2984 and right_operand x3014
        // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x2971 and addition_operand x3190
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x350, x371, 8388608);
        // end allocating gpu array for convolution backward filter workspace
        // begin convolution backward filter pass
        float x461 = 1.0;
        float x462 = 0.0;
        CUDNNCHECK(cudnnConvolutionBackwardFilter(x13, &x461, x355, x284 + x455, x355, x350, x358, x376, x378, x377, &x462, x372, x373));
        // end convolution backward filter pass
        cudaStreamSynchronize(0);
        ncclAllReduce(x373, x373, (size_t)73984, ncclFloat32, ncclSum, x11, x5);
        CUDA_CALL(cudaStreamSynchronize(x5));
        // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x66 with base_operand x2171 and addition_operand x3219
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x275, x373, 73984);
        // end allocating gpu array for convolution backward data workspace
        // begin convolution backward data pass
        float x463 = 1.0;
        float x464 = 0.0;
        CUDNNCHECK(cudnnConvolutionBackwardData(x13, &x463, x372, x274, x355, x350, x358, x382, x384, x383, &x464, x355, x379));
        // end convolution backward data pass
        // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x2958 and addition_operand x3271
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x349, x379, 8388608);
        x120<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x349, x290 + x455, x385, 8388608);
        // end computing RELU_GRAD on GPU for size 8388608 and type Float at device (pre-rename) x66 with left_operand x2958 and right_operand x3019
        // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x2945 and addition_operand x3316
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x348, x385, 8388608);
        // end allocating gpu array for convolution backward filter workspace
        // begin convolution backward filter pass
        float x465 = 1.0;
        float x466 = 0.0;
        CUDNNCHECK(cudnnConvolutionBackwardFilter(x13, &x465, x355, x287 + x455, x355, x348, x358, x389, x391, x390, &x466, x372, x386));
        // end convolution backward filter pass
        cudaStreamSynchronize(0);
        ncclAllReduce(x386, x386, (size_t)73984, ncclFloat32, ncclSum, x11, x5);
        CUDA_CALL(cudaStreamSynchronize(x5));
        // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x66 with base_operand x2135 and addition_operand x3336
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x272, x386, 73984);
        // end allocating gpu array for convolution backward data workspace
        // begin convolution backward data pass
        float x467 = 1.0;
        float x468 = 0.0;
        CUDNNCHECK(cudnnConvolutionBackwardData(x13, &x467, x372, x271, x355, x348, x358, x395, x397, x396, &x468, x355, x392));
        // end convolution backward data pass
        // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x2932 and addition_operand x3388
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x347, x392, 8388608);
        x120<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x347, x289 + x455, x398, 8388608);
        // end computing RELU_GRAD on GPU for size 8388608 and type Float at device (pre-rename) x66 with left_operand x2932 and right_operand x3018
        // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x2919 and addition_operand x3433
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x346, x398, 8388608);
        // end allocating gpu array for convolution backward filter workspace
        // begin convolution backward filter pass
        float x469 = 1.0;
        float x470 = 0.0;
        CUDNNCHECK(cudnnConvolutionBackwardFilter(x13, &x469, x355, x280 + x455, x355, x346, x358, x402, x404, x403, &x470, x372, x399));
        // end convolution backward filter pass
        cudaStreamSynchronize(0);
        ncclAllReduce(x399, x399, (size_t)73984, ncclFloat32, ncclSum, x11, x5);
        CUDA_CALL(cudaStreamSynchronize(x5));
        // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x66 with base_operand x2099 and addition_operand x3453
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x269, x399, 73984);
        // end allocating gpu array for convolution backward data workspace
        // begin convolution backward data pass
        float x471 = 1.0;
        float x472 = 0.0;
        CUDNNCHECK(cudnnConvolutionBackwardData(x13, &x471, x372, x268, x355, x346, x358, x408, x410, x409, &x472, x355, x405));
        // end convolution backward data pass
        // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x2906 and addition_operand x3505
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x345, x405, 8388608);
        x120<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x345, x283 + x455, x411, 8388608);
        // end computing RELU_GRAD on GPU for size 8388608 and type Float at device (pre-rename) x66 with left_operand x2906 and right_operand x3012
        // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x2893 and addition_operand x3550
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x344, x411, 8388608);
        // end allocating gpu array for convolution backward filter workspace
        // begin convolution backward filter pass
        float x473 = 1.0;
        float x474 = 0.0;
        CUDNNCHECK(cudnnConvolutionBackwardFilter(x13, &x473, x355, x288 + x455, x355, x344, x358, x415, x417, x416, &x474, x372, x412));
        // end convolution backward filter pass
        cudaStreamSynchronize(0);
        ncclAllReduce(x412, x412, (size_t)73984, ncclFloat32, ncclSum, x11, x5);
        CUDA_CALL(cudaStreamSynchronize(x5));
        // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x66 with base_operand x2063 and addition_operand x3570
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x266, x412, 73984);
        // end allocating gpu array for convolution backward data workspace
        // begin convolution backward data pass
        float x475 = 1.0;
        float x476 = 0.0;
        CUDNNCHECK(cudnnConvolutionBackwardData(x13, &x475, x372, x265, x355, x344, x358, x421, x423, x422, &x476, x355, x418));
        // end convolution backward data pass
        // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x2880 and addition_operand x3622
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x343, x418, 8388608);
        x120<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x343, x282 + x455, x424, 8388608);
        // end computing RELU_GRAD on GPU for size 8388608 and type Float at device (pre-rename) x66 with left_operand x2880 and right_operand x3011
        // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x2867 and addition_operand x3667
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x342, x424, 8388608);
        // end allocating gpu array for convolution backward filter workspace
        // begin convolution backward filter pass
        float x477 = 1.0;
        float x478 = 0.0;
        CUDNNCHECK(cudnnConvolutionBackwardFilter(x13, &x477, x356, x291 + x456, x355, x342, x358, x428, x430, x429, &x478, x372, x425));
        // end convolution backward filter pass
        cudaStreamSynchronize(0);
        ncclAllReduce(x425, x425, (size_t)73984, ncclFloat32, ncclSum, x11, x5);
        CUDA_CALL(cudaStreamSynchronize(x5));
        // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x66 with base_operand x2027 and addition_operand x3687
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x263, x425, 73984);
        // end allocating gpu array for convolution backward data workspace
        // begin convolution backward data pass
        float x479 = 1.0;
        float x480 = 0.0;
        CUDNNCHECK(cudnnConvolutionBackwardData(x13, &x479, x372, x262, x355, x342, x358, x434, x436, x435, &x480, x356, x431));
        // end convolution backward data pass
        // begin computing ACCUM on GPU for size 32768 and type Float at device (pre-rename) x66 with base_operand x2854 and addition_operand x3739
        CUDA_CALL(cudaSetDevice(x12));
        x128<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x341, x431, 32768);
        // end computing ACCUM on GPU for size 32768 and type Float at device (pre-rename) x66 with base_operand x2854 and addition_operand x3739
        cudaStreamSynchronize(0);
        NCCLCHECK(ncclSend(x341, (size_t)32768, ncclFloat32, x294, x4, x5));
        x453 = x453 + 1;
      }
      // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x2017, grad x2027, and momentum x2040
      CUDA_CALL(cudaSetDevice(x12));
      x211<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x262, x263, x264, 73984);
      // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x2017, grad x2027, and momentum x2040
      // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x2053, grad x2063, and momentum x2076
      CUDA_CALL(cudaSetDevice(x12));
      x211<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x265, x266, x267, 73984);
      // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x2053, grad x2063, and momentum x2076
      // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x2089, grad x2099, and momentum x2112
      CUDA_CALL(cudaSetDevice(x12));
      x211<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x268, x269, x270, 73984);
      // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x2089, grad x2099, and momentum x2112
      // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x2125, grad x2135, and momentum x2148
      CUDA_CALL(cudaSetDevice(x12));
      x211<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x271, x272, x273, 73984);
      // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x2125, grad x2135, and momentum x2148
      // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x2161, grad x2171, and momentum x2184
      CUDA_CALL(cudaSetDevice(x12));
      x211<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x274, x275, x276, 73984);
      // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x2161, grad x2171, and momentum x2184
      // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x2197, grad x2207, and momentum x2220
      CUDA_CALL(cudaSetDevice(x12));
      x211<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x277, x278, x279, 73984);
      // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x2197, grad x2207, and momentum x2220
      x292 = x292 + 1;
    }
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
  NCCLCHECK(ncclCommDestroy(x11));
  NCCLCHECK(ncclCommDestroy(x4));
  CUDNNCHECK(cudnnDestroy(x13));
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
