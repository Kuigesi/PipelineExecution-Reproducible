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
__global__ void x51(float* x52, float* x53, int x54) {
  // begin generating kernel function for RELU of type Float
  int x55 = gridDim.x * blockDim.x;
  int x56 = threadIdx.x + blockIdx.x * blockDim.x;
  while (x56 < x54) {
    int x57 = x56;
    x53[x57] = max(0.0, x52[x57]);
    x56 = x56 + x55;
  }
  // end generating kernel function for RELU of type Float
}
__global__ void x108(float* x109, float* x110, float* x111, int x112) {
  // begin generating kernel function for RELU_GRAD of type Float
  int x113 = gridDim.x * blockDim.x;
  int x114 = threadIdx.x + blockIdx.x * blockDim.x;
  while (x114 < x112) {
    int x115 = x114;
    x111[x115] = x110[x115] > 0.0 ? x109[x115] : 0.0;
    x114 = x114 + x113;
  }
  // end generating kernel function for RELU_GRAD of type Float
}
__global__ void x116(float* x117, float* x118, int x119) {
  // begin generating kernel function for ACCUM of type Float
  int x120 = gridDim.x * blockDim.x;
  int x121 = threadIdx.x + blockIdx.x * blockDim.x;
  while (x121 < x119) {
    int x122 = x121;
    x117[x122] = x117[x122] + x118[x122];
    x121 = x121 + x120;
  }
  // end generating kernel function for ACCUM of type Float
}
__global__ void x194(float* x195, float* x196, float* x197, int x198) {
  // begin generating kernel function for SGD of type Float
  int x199 = gridDim.x * blockDim.x;
  int x200 = threadIdx.x + blockIdx.x * blockDim.x;
  while (x200 < x198) {
    int x201 = x200;
    float x202 = x197[x201] * 0.5 + x196[x201];
    x195[x201] = x195[x201] - x202 * 1.0E-4;
    x197[x201] = x202;
    x200 = x200 + x199;
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
    int x38 = 0;
    float* x39 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x39, (size_t)(32768 * sizeof(float))));
    cudnnTensorDescriptor_t x40;
    cudnnCreateTensorDescriptor(&x40);
    cudnnSetTensor4dDescriptor(x40, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2, 1, 128, 128);
    cudnnFilterDescriptor_t x41;
    cudnnCreateFilterDescriptor(&x41);
    cudnnSetFilter4dDescriptor(x41, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 256, 1, 17, 17);
    cudnnConvolutionDescriptor_t x42;
    cudnnCreateConvolutionDescriptor(&x42);
    cudnnSetConvolution2dDescriptor(x42, 8, 8, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnTensorDescriptor_t x43;
    cudnnCreateTensorDescriptor(&x43);
    cudnnSetTensor4dDescriptor(x43, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2, 256, 128, 128);
    float* x44 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x44, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionFwdAlgoPerf_t x45;
    int x46 = 0;
    cudnnFindConvolutionForwardAlgorithm(x13, x40, x41, x42, x43, 1, &x46, &x45);
    cudnnConvolutionFwdAlgo_t x47 = x45.algo;
    size_t x48 = (size_t)0;
    cudnnGetConvolutionForwardWorkspaceSize(x13, x40, x41, x42, x43, x47, &x48);
    float* x49 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x49, (size_t)x48));
    float* x50 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x50, (size_t)(8388608 * sizeof(float))));
    float* x58 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x58, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionFwdAlgoPerf_t x59;
    int x60 = 0;
    cudnnFindConvolutionForwardAlgorithm(x13, x43, x41, x42, x43, 1, &x60, &x59);
    cudnnConvolutionFwdAlgo_t x61 = x59.algo;
    size_t x62 = (size_t)0;
    cudnnGetConvolutionForwardWorkspaceSize(x13, x43, x41, x42, x43, x61, &x62);
    float* x63 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x63, (size_t)x62));
    float* x64 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x64, (size_t)(8388608 * sizeof(float))));
    float* x65 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x65, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionFwdAlgoPerf_t x66;
    int x67 = 0;
    cudnnFindConvolutionForwardAlgorithm(x13, x43, x41, x42, x43, 1, &x67, &x66);
    cudnnConvolutionFwdAlgo_t x68 = x66.algo;
    size_t x69 = (size_t)0;
    cudnnGetConvolutionForwardWorkspaceSize(x13, x43, x41, x42, x43, x68, &x69);
    float* x70 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x70, (size_t)x69));
    float* x71 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x71, (size_t)(8388608 * sizeof(float))));
    float* x72 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x72, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionFwdAlgoPerf_t x73;
    int x74 = 0;
    cudnnFindConvolutionForwardAlgorithm(x13, x43, x41, x42, x43, 1, &x74, &x73);
    cudnnConvolutionFwdAlgo_t x75 = x73.algo;
    size_t x76 = (size_t)0;
    cudnnGetConvolutionForwardWorkspaceSize(x13, x43, x41, x42, x43, x75, &x76);
    float* x77 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x77, (size_t)x76));
    float* x78 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x78, (size_t)(8388608 * sizeof(float))));
    float* x79 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x79, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionFwdAlgoPerf_t x80;
    int x81 = 0;
    cudnnFindConvolutionForwardAlgorithm(x13, x43, x41, x42, x43, 1, &x81, &x80);
    cudnnConvolutionFwdAlgo_t x82 = x80.algo;
    size_t x83 = (size_t)0;
    cudnnGetConvolutionForwardWorkspaceSize(x13, x43, x41, x42, x43, x82, &x83);
    float* x84 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x84, (size_t)x83));
    float* x85 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x85, (size_t)(8388608 * sizeof(float))));
    cudnnFilterDescriptor_t x86;
    cudnnCreateFilterDescriptor(&x86);
    cudnnSetFilter4dDescriptor(x86, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 256, 17, 17);
    float* x87 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x87, (size_t)(32768 * sizeof(float))));
    cudnnConvolutionFwdAlgoPerf_t x88;
    int x89 = 0;
    cudnnFindConvolutionForwardAlgorithm(x13, x43, x86, x42, x40, 1, &x89, &x88);
    cudnnConvolutionFwdAlgo_t x90 = x88.algo;
    size_t x91 = (size_t)0;
    cudnnGetConvolutionForwardWorkspaceSize(x13, x43, x86, x42, x40, x90, &x91);
    float* x92 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x92, (size_t)x91));
    float* x93 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x93, (size_t)(32768 * sizeof(float))));
    int x94 = x12 + 2;
    float* x95 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x95, (size_t)(8388608 * sizeof(float))));
    float* x96 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x96, (size_t)(8388608 * sizeof(float))));
    float* x97 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x97, (size_t)(8388608 * sizeof(float))));
    float* x98 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x98, (size_t)(8388608 * sizeof(float))));
    float* x99 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x99, (size_t)(8388608 * sizeof(float))));
    float* x100 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x100, (size_t)(8388608 * sizeof(float))));
    float* x101 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x101, (size_t)(8388608 * sizeof(float))));
    float* x102 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x102, (size_t)(8388608 * sizeof(float))));
    float* x103 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x103, (size_t)(8388608 * sizeof(float))));
    float* x104 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x104, (size_t)(8388608 * sizeof(float))));
    float* x105 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x105, (size_t)(32768 * sizeof(float))));
    float* x106 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x106, (size_t)(32768 * sizeof(float))));
    float* x107 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x107, (size_t)(32768 * sizeof(float))));
    float* x123 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x123, (size_t)(73984 * sizeof(float))));
    cudnnConvolutionBwdFilterAlgoPerf_t x124;
    int x125 = 0;
    cudnnFindConvolutionBackwardFilterAlgorithm(x13, x43, x40, x42, x86, 1, &x125, &x124);
    cudnnConvolutionBwdFilterAlgo_t x126 = x124.algo;
    size_t x127 = (size_t)0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(x13, x43, x40, x42, x86, x126, &x127);
    float* x128 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x128, (size_t)x127));
    float* x129 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x129, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionBwdDataAlgoPerf_t x130;
    int x131 = 0;
    cudnnFindConvolutionBackwardDataAlgorithm(x13, x86, x40, x42, x43, 1, &x131, &x130);
    cudnnConvolutionBwdDataAlgo_t x132 = x130.algo;
    size_t x133 = (size_t)0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(x13, x86, x40, x42, x43, x132, &x133);
    float* x134 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x134, (size_t)x133));
    float* x135 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x135, (size_t)(8388608 * sizeof(float))));
    float* x136 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x136, (size_t)(73984 * sizeof(float))));
    cudnnConvolutionBwdFilterAlgoPerf_t x137;
    int x138 = 0;
    cudnnFindConvolutionBackwardFilterAlgorithm(x13, x43, x43, x42, x41, 1, &x138, &x137);
    cudnnConvolutionBwdFilterAlgo_t x139 = x137.algo;
    size_t x140 = (size_t)0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(x13, x43, x43, x42, x41, x139, &x140);
    float* x141 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x141, (size_t)x140));
    float* x142 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x142, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionBwdDataAlgoPerf_t x143;
    int x144 = 0;
    cudnnFindConvolutionBackwardDataAlgorithm(x13, x41, x43, x42, x43, 1, &x144, &x143);
    cudnnConvolutionBwdDataAlgo_t x145 = x143.algo;
    size_t x146 = (size_t)0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(x13, x41, x43, x42, x43, x145, &x146);
    float* x147 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x147, (size_t)x146));
    float* x148 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x148, (size_t)(8388608 * sizeof(float))));
    float* x149 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x149, (size_t)(73984 * sizeof(float))));
    cudnnConvolutionBwdFilterAlgoPerf_t x150;
    int x151 = 0;
    cudnnFindConvolutionBackwardFilterAlgorithm(x13, x43, x43, x42, x41, 1, &x151, &x150);
    cudnnConvolutionBwdFilterAlgo_t x152 = x150.algo;
    size_t x153 = (size_t)0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(x13, x43, x43, x42, x41, x152, &x153);
    float* x154 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x154, (size_t)x153));
    float* x155 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x155, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionBwdDataAlgoPerf_t x156;
    int x157 = 0;
    cudnnFindConvolutionBackwardDataAlgorithm(x13, x41, x43, x42, x43, 1, &x157, &x156);
    cudnnConvolutionBwdDataAlgo_t x158 = x156.algo;
    size_t x159 = (size_t)0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(x13, x41, x43, x42, x43, x158, &x159);
    float* x160 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x160, (size_t)x159));
    float* x161 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x161, (size_t)(8388608 * sizeof(float))));
    float* x162 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x162, (size_t)(73984 * sizeof(float))));
    cudnnConvolutionBwdFilterAlgoPerf_t x163;
    int x164 = 0;
    cudnnFindConvolutionBackwardFilterAlgorithm(x13, x43, x43, x42, x41, 1, &x164, &x163);
    cudnnConvolutionBwdFilterAlgo_t x165 = x163.algo;
    size_t x166 = (size_t)0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(x13, x43, x43, x42, x41, x165, &x166);
    float* x167 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x167, (size_t)x166));
    float* x168 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x168, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionBwdDataAlgoPerf_t x169;
    int x170 = 0;
    cudnnFindConvolutionBackwardDataAlgorithm(x13, x41, x43, x42, x43, 1, &x170, &x169);
    cudnnConvolutionBwdDataAlgo_t x171 = x169.algo;
    size_t x172 = (size_t)0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(x13, x41, x43, x42, x43, x171, &x172);
    float* x173 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x173, (size_t)x172));
    float* x174 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x174, (size_t)(8388608 * sizeof(float))));
    float* x175 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x175, (size_t)(73984 * sizeof(float))));
    cudnnConvolutionBwdFilterAlgoPerf_t x176;
    int x177 = 0;
    cudnnFindConvolutionBackwardFilterAlgorithm(x13, x43, x43, x42, x41, 1, &x177, &x176);
    cudnnConvolutionBwdFilterAlgo_t x178 = x176.algo;
    size_t x179 = (size_t)0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(x13, x43, x43, x42, x41, x178, &x179);
    float* x180 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x180, (size_t)x179));
    float* x181 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x181, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionBwdDataAlgoPerf_t x182;
    int x183 = 0;
    cudnnFindConvolutionBackwardDataAlgorithm(x13, x41, x43, x42, x43, 1, &x183, &x182);
    cudnnConvolutionBwdDataAlgo_t x184 = x182.algo;
    size_t x185 = (size_t)0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(x13, x41, x43, x42, x43, x184, &x185);
    float* x186 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x186, (size_t)x185));
    float* x187 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x187, (size_t)(8388608 * sizeof(float))));
    float* x188 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x188, (size_t)(73984 * sizeof(float))));
    cudnnConvolutionBwdFilterAlgoPerf_t x189;
    int x190 = 0;
    cudnnFindConvolutionBackwardFilterAlgorithm(x13, x40, x43, x42, x41, 1, &x190, &x189);
    cudnnConvolutionBwdFilterAlgo_t x191 = x189.algo;
    size_t x192 = (size_t)0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(x13, x40, x43, x42, x41, x191, &x192);
    float* x193 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x193, (size_t)x192));

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    CUDA_CALL(cudaEventCreate(&start_event));
    CUDA_CALL(cudaEventCreate(&finish_event));
    CUDA_CALL(cudaEventRecord(start_event));
    while (x38 != 40) {
      float x203 = 1.0;
      float x204 = 0.0;
      cudnnConvolutionForward(x13, &x203, x40, x39, x41, x14, x42, x47, x49, x48, &x204, x43, x44);
      x51<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x44, x50, 8388608);
      float x205 = 1.0;
      float x206 = 0.0;
      cudnnConvolutionForward(x13, &x205, x43, x50, x41, x23, x42, x61, x63, x62, &x206, x43, x58);
      x51<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x58, x64, 8388608);
      float x207 = 1.0;
      float x208 = 0.0;
      cudnnConvolutionForward(x13, &x207, x43, x64, x41, x26, x42, x68, x70, x69, &x208, x43, x65);
      x51<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x65, x71, 8388608);
      float x209 = 1.0;
      float x210 = 0.0;
      cudnnConvolutionForward(x13, &x209, x43, x71, x41, x29, x42, x75, x77, x76, &x210, x43, x72);
      x51<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x72, x78, 8388608);
      float x211 = 1.0;
      float x212 = 0.0;
      cudnnConvolutionForward(x13, &x211, x43, x78, x41, x32, x42, x82, x84, x83, &x212, x43, x79);
      x51<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x79, x85, 8388608);
      float x213 = 1.0;
      float x214 = 0.0;
      cudnnConvolutionForward(x13, &x213, x43, x85, x86, x35, x42, x90, x92, x91, &x214, x40, x87);
      x51<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x87, x93, 32768);
      cudaStreamSynchronize(0);
      ncclSend(x93, (size_t)32768, ncclFloat32, x94, x4, x5);
      x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x95, 0, 8388608);
      x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x96, 0, 8388608);
      x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x97, 0, 8388608);
      x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x98, 0, 8388608);
      x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x99, 0, 8388608);
      x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x100, 0, 8388608);
      x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x101, 0, 8388608);
      x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x102, 0, 8388608);
      x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x103, 0, 8388608);
      x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x104, 0, 8388608);
      x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x105, 0, 32768);
      x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x106, 0, 32768);
      ncclRecv(x106, (size_t)32768, ncclFloat32, x94, x4, x5);
      cudaStreamSynchronize(x5);
      x108<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x106, x87, x107, 32768);
      // end computing RELU_GRAD on GPU for size 32768 and type Float at device (pre-rename) x66 with left_operand x868 and right_operand x668
      // begin computing ACCUM on GPU for size 32768 and type Float at device (pre-rename) x66 with base_operand x855 and addition_operand x887
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x105, x107, 32768);
      // end allocating gpu array for convolution backward filter workspace
      // begin convolution backward filter pass
      float x215 = 1.0;
      float x216 = 0.0;
      CUDNNCHECK(cudnnConvolutionBackwardFilter(x13, &x215, x43, x85, x40, x105, x42, x126, x128, x127, &x216, x86, x123));
      // end convolution backward filter pass
      cudaStreamSynchronize(0);
      ncclAllReduce(x123, x123, (size_t)73984, ncclFloat32, ncclSum, x11, x5);
      CUDA_CALL(cudaStreamSynchronize(x5));
      // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x66 with base_operand x298 and addition_operand x971
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x36, x123, 73984);
      // end allocating gpu array for convolution backward data workspace
      // begin convolution backward data pass
      float x217 = 1.0;
      float x218 = 0.0;
      CUDNNCHECK(cudnnConvolutionBackwardData(x13, &x217, x86, x35, x40, x105, x42, x132, x134, x133, &x218, x43, x129));
      // end convolution backward data pass
      // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x842 and addition_operand x1023
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x104, x129, 8388608);
      x108<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x104, x79, x135, 8388608);
      // end computing RELU_GRAD on GPU for size 8388608 and type Float at device (pre-rename) x66 with left_operand x842 and right_operand x608
      // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x829 and addition_operand x1068
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x103, x135, 8388608);
      // end allocating gpu array for convolution backward filter workspace
      // begin convolution backward filter pass
      float x219 = 1.0;
      float x220 = 0.0;
      CUDNNCHECK(cudnnConvolutionBackwardFilter(x13, &x219, x43, x78, x43, x103, x42, x139, x141, x140, &x220, x41, x136));
      // end convolution backward filter pass
      cudaStreamSynchronize(0);
      ncclAllReduce(x136, x136, (size_t)73984, ncclFloat32, ncclSum, x11, x5);
      CUDA_CALL(cudaStreamSynchronize(x5));
      // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x66 with base_operand x262 and addition_operand x1088
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x33, x136, 73984);
      // end allocating gpu array for convolution backward data workspace
      // begin convolution backward data pass
      float x221 = 1.0;
      float x222 = 0.0;
      CUDNNCHECK(cudnnConvolutionBackwardData(x13, &x221, x41, x32, x43, x103, x42, x145, x147, x146, &x222, x43, x142));
      // end convolution backward data pass
      // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x816 and addition_operand x1140
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x102, x142, 8388608);
      x108<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x102, x72, x148, 8388608);
      // end computing RELU_GRAD on GPU for size 8388608 and type Float at device (pre-rename) x66 with left_operand x816 and right_operand x557
      // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x803 and addition_operand x1185
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x101, x148, 8388608);
      // end allocating gpu array for convolution backward filter workspace
      // begin convolution backward filter pass
      float x223 = 1.0;
      float x224 = 0.0;
      CUDNNCHECK(cudnnConvolutionBackwardFilter(x13, &x223, x43, x71, x43, x101, x42, x152, x154, x153, &x224, x41, x149));
      // end convolution backward filter pass
      cudaStreamSynchronize(0);
      ncclAllReduce(x149, x149, (size_t)73984, ncclFloat32, ncclSum, x11, x5);
      CUDA_CALL(cudaStreamSynchronize(x5));
      // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x66 with base_operand x226 and addition_operand x1205
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x30, x149, 73984);
      // end allocating gpu array for convolution backward data workspace
      // begin convolution backward data pass
      float x225 = 1.0;
      float x226 = 0.0;
      CUDNNCHECK(cudnnConvolutionBackwardData(x13, &x225, x41, x29, x43, x101, x42, x158, x160, x159, &x226, x43, x155));
      // end convolution backward data pass
      // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x790 and addition_operand x1257
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x100, x155, 8388608);
      x108<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x100, x65, x161, 8388608);
      // end computing RELU_GRAD on GPU for size 8388608 and type Float at device (pre-rename) x66 with left_operand x790 and right_operand x506
      // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x777 and addition_operand x1302
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x99, x161, 8388608);
      // end allocating gpu array for convolution backward filter workspace
      // begin convolution backward filter pass
      float x227 = 1.0;
      float x228 = 0.0;
      CUDNNCHECK(cudnnConvolutionBackwardFilter(x13, &x227, x43, x64, x43, x99, x42, x165, x167, x166, &x228, x41, x162));
      // end convolution backward filter pass
      cudaStreamSynchronize(0);
      ncclAllReduce(x162, x162, (size_t)73984, ncclFloat32, ncclSum, x11, x5);
      CUDA_CALL(cudaStreamSynchronize(x5));
      // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x66 with base_operand x190 and addition_operand x1322
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x27, x162, 73984);
      // end allocating gpu array for convolution backward data workspace
      // begin convolution backward data pass
      float x229 = 1.0;
      float x230 = 0.0;
      CUDNNCHECK(cudnnConvolutionBackwardData(x13, &x229, x41, x26, x43, x99, x42, x171, x173, x172, &x230, x43, x168));
      // end convolution backward data pass
      // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x764 and addition_operand x1374
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x98, x168, 8388608);
      x108<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x98, x58, x174, 8388608);
      // end computing RELU_GRAD on GPU for size 8388608 and type Float at device (pre-rename) x66 with left_operand x764 and right_operand x455
      // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x751 and addition_operand x1419
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x97, x174, 8388608);
      // end allocating gpu array for convolution backward filter workspace
      // begin convolution backward filter pass
      float x231 = 1.0;
      float x232 = 0.0;
      CUDNNCHECK(cudnnConvolutionBackwardFilter(x13, &x231, x43, x50, x43, x97, x42, x178, x180, x179, &x232, x41, x175));
      // end convolution backward filter pass
      cudaStreamSynchronize(0);
      ncclAllReduce(x175, x175, (size_t)73984, ncclFloat32, ncclSum, x11, x5);
      CUDA_CALL(cudaStreamSynchronize(x5));
      // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x66 with base_operand x154 and addition_operand x1439
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x24, x175, 73984);
      // end allocating gpu array for convolution backward data workspace
      // begin convolution backward data pass
      float x233 = 1.0;
      float x234 = 0.0;
      CUDNNCHECK(cudnnConvolutionBackwardData(x13, &x233, x41, x23, x43, x97, x42, x184, x186, x185, &x234, x43, x181));
      // end convolution backward data pass
      // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x738 and addition_operand x1491
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x96, x181, 8388608);
      x108<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x96, x44, x187, 8388608);
      // end computing RELU_GRAD on GPU for size 8388608 and type Float at device (pre-rename) x66 with left_operand x738 and right_operand x375
      // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x725 and addition_operand x1536
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x95, x187, 8388608);
      // end allocating gpu array for convolution backward filter workspace
      // begin convolution backward filter pass
      float x235 = 1.0;
      float x236 = 0.0;
      CUDNNCHECK(cudnnConvolutionBackwardFilter(x13, &x235, x40, x39, x43, x95, x42, x191, x193, x192, &x236, x41, x188));
      // end convolution backward filter pass
      cudaStreamSynchronize(0);
      ncclAllReduce(x188, x188, (size_t)73984, ncclFloat32, ncclSum, x11, x5);
      CUDA_CALL(cudaStreamSynchronize(x5));
      // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x66 with base_operand x91 and addition_operand x1556
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x15, x188, 73984);
      // end computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x66 with base_operand x91 and addition_operand x1556
      // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x81, grad x91, and momentum x131
      CUDA_CALL(cudaSetDevice(x12));
      x194<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x14, x15, x22, 73984);
      // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x81, grad x91, and momentum x131
      // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x144, grad x154, and momentum x167
      CUDA_CALL(cudaSetDevice(x12));
      x194<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x23, x24, x25, 73984);
      // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x144, grad x154, and momentum x167
      // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x180, grad x190, and momentum x203
      CUDA_CALL(cudaSetDevice(x12));
      x194<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x26, x27, x28, 73984);
      // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x180, grad x190, and momentum x203
      // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x216, grad x226, and momentum x239
      CUDA_CALL(cudaSetDevice(x12));
      x194<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x29, x30, x31, 73984);
      // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x216, grad x226, and momentum x239
      // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x252, grad x262, and momentum x275
      CUDA_CALL(cudaSetDevice(x12));
      x194<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x32, x33, x34, 73984);
      // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x252, grad x262, and momentum x275
      // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x288, grad x298, and momentum x311
      CUDA_CALL(cudaSetDevice(x12));
      x194<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x35, x36, x37, 73984);
      // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x288, grad x298, and momentum x311
      x38 = x38 + 1;
    }
  }
  if (x12 >= 2 && x12 < 4) {
    // begin initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x237 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x237, (size_t)(73984 * sizeof(float))));
    // end initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x238 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x238, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x238, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x239 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x239, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x239, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x240 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x240, (size_t)(73984 * sizeof(float))));
    // end initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x241 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x241, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x241, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x242 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x242, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x242, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x243 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x243, (size_t)(73984 * sizeof(float))));
    // end initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x244 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x244, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x244, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x245 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x245, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x245, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x246 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x246, (size_t)(73984 * sizeof(float))));
    // end initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x247 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x247, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x247, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x248 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x248, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x248, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x249 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x249, (size_t)(73984 * sizeof(float))));
    // end initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x250 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x250, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x250, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x251 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x251, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x251, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x252 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x252, (size_t)(73984 * sizeof(float))));
    // end initializing random GPU array of size 73984 and type Float at device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x253 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x253, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x253, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    // begin initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    CUDA_CALL(cudaSetDevice(x12));
    float* x254 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x254, (size_t)(73984 * sizeof(float))));
    x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x254, 0, 73984);
    // end initializing fixed GPU array of size 73984 and type Float and device (pre-rename) x66
    int x255 = 0;
    float* x256 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x256, (size_t)(32768 * sizeof(float))));
    int x257 = x12 - 2;
    cudnnTensorDescriptor_t x258;
    cudnnCreateTensorDescriptor(&x258);
    cudnnSetTensor4dDescriptor(x258, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2, 1, 128, 128);
    cudnnFilterDescriptor_t x259;
    cudnnCreateFilterDescriptor(&x259);
    cudnnSetFilter4dDescriptor(x259, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 256, 1, 17, 17);
    cudnnConvolutionDescriptor_t x260;
    cudnnCreateConvolutionDescriptor(&x260);
    cudnnSetConvolution2dDescriptor(x260, 8, 8, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnTensorDescriptor_t x261;
    cudnnCreateTensorDescriptor(&x261);
    cudnnSetTensor4dDescriptor(x261, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2, 256, 128, 128);
    float* x262 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x262, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionFwdAlgoPerf_t x263;
    int x264 = 0;
    cudnnFindConvolutionForwardAlgorithm(x13, x258, x259, x260, x261, 1, &x264, &x263);
    cudnnConvolutionFwdAlgo_t x265 = x263.algo;
    size_t x266 = (size_t)0;
    cudnnGetConvolutionForwardWorkspaceSize(x13, x258, x259, x260, x261, x265, &x266);
    float* x267 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x267, (size_t)x266));
    float* x268 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x268, (size_t)(8388608 * sizeof(float))));
    float* x269 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x269, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionFwdAlgoPerf_t x270;
    int x271 = 0;
    cudnnFindConvolutionForwardAlgorithm(x13, x261, x259, x260, x261, 1, &x271, &x270);
    cudnnConvolutionFwdAlgo_t x272 = x270.algo;
    size_t x273 = (size_t)0;
    cudnnGetConvolutionForwardWorkspaceSize(x13, x261, x259, x260, x261, x272, &x273);
    float* x274 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x274, (size_t)x273));
    float* x275 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x275, (size_t)(8388608 * sizeof(float))));
    float* x276 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x276, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionFwdAlgoPerf_t x277;
    int x278 = 0;
    cudnnFindConvolutionForwardAlgorithm(x13, x261, x259, x260, x261, 1, &x278, &x277);
    cudnnConvolutionFwdAlgo_t x279 = x277.algo;
    size_t x280 = (size_t)0;
    cudnnGetConvolutionForwardWorkspaceSize(x13, x261, x259, x260, x261, x279, &x280);
    float* x281 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x281, (size_t)x280));
    float* x282 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x282, (size_t)(8388608 * sizeof(float))));
    float* x283 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x283, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionFwdAlgoPerf_t x284;
    int x285 = 0;
    cudnnFindConvolutionForwardAlgorithm(x13, x261, x259, x260, x261, 1, &x285, &x284);
    cudnnConvolutionFwdAlgo_t x286 = x284.algo;
    size_t x287 = (size_t)0;
    cudnnGetConvolutionForwardWorkspaceSize(x13, x261, x259, x260, x261, x286, &x287);
    float* x288 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x288, (size_t)x287));
    float* x289 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x289, (size_t)(8388608 * sizeof(float))));
    float* x290 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x290, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionFwdAlgoPerf_t x291;
    int x292 = 0;
    cudnnFindConvolutionForwardAlgorithm(x13, x261, x259, x260, x261, 1, &x292, &x291);
    cudnnConvolutionFwdAlgo_t x293 = x291.algo;
    size_t x294 = (size_t)0;
    cudnnGetConvolutionForwardWorkspaceSize(x13, x261, x259, x260, x261, x293, &x294);
    float* x295 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x295, (size_t)x294));
    float* x296 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x296, (size_t)(8388608 * sizeof(float))));
    cudnnFilterDescriptor_t x297;
    cudnnCreateFilterDescriptor(&x297);
    cudnnSetFilter4dDescriptor(x297, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 256, 17, 17);
    float* x298 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x298, (size_t)(32768 * sizeof(float))));
    cudnnConvolutionFwdAlgoPerf_t x299;
    int x300 = 0;
    cudnnFindConvolutionForwardAlgorithm(x13, x261, x297, x260, x258, 1, &x300, &x299);
    cudnnConvolutionFwdAlgo_t x301 = x299.algo;
    size_t x302 = (size_t)0;
    cudnnGetConvolutionForwardWorkspaceSize(x13, x261, x297, x260, x258, x301, &x302);
    float* x303 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x303, (size_t)x302));
    float* x304 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x304, (size_t)(32768 * sizeof(float))));
    float* x305 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x305, (size_t)(8388608 * sizeof(float))));
    float* x306 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x306, (size_t)(8388608 * sizeof(float))));
    float* x307 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x307, (size_t)(8388608 * sizeof(float))));
    float* x308 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x308, (size_t)(8388608 * sizeof(float))));
    float* x309 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x309, (size_t)(8388608 * sizeof(float))));
    float* x310 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x310, (size_t)(8388608 * sizeof(float))));
    float* x311 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x311, (size_t)(8388608 * sizeof(float))));
    float* x312 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x312, (size_t)(8388608 * sizeof(float))));
    float* x313 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x313, (size_t)(8388608 * sizeof(float))));
    float* x314 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x314, (size_t)(8388608 * sizeof(float))));
    float* x315 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x315, (size_t)(32768 * sizeof(float))));
    float* x316 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x316, (size_t)(32768 * sizeof(float))));
    float* x317 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x317, (size_t)(32768 * sizeof(float))));
    float* x318 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x318, (size_t)(73984 * sizeof(float))));
    cudnnConvolutionBwdFilterAlgoPerf_t x319;
    int x320 = 0;
    cudnnFindConvolutionBackwardFilterAlgorithm(x13, x261, x258, x260, x297, 1, &x320, &x319);
    cudnnConvolutionBwdFilterAlgo_t x321 = x319.algo;
    size_t x322 = (size_t)0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(x13, x261, x258, x260, x297, x321, &x322);
    float* x323 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x323, (size_t)x322));
    float* x324 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x324, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionBwdDataAlgoPerf_t x325;
    int x326 = 0;
    cudnnFindConvolutionBackwardDataAlgorithm(x13, x297, x258, x260, x261, 1, &x326, &x325);
    cudnnConvolutionBwdDataAlgo_t x327 = x325.algo;
    size_t x328 = (size_t)0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(x13, x297, x258, x260, x261, x327, &x328);
    float* x329 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x329, (size_t)x328));
    float* x330 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x330, (size_t)(8388608 * sizeof(float))));
    float* x331 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x331, (size_t)(73984 * sizeof(float))));
    cudnnConvolutionBwdFilterAlgoPerf_t x332;
    int x333 = 0;
    cudnnFindConvolutionBackwardFilterAlgorithm(x13, x261, x261, x260, x259, 1, &x333, &x332);
    cudnnConvolutionBwdFilterAlgo_t x334 = x332.algo;
    size_t x335 = (size_t)0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(x13, x261, x261, x260, x259, x334, &x335);
    float* x336 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x336, (size_t)x335));
    float* x337 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x337, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionBwdDataAlgoPerf_t x338;
    int x339 = 0;
    cudnnFindConvolutionBackwardDataAlgorithm(x13, x259, x261, x260, x261, 1, &x339, &x338);
    cudnnConvolutionBwdDataAlgo_t x340 = x338.algo;
    size_t x341 = (size_t)0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(x13, x259, x261, x260, x261, x340, &x341);
    float* x342 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x342, (size_t)x341));
    float* x343 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x343, (size_t)(8388608 * sizeof(float))));
    float* x344 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x344, (size_t)(73984 * sizeof(float))));
    cudnnConvolutionBwdFilterAlgoPerf_t x345;
    int x346 = 0;
    cudnnFindConvolutionBackwardFilterAlgorithm(x13, x261, x261, x260, x259, 1, &x346, &x345);
    cudnnConvolutionBwdFilterAlgo_t x347 = x345.algo;
    size_t x348 = (size_t)0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(x13, x261, x261, x260, x259, x347, &x348);
    float* x349 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x349, (size_t)x348));
    float* x350 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x350, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionBwdDataAlgoPerf_t x351;
    int x352 = 0;
    cudnnFindConvolutionBackwardDataAlgorithm(x13, x259, x261, x260, x261, 1, &x352, &x351);
    cudnnConvolutionBwdDataAlgo_t x353 = x351.algo;
    size_t x354 = (size_t)0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(x13, x259, x261, x260, x261, x353, &x354);
    float* x355 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x355, (size_t)x354));
    float* x356 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x356, (size_t)(8388608 * sizeof(float))));
    float* x357 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x357, (size_t)(73984 * sizeof(float))));
    cudnnConvolutionBwdFilterAlgoPerf_t x358;
    int x359 = 0;
    cudnnFindConvolutionBackwardFilterAlgorithm(x13, x261, x261, x260, x259, 1, &x359, &x358);
    cudnnConvolutionBwdFilterAlgo_t x360 = x358.algo;
    size_t x361 = (size_t)0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(x13, x261, x261, x260, x259, x360, &x361);
    float* x362 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x362, (size_t)x361));
    float* x363 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x363, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionBwdDataAlgoPerf_t x364;
    int x365 = 0;
    cudnnFindConvolutionBackwardDataAlgorithm(x13, x259, x261, x260, x261, 1, &x365, &x364);
    cudnnConvolutionBwdDataAlgo_t x366 = x364.algo;
    size_t x367 = (size_t)0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(x13, x259, x261, x260, x261, x366, &x367);
    float* x368 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x368, (size_t)x367));
    float* x369 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x369, (size_t)(8388608 * sizeof(float))));
    float* x370 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x370, (size_t)(73984 * sizeof(float))));
    cudnnConvolutionBwdFilterAlgoPerf_t x371;
    int x372 = 0;
    cudnnFindConvolutionBackwardFilterAlgorithm(x13, x261, x261, x260, x259, 1, &x372, &x371);
    cudnnConvolutionBwdFilterAlgo_t x373 = x371.algo;
    size_t x374 = (size_t)0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(x13, x261, x261, x260, x259, x373, &x374);
    float* x375 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x375, (size_t)x374));
    float* x376 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x376, (size_t)(8388608 * sizeof(float))));
    cudnnConvolutionBwdDataAlgoPerf_t x377;
    int x378 = 0;
    cudnnFindConvolutionBackwardDataAlgorithm(x13, x259, x261, x260, x261, 1, &x378, &x377);
    cudnnConvolutionBwdDataAlgo_t x379 = x377.algo;
    size_t x380 = (size_t)0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(x13, x259, x261, x260, x261, x379, &x380);
    float* x381 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x381, (size_t)x380));
    float* x382 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x382, (size_t)(8388608 * sizeof(float))));
    float* x383 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x383, (size_t)(73984 * sizeof(float))));
    cudnnConvolutionBwdFilterAlgoPerf_t x384;
    int x385 = 0;
    cudnnFindConvolutionBackwardFilterAlgorithm(x13, x258, x261, x260, x259, 1, &x385, &x384);
    cudnnConvolutionBwdFilterAlgo_t x386 = x384.algo;
    size_t x387 = (size_t)0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(x13, x258, x261, x260, x259, x386, &x387);
    float* x388 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x388, (size_t)x387));
    float* x389 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x389, (size_t)(32768 * sizeof(float))));
    cudnnConvolutionBwdDataAlgoPerf_t x390;
    int x391 = 0;
    cudnnFindConvolutionBackwardDataAlgorithm(x13, x259, x261, x260, x258, 1, &x391, &x390);
    cudnnConvolutionBwdDataAlgo_t x392 = x390.algo;
    size_t x393 = (size_t)0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(x13, x259, x261, x260, x258, x392, &x393);
    float* x394 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x394, (size_t)x393));

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    CUDA_CALL(cudaEventCreate(&start_event));
    CUDA_CALL(cudaEventCreate(&finish_event));
    CUDA_CALL(cudaEventRecord(start_event));
    while (x255 != 40) {
      x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x256, 0, 32768);
      ncclRecv(x256, (size_t)32768, ncclFloat32, x257, x4, x5);
      cudaStreamSynchronize(x5);
      float x395 = 1.0;
      float x396 = 0.0;
      cudnnConvolutionForward(x13, &x395, x258, x256, x259, x237, x260, x265, x267, x266, &x396, x261, x262);
      x51<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x262, x268, 8388608);
      float x397 = 1.0;
      float x398 = 0.0;
      cudnnConvolutionForward(x13, &x397, x261, x268, x259, x240, x260, x272, x274, x273, &x398, x261, x269);
      x51<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x269, x275, 8388608);
      float x399 = 1.0;
      float x400 = 0.0;
      cudnnConvolutionForward(x13, &x399, x261, x275, x259, x243, x260, x279, x281, x280, &x400, x261, x276);
      x51<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x276, x282, 8388608);
      float x401 = 1.0;
      float x402 = 0.0;
      cudnnConvolutionForward(x13, &x401, x261, x282, x259, x246, x260, x286, x288, x287, &x402, x261, x283);
      x51<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x283, x289, 8388608);
      float x403 = 1.0;
      float x404 = 0.0;
      cudnnConvolutionForward(x13, &x403, x261, x289, x259, x249, x260, x293, x295, x294, &x404, x261, x290);
      x51<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x290, x296, 8388608);
      float x405 = 1.0;
      float x406 = 0.0;
      cudnnConvolutionForward(x13, &x405, x261, x296, x297, x252, x260, x301, x303, x302, &x406, x258, x298);
      x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x304, 0, 32768);
      x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x305, 0, 8388608);
      x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x306, 0, 8388608);
      x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x307, 0, 8388608);
      x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x308, 0, 8388608);
      x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x309, 0, 8388608);
      x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x310, 0, 8388608);
      x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x311, 0, 8388608);
      x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x312, 0, 8388608);
      x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x313, 0, 8388608);
      x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x314, 0, 8388608);
      x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x315, 0, 32768);
      x16<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x316, 1, 32768);
      x108<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x316, x298, x317, 32768);
      // end computing RELU_GRAD on GPU for size 32768 and type Float at device (pre-rename) x66 with left_operand x2431 and right_operand x2237
      // begin computing ACCUM on GPU for size 32768 and type Float at device (pre-rename) x66 with base_operand x2418 and addition_operand x2444
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x315, x317, 32768);
      // end allocating gpu array for convolution backward filter workspace
      // begin convolution backward filter pass
      float x407 = 1.0;
      float x408 = 0.0;
      CUDNNCHECK(cudnnConvolutionBackwardFilter(x13, &x407, x261, x296, x258, x315, x260, x321, x323, x322, &x408, x297, x318));
      // end convolution backward filter pass
      cudaStreamSynchronize(0);
      ncclAllReduce(x318, x318, (size_t)73984, ncclFloat32, ncclSum, x11, x5);
      CUDA_CALL(cudaStreamSynchronize(x5));
      // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x66 with base_operand x1886 and addition_operand x2464
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x253, x318, 73984);
      // end allocating gpu array for convolution backward data workspace
      // begin convolution backward data pass
      float x409 = 1.0;
      float x410 = 0.0;
      CUDNNCHECK(cudnnConvolutionBackwardData(x13, &x409, x297, x252, x258, x315, x260, x327, x329, x328, &x410, x261, x324));
      // end convolution backward data pass
      // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x2405 and addition_operand x2516
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x314, x324, 8388608);
      x108<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x314, x290, x330, 8388608);
      // end computing RELU_GRAD on GPU for size 8388608 and type Float at device (pre-rename) x66 with left_operand x2405 and right_operand x2177
      // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x2392 and addition_operand x2561
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x313, x330, 8388608);
      // end allocating gpu array for convolution backward filter workspace
      // begin convolution backward filter pass
      float x411 = 1.0;
      float x412 = 0.0;
      CUDNNCHECK(cudnnConvolutionBackwardFilter(x13, &x411, x261, x289, x261, x313, x260, x334, x336, x335, &x412, x259, x331));
      // end convolution backward filter pass
      cudaStreamSynchronize(0);
      ncclAllReduce(x331, x331, (size_t)73984, ncclFloat32, ncclSum, x11, x5);
      CUDA_CALL(cudaStreamSynchronize(x5));
      // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x66 with base_operand x1850 and addition_operand x2581
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x250, x331, 73984);
      // end allocating gpu array for convolution backward data workspace
      // begin convolution backward data pass
      float x413 = 1.0;
      float x414 = 0.0;
      CUDNNCHECK(cudnnConvolutionBackwardData(x13, &x413, x259, x249, x261, x313, x260, x340, x342, x341, &x414, x261, x337));
      // end convolution backward data pass
      // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x2379 and addition_operand x2633
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x312, x337, 8388608);
      x108<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x312, x283, x343, 8388608);
      // end computing RELU_GRAD on GPU for size 8388608 and type Float at device (pre-rename) x66 with left_operand x2379 and right_operand x2126
      // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x2366 and addition_operand x2678
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x311, x343, 8388608);
      // end allocating gpu array for convolution backward filter workspace
      // begin convolution backward filter pass
      float x415 = 1.0;
      float x416 = 0.0;
      CUDNNCHECK(cudnnConvolutionBackwardFilter(x13, &x415, x261, x282, x261, x311, x260, x347, x349, x348, &x416, x259, x344));
      // end convolution backward filter pass
      cudaStreamSynchronize(0);
      ncclAllReduce(x344, x344, (size_t)73984, ncclFloat32, ncclSum, x11, x5);
      CUDA_CALL(cudaStreamSynchronize(x5));
      // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x66 with base_operand x1814 and addition_operand x2698
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x247, x344, 73984);
      // end allocating gpu array for convolution backward data workspace
      // begin convolution backward data pass
      float x417 = 1.0;
      float x418 = 0.0;
      CUDNNCHECK(cudnnConvolutionBackwardData(x13, &x417, x259, x246, x261, x311, x260, x353, x355, x354, &x418, x261, x350));
      // end convolution backward data pass
      // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x2353 and addition_operand x2750
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x310, x350, 8388608);
      x108<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x310, x276, x356, 8388608);
      // end computing RELU_GRAD on GPU for size 8388608 and type Float at device (pre-rename) x66 with left_operand x2353 and right_operand x2075
      // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x2340 and addition_operand x2795
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x309, x356, 8388608);
      // end allocating gpu array for convolution backward filter workspace
      // begin convolution backward filter pass
      float x419 = 1.0;
      float x420 = 0.0;
      CUDNNCHECK(cudnnConvolutionBackwardFilter(x13, &x419, x261, x275, x261, x309, x260, x360, x362, x361, &x420, x259, x357));
      // end convolution backward filter pass
      cudaStreamSynchronize(0);
      ncclAllReduce(x357, x357, (size_t)73984, ncclFloat32, ncclSum, x11, x5);
      CUDA_CALL(cudaStreamSynchronize(x5));
      // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x66 with base_operand x1778 and addition_operand x2815
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x244, x357, 73984);
      // end allocating gpu array for convolution backward data workspace
      // begin convolution backward data pass
      float x421 = 1.0;
      float x422 = 0.0;
      CUDNNCHECK(cudnnConvolutionBackwardData(x13, &x421, x259, x243, x261, x309, x260, x366, x368, x367, &x422, x261, x363));
      // end convolution backward data pass
      // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x2327 and addition_operand x2867
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x308, x363, 8388608);
      x108<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x308, x269, x369, 8388608);
      // end computing RELU_GRAD on GPU for size 8388608 and type Float at device (pre-rename) x66 with left_operand x2327 and right_operand x2024
      // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x2314 and addition_operand x2912
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x307, x369, 8388608);
      // end allocating gpu array for convolution backward filter workspace
      // begin convolution backward filter pass
      float x423 = 1.0;
      float x424 = 0.0;
      CUDNNCHECK(cudnnConvolutionBackwardFilter(x13, &x423, x261, x268, x261, x307, x260, x373, x375, x374, &x424, x259, x370));
      // end convolution backward filter pass
      cudaStreamSynchronize(0);
      ncclAllReduce(x370, x370, (size_t)73984, ncclFloat32, ncclSum, x11, x5);
      CUDA_CALL(cudaStreamSynchronize(x5));
      // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x66 with base_operand x1742 and addition_operand x2932
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x241, x370, 73984);
      // end allocating gpu array for convolution backward data workspace
      // begin convolution backward data pass
      float x425 = 1.0;
      float x426 = 0.0;
      CUDNNCHECK(cudnnConvolutionBackwardData(x13, &x425, x259, x240, x261, x307, x260, x379, x381, x380, &x426, x261, x376));
      // end convolution backward data pass
      // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x2301 and addition_operand x2984
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x306, x376, 8388608);
      x108<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x306, x262, x382, 8388608);
      // end computing RELU_GRAD on GPU for size 8388608 and type Float at device (pre-rename) x66 with left_operand x2301 and right_operand x1973
      // begin computing ACCUM on GPU for size 8388608 and type Float at device (pre-rename) x66 with base_operand x2288 and addition_operand x3029
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x305, x382, 8388608);
      // end allocating gpu array for convolution backward filter workspace
      // begin convolution backward filter pass
      float x427 = 1.0;
      float x428 = 0.0;
      CUDNNCHECK(cudnnConvolutionBackwardFilter(x13, &x427, x258, x256, x261, x305, x260, x386, x388, x387, &x428, x259, x383));
      // end convolution backward filter pass
      cudaStreamSynchronize(0);
      ncclAllReduce(x383, x383, (size_t)73984, ncclFloat32, ncclSum, x11, x5);
      CUDA_CALL(cudaStreamSynchronize(x5));
      // begin computing ACCUM on GPU for size 73984 and type Float at device (pre-rename) x66 with base_operand x1706 and addition_operand x3049
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x238, x383, 73984);
      // end allocating gpu array for convolution backward data workspace
      // begin convolution backward data pass
      float x429 = 1.0;
      float x430 = 0.0;
      CUDNNCHECK(cudnnConvolutionBackwardData(x13, &x429, x259, x237, x261, x305, x260, x392, x394, x393, &x430, x258, x389));
      // end convolution backward data pass
      // begin computing ACCUM on GPU for size 32768 and type Float at device (pre-rename) x66 with base_operand x2275 and addition_operand x3101
      CUDA_CALL(cudaSetDevice(x12));
      x116<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x304, x389, 32768);
      // end computing ACCUM on GPU for size 32768 and type Float at device (pre-rename) x66 with base_operand x2275 and addition_operand x3101
      cudaStreamSynchronize(0);
      NCCLCHECK(ncclSend(x304, (size_t)32768, ncclFloat32, x257, x4, x5));
      // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x1696, grad x1706, and momentum x1719
      CUDA_CALL(cudaSetDevice(x12));
      x194<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x237, x238, x239, 73984);
      // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x1696, grad x1706, and momentum x1719
      // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x1732, grad x1742, and momentum x1755
      CUDA_CALL(cudaSetDevice(x12));
      x194<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x240, x241, x242, 73984);
      // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x1732, grad x1742, and momentum x1755
      // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x1768, grad x1778, and momentum x1791
      CUDA_CALL(cudaSetDevice(x12));
      x194<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x243, x244, x245, 73984);
      // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x1768, grad x1778, and momentum x1791
      // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x1804, grad x1814, and momentum x1827
      CUDA_CALL(cudaSetDevice(x12));
      x194<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x246, x247, x248, 73984);
      // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x1804, grad x1814, and momentum x1827
      // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x1840, grad x1850, and momentum x1863
      CUDA_CALL(cudaSetDevice(x12));
      x194<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x249, x250, x251, 73984);
      // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x1840, grad x1850, and momentum x1863
      // begin computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x1876, grad x1886, and momentum x1899
      CUDA_CALL(cudaSetDevice(x12));
      x194<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x252, x253, x254, 73984);
      // end computing SGD on GPU for size 73984 and type Float at device (pre-name) x66 with weight x1876, grad x1886, and momentum x1899
      x255 = x255 + 1;
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
