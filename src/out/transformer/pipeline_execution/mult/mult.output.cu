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
__global__ void x17(float* x18, float x19, int x20) {
  // begin generating kernel function for FILL of type Float
  int x21 = gridDim.x * blockDim.x;
  int x22 = threadIdx.x + blockIdx.x * blockDim.x;
  while (x22 < x20) {
    x18[x22] = x19;
    x22 = x22 + x21;
  }
  // end generating kernel function for FILL of type Float
}
__global__ void x26(float* x27, float* x28, float* x29, int x30) {
  // begin generating kernel function for MULT of type Float
  int x31 = gridDim.x * blockDim.x;
  int x32 = threadIdx.x + blockIdx.x * blockDim.x;
  while (x32 < x30) {
    int x33 = x32;
    x29[x33] = x27[x33] * x28[x33];
    x32 = x32 + x31;
  }
  // end generating kernel function for MULT of type Float
}
__global__ void x35(float* x36, float* x37, int x38) {
  // begin generating kernel function for ACCUM of type Float
  int x39 = gridDim.x * blockDim.x;
  int x40 = threadIdx.x + blockIdx.x * blockDim.x;
  while (x40 < x38) {
    int x41 = x40;
    x36[x41] = x36[x41] + x37[x41];
    x40 = x40 + x39;
  }
  // end generating kernel function for ACCUM of type Float
}
__global__ void x42(float* x43, float* x44, float* x45, int x46) {
  // begin generating kernel function for SGD of type Float
  int x47 = gridDim.x * blockDim.x;
  int x48 = threadIdx.x + blockIdx.x * blockDim.x;
  while (x48 < x46) {
    int x49 = x48;
    float x50 = x45[x49] * 0.5 + x44[x49];
    x43[x49] = x43[x49] - x50 * 1.0E-4;
    x45[x49] = x50;
    x48 = x48 + x47;
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
  if (x12 >= 0 && x12 < 2) {
    int x13 = x9;
    // begin initializing GPU array of size 512 and type Float
    float* x14 = (float*)malloc(512 * sizeof(float));
    float* x15 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x15, (size_t)(512 * sizeof(float))));
    scan_float_array(x14, 512, "golden/weight_rank_%d.data", x13);
    CUDA_CALL(cudaMemcpy(x15, x14, (size_t)(512 * sizeof(float)), cudaMemcpyHostToDevice));
    // end initializing GPU array of size 512 and type Float
    // begin initializing fixed GPU array of size 512 and type Float and device (pre-rename) x64
    float* x16 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x16, (size_t)(512 * sizeof(float))));
    x17<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x16, 0, 512);
    // end initializing fixed GPU array of size 512 and type Float and device (pre-rename) x64
    // begin initializing fixed GPU array of size 512 and type Float and device (pre-rename) x64
    float* x23 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x23, (size_t)(512 * sizeof(float))));
    x17<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x23, 0, 512);
    // end initializing fixed GPU array of size 512 and type Float and device (pre-rename) x64
    // begin initializing fixed GPU array of size 1536 and type Float and device (pre-rename) x64
    float* x24 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x24, (size_t)(1536 * sizeof(float))));
    x17<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x24, 0, 1536);
    // end initializing fixed GPU array of size 1536 and type Float and device (pre-rename) x64
    int x25 = 0;
    int x34 = x12 + 2;
    while (x25 != 3) {
      int x51 = 0;
      while (x51 != 3) {
        // begin initializing GPU array of size 512 and type Float
        float* x52 = (float*)malloc(512 * sizeof(float));
        float* x53 = (float*)malloc(0 * sizeof(float));
        CUDA_CALL(cudaMalloc(&x53, (size_t)(512 * sizeof(float))));
        scan_float_array(x52, 512, "golden/input1_rank_%d.data", x13);
        CUDA_CALL(cudaMemcpy(x53, x52, (size_t)(512 * sizeof(float)), cudaMemcpyHostToDevice));
        // end initializing GPU array of size 512 and type Float
        CUDA_CALL(cudaMemcpy(x24 + 512 * x51, x53, (size_t)(512 * sizeof(float)), cudaMemcpyDeviceToDevice));
        // begin computing MULT on GPU for size 512 and type Float at device (pre-rename) x64 with left_operand x162 and right_operand x75
        float* x54 = (float*)malloc(0 * sizeof(float));
        CUDA_CALL(cudaMalloc(&x54, (size_t)(512 * sizeof(float))));
        x26<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x53, x15, x54, 512);
        // end computing MULT on GPU for size 512 and type Float at device (pre-rename) x64 with left_operand x162 and right_operand x75
        NCCLCHECK(ncclSend(x54, (size_t)512, ncclFloat32, x34, x4, x5));
        CUDA_CALL(cudaStreamSynchronize(x5));
        x51 = x51 + 1;
      }
      int x55 = 0;
      while (x55 != 3) {
        // begin initializing fixed GPU array of size 512 and type Float and device (pre-rename) x64
        float* x56 = (float*)malloc(0 * sizeof(float));
        CUDA_CALL(cudaMalloc(&x56, (size_t)(512 * sizeof(float))));
        x17<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x56, 0, 512);
        // end initializing fixed GPU array of size 512 and type Float and device (pre-rename) x64
        NCCLCHECK(ncclRecv(x56, (size_t)512, ncclFloat32, x34, x4, x5));
        CUDA_CALL(cudaStreamSynchronize(x5));
        // begin computing MULT on GPU for size 512 and type Float at device (pre-rename) x64 with left_operand x244 and right_operand x246
        float* x57 = (float*)malloc(0 * sizeof(float));
        CUDA_CALL(cudaMalloc(&x57, (size_t)(512 * sizeof(float))));
        x26<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x24 + 512 * x55, x56, x57, 512);
        // end computing MULT on GPU for size 512 and type Float at device (pre-rename) x64 with left_operand x244 and right_operand x246
        // begin computing ACCUM on GPU for size 512 and type Float at device (pre-rename) x64 with base_operand x90 and addition_operand x263
        x35<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x16, x57, 512);
        // end computing ACCUM on GPU for size 512 and type Float at device (pre-rename) x64 with base_operand x90 and addition_operand x263
        x55 = x55 + 1;
      }
      // begin computing SGD on GPU for size 512 and type Float at device (pre-name) x64 with weight x75, grad x90, and momentum x128
      x42<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x15, x16, x23, 512);
      // end computing SGD on GPU for size 512 and type Float at device (pre-name) x64 with weight x75, grad x90, and momentum x128
      // begin checking GPU array of size 512 and type Float
      float* x58 = (float*)malloc(512 * sizeof(float));
      CUDA_CALL(cudaMemcpy(x58, x16, (size_t)(512 * sizeof(float)), cudaMemcpyDeviceToHost));
      check_float_array_with_file(x58, 512, "golden/weight_grad_rank_%d.data", x13);
      // end checking GPU array of size 512 and type Float
      // begin checking GPU array of size 512 and type Float
      float* x59 = (float*)malloc(512 * sizeof(float));
      CUDA_CALL(cudaMemcpy(x59, x15, (size_t)(512 * sizeof(float)), cudaMemcpyDeviceToHost));
      check_float_array_with_file(x59, 512, "golden/weight_rank_%d.data", x13);
      // end checking GPU array of size 512 and type Float
      x25 = x25 + 1;
    }
  }
  if (x12 >= 2 && x12 < 4) {
    int x13 = x9;
    // begin initializing fixed GPU array of size 1536 and type Float and device (pre-rename) x64
    float* x60 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x60, (size_t)(1536 * sizeof(float))));
    x17<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x60, 0, 1536);
    // end initializing fixed GPU array of size 1536 and type Float and device (pre-rename) x64
    // begin initializing fixed GPU array of size 1536 and type Float and device (pre-rename) x64
    float* x61 = (float*)malloc(0 * sizeof(float));
    CUDA_CALL(cudaMalloc(&x61, (size_t)(1536 * sizeof(float))));
    x17<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x61, 0, 1536);
    // end initializing fixed GPU array of size 1536 and type Float and device (pre-rename) x64
    int x62 = 0;
    int x63 = x12 - 2;
    while (x62 != 3) {
      int x64 = 0;
      while (x64 != 3) {
        // begin initializing GPU array of size 512 and type Float
        float* x65 = (float*)malloc(512 * sizeof(float));
        float* x66 = (float*)malloc(0 * sizeof(float));
        CUDA_CALL(cudaMalloc(&x66, (size_t)(512 * sizeof(float))));
        scan_float_array(x65, 512, "golden/input2_rank_%d.data", x13);
        CUDA_CALL(cudaMemcpy(x66, x65, (size_t)(512 * sizeof(float)), cudaMemcpyHostToDevice));
        // end initializing GPU array of size 512 and type Float
        int x67 = 512 * x64;
        CUDA_CALL(cudaMemcpy(x61 + x67, x66, (size_t)(512 * sizeof(float)), cudaMemcpyDeviceToDevice));
        // begin initializing fixed GPU array of size 512 and type Float and device (pre-rename) x64
        float* x68 = (float*)malloc(0 * sizeof(float));
        CUDA_CALL(cudaMalloc(&x68, (size_t)(512 * sizeof(float))));
        x17<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x68, 0, 512);
        // end initializing fixed GPU array of size 512 and type Float and device (pre-rename) x64
        NCCLCHECK(ncclRecv(x68, (size_t)512, ncclFloat32, x63, x4, x5));
        CUDA_CALL(cudaStreamSynchronize(x5));
        CUDA_CALL(cudaMemcpy(x60 + x67, x68, (size_t)(512 * sizeof(float)), cudaMemcpyDeviceToDevice));
        x64 = x64 + 1;
      }
      int x69 = 0;
      while (x69 != 3) {
        // begin initializing fixed GPU array of size 512 and type Float and device (pre-rename) x64
        float* x70 = (float*)malloc(0 * sizeof(float));
        CUDA_CALL(cudaMalloc(&x70, (size_t)(512 * sizeof(float))));
        x17<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x70, 0, 512);
        // end initializing fixed GPU array of size 512 and type Float and device (pre-rename) x64
        // begin initializing fixed GPU array of size 512 and type Float and device (pre-rename) x64
        float* x71 = (float*)malloc(0 * sizeof(float));
        CUDA_CALL(cudaMalloc(&x71, (size_t)(512 * sizeof(float))));
        x17<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x71, 1, 512);
        // end initializing fixed GPU array of size 512 and type Float and device (pre-rename) x64
        // begin computing MULT on GPU for size 512 and type Float at device (pre-rename) x64 with left_operand x487 and right_operand x489
        float* x72 = (float*)malloc(0 * sizeof(float));
        CUDA_CALL(cudaMalloc(&x72, (size_t)(512 * sizeof(float))));
        x26<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x61 + 512 * x69, x71, x72, 512);
        // end computing MULT on GPU for size 512 and type Float at device (pre-rename) x64 with left_operand x487 and right_operand x489
        // begin computing ACCUM on GPU for size 512 and type Float at device (pre-rename) x64 with base_operand x476 and addition_operand x500
        x35<<<dim3(28, 1, 1), dim3(512, 1, 1)>>>(x70, x72, 512);
        // end computing ACCUM on GPU for size 512 and type Float at device (pre-rename) x64 with base_operand x476 and addition_operand x500
        NCCLCHECK(ncclSend(x70, (size_t)512, ncclFloat32, x63, x4, x5));
        CUDA_CALL(cudaStreamSynchronize(x5));
        x69 = x69 + 1;
      }
      x62 = x62 + 1;
    }
  }
  NCCLCHECK(ncclCommDestroy(x11));
  NCCLCHECK(ncclCommDestroy(x4));
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
