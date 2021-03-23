 /* Device code. */
 #include "gauss_eliminate.h"
 #include <stdio.h>
 #include <math.h>

// FIX ME
 __global__ void division__kernel(float *U, int matrix_dim, int k)
 {
    float pivot = U[matrix_dim * k + k];

    int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    while (k+1 + tid < matrix_dim){
      float temp = U[k * matrix_dim + k+1 + tid];
      U[k * matrix_dim + k+1 + tid] = (float) temp/pivot; 
      tid = tid + stride;
    }
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0)
      U[matrix_dim * k + k] = 1;
    return;
 }
 
 // FIX ME
 __global__ void elimination__kernel(float *U, int matrix_dim, int k)
 {

    int row = k+1 + blockIdx.x;
    
    while (row < matrix_dim){
      int tid = threadIdx.x;
      while (k+1 + tid < matrix_dim){
        // a - bc = x -> x = b(a/b - c)
        float a = U[row * matrix_dim + k+1 + tid];
        float b = __fmul_rn(U[row * matrix_dim + k], U[k * matrix_dim + k+1 +tid]);
        U[row * matrix_dim + k+1 + tid] = a - b;
        tid = tid + blockDim.x;
      }
      __syncthreads();
      if (threadIdx.x == 0){
        U[row * matrix_dim + k] = 0;
      }
      row += gridDim.x;
    }
    
    return;
 }
 