/*
by Qin Yu, Apr 2019
*/

#include <fstream>
using namespace std;

#include "hip/hip_runtime.h"
#include <hip/hip_cooperative_groups.h>
namespace cg = cooperative_groups;

#define CUPRINTF(fmt, ...)                                                     \
  printf("[%d, %d]:\t" fmt, blockIdx.y *gridDim.x + blockIdx.x,                \
         threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +    \
             threadIdx.x,                                                      \
         __VA_ARGS__) // Idiom, not used, put here for convenient debugging.

__global__ void kernel_minibatch(int *iters, float *alpha, float *sigma,
                                 float *K, int *y, int l, int C) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  int can_stop = 0;
  float delta_ = 0;
  __shared__ float delta;
  float last_alpha_j, last_alpha;

  int counter = 0;
  while (true) {
    counter++;
    last_alpha_j = alpha[j];
    for (int i = 0; i < l; i++) {
      if (j == i) {
        last_alpha = alpha[i];
        delta = 1 / K[i * l + i] * (1 - y[i] * sigma[i]);
        alpha[i] += delta;
        if (alpha[i] < 0) {
          alpha[i] = 0;
          delta = 0 - last_alpha;
        }
        if (alpha[i] > C) {
          alpha[i] = C;
          delta = C - last_alpha;
        }
      }
      __syncthreads();
      sigma[j] += delta * y[i] * K[i * l + j];
    }
    can_stop = 0;
    delta_ = alpha[j] - last_alpha_j;
    if (-0.0001f < delta_ && delta_ < 0.0001f)
      can_stop = 1;
    // CUPRINTF("%d, %9.6f, %9.6f, %9.6f, %d\n", counter, alpha[j],
    // last_alpha_j, delta_, can_stop);
    if (__syncthreads_and(can_stop) > 0) {
      if (j == 1) {
        // CUPRINTF("iters = %d\n", counter);
        iters[0] = counter;
      }
      break;
    }
  }
}

extern "C" __global__ void kernel_minibatch_g(int *iters, float *alpha,
                                              float *sigma, float *K, int *y,
                                              int *d, int ddim, float *delta,
                                              int l, int C) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < l) {
    cg::grid_group grid = cg::this_grid();
    // if (j == l-1) CUPRINTF("l = %d, C = %d\n", l, C);

    int can_break = 0;
    int can_stop = 0;
    float delta_ = 0;
    float last_alpha_j, last_alpha;

    int counter = 0;
    while (true) {
      // for (int counter = 0; counter < 2000; counter++) {
      counter++;
      if (threadIdx.x == 1)
        d[blockIdx.x] = 0;
      last_alpha_j = alpha[j];
      for (uint32_t i = 0; i < l; i++) {
        if (j == i) {
          // if (threadIdx.x == i) {  // This was a big big bug
          last_alpha = alpha[i];
          delta[0] = 1 / K[i * l + i] * (1 - y[i] * sigma[i]);
          alpha[i] += delta[0];
          // alpha[i] += delta;
          if (alpha[i] < 0) {
            alpha[i] = 0;
            delta[0] = 0 - last_alpha;
          }
          if (alpha[i] > C) {
            alpha[i] = C;
            delta[0] = C - last_alpha;
          }
        }
        cg::sync(grid);
        sigma[j] += delta[0] * y[i] * K[i * l + j];
      }
      can_stop = 0;
      delta_ = alpha[j] - last_alpha_j;
      if (-0.0001f < delta_ && delta_ < 0.0001f)
        can_stop = 1;
      if (__syncthreads_and(can_stop) > 0)
        if (threadIdx.x == 1)
          d[blockIdx.x] = 1;
      cg::sync(grid);
      can_break = 0;
      for (int i = 0; i < ddim; i++) {
        can_break += d[i];
      }
      // if (j == 1) CUPRINTF("iters = %d\n", counter);
      if (can_break == ddim) {
        if (j == 1) {
          // CUPRINTF("iters = %d\n", counter);
          iters[0] = counter;
        }
        // cg::sync(grid);
        break;
      }
    }
  }
}

// Helper function for using CUDA to update sigma in parallel:
hipError_t kernel_minibatch_wrapper(int *iters, float *alpha, float *sigma,
                                     float *K, int *y, int l, int C) {
  // int *dev_iters = 0;
  // float *dev_alpha = 0;
  // float *dev_sigma = 0;
  // float *dev_K = 0;
  // int *dev_y = 0;
  int *dev_block_done = 0;
  float *dev_delta = 0;

  const int block_dim_max = 1024;
  int block_dimension = block_dim_max;
  int grid_dimension = (l - 1) / block_dim_max + 1;
  dim3 block(block_dimension);
  dim3 grid(grid_dimension);

  // void *args[10] = {
  //     &dev_iters,      &dev_alpha,      &dev_sigma, &dev_K, &dev_y,
  //     &dev_block_done, &grid_dimension, &dev_delta, &l,     &C};

  void *args[10] = {&iters,          &alpha,          &sigma,     &K, &y,
                    &dev_block_done, &grid_dimension, &dev_delta, &l, &C};

  hipError_t hipStatus;

  // Allocate GPU buffers for all vectors:
  // hipStatus = hipMalloc(&dev_iters, sizeof(int));
  // if (hipStatus != hipSuccess) {
  //   fprintf(stderr, "hipMalloc failed!");
  //   goto Error1;
  // }
  // hipStatus = hipMalloc(&dev_alpha, l * sizeof(float));
  // if (hipStatus != hipSuccess) {
  //   fprintf(stderr, "hipMalloc failed!");
  //   goto Error2;
  // }
  // hipStatus = hipMalloc(&dev_sigma, l * sizeof(float));
  // if (hipStatus != hipSuccess) {
  //   fprintf(stderr, "hipMalloc failed!");
  //   goto Error3;
  // }
  // hipStatus = hipMalloc(&dev_K, l * l * sizeof(float));
  // if (hipStatus != hipSuccess) {
  //   fprintf(stderr, "hipMalloc failed!");
  //   goto Error4;
  // }
  // hipStatus = hipMalloc(&dev_y, l * sizeof(int));
  // if (hipStatus != hipSuccess) {
  //   fprintf(stderr, "hipMalloc failed!");
  //   goto Error5;
  // }
  hipStatus = hipMallocManaged(&dev_block_done, grid_dimension * sizeof(int));
  if (hipStatus != hipSuccess) {
    fprintf(stderr, "hipMallocManaged failed!");
    goto Error5;
  }
  hipStatus = hipMallocManaged(&dev_delta, 1 * sizeof(float));
  if (hipStatus != hipSuccess) {
    fprintf(stderr, "hipMallocManaged failed!");
    goto Error5;
  }

  // // Copy input vectors from host memory to GPU buffers.
  // hipStatus =
  //     hipMemcpy(dev_K, K, l * l * sizeof(float), hipMemcpyHostToDevice);
  // if (hipStatus != hipSuccess) {
  //   fprintf(stderr, "hipMemcpy failed!");
  //   goto Error5;
  // }
  // hipStatus = hipMemcpy(dev_y, y, l * sizeof(int), hipMemcpyHostToDevice);
  // if (hipStatus != hipSuccess) {
  //   fprintf(stderr, "hipMemcpy failed!");
  //   goto Error5;
  // }

  // printf("READY TO CALL KERNEL\n");
  hipStatus =
      hipLaunchCooperativeKernel((void *)kernel_minibatch_g, grid, block, args,
                                  sizeof(float), hipStream_t(0));
  if (hipStatus != hipSuccess) {
    fprintf(stderr, "kernel_minibatch_g launch failed: %s\n",
            hipGetErrorString(hipStatus));
    goto Error5;
  }

  // Check for any errors launching the kernel
  hipStatus = hipGetLastError();
  if (hipStatus != hipSuccess) {
    fprintf(stderr, "kernel_minibatch_g launch failed: %s\n",
            hipGetErrorString(hipStatus));
    goto Error5;
  }

  // hipDeviceSynchronize waits for the kernel to finish, and returns
  // any errors encountered during the launch.
  hipStatus = hipDeviceSynchronize();
  if (hipStatus != hipSuccess) {
    fprintf(stderr,
            "hipDeviceSynchronize returned error code %d after launching "
            "addKernel!\n",
            hipStatus);
    goto Error5;
  }

  // Copy output vector from GPU buffer to host memory.
  // hipStatus =
  //     hipMemcpy(iters, dev_iters, sizeof(int), hipMemcpyDeviceToHost);
  // if (hipStatus != hipSuccess) {
  //   fprintf(stderr, "hipMemcpy failed!");
  //   goto Error5;
  // }

  // hipStatus =
  //     hipMemcpy(alpha, dev_alpha, l * sizeof(float),
  //     hipMemcpyDeviceToHost);
  // if (hipStatus != hipSuccess) {
  //   fprintf(stderr, "hipMemcpy failed!");
  //   goto Error5;
  // }

  // hipStatus =
  //     hipMemcpy(sigma, dev_sigma, l * sizeof(float),
  //     hipMemcpyDeviceToHost);
  // if (hipStatus != hipSuccess) {
  //   fprintf(stderr, "hipMemcpy failed!");
  //   goto Error5;
  // }
  // Error6:
  hipFree(dev_block_done);
  hipFree(dev_delta);
Error5:
  //   hipFree(dev_y);
  // Error4:
  //   hipFree(dev_K);
  // Error3:
  //   hipFree(dev_sigma);
  // Error2:
  //   hipFree(dev_alpha);
  // Error1:
  //   hipFree(dev_iters);
  // Error0:
  return hipStatus;
}

hipError_t kernel_minibatch_block_wrapper(int *iters, float *alpha,
                                           float *sigma, float *K, int *y,
                                           int l, int C) {
  // int *dev_iters = 0;
  // float *dev_alpha = 0;
  // float *dev_sigma = 0;
  // float *dev_K = 0;
  // int *dev_y = 0;

  dim3 grid(1);
  dim3 block(l);
  void *args[7] = {&iters, &alpha, &sigma, &K, &y, &l, &C};

  hipError_t hipStatus;

  // Allocate GPU buffers for all vectors:
  // hipStatus = hipMalloc(&dev_iters, sizeof(int));
  // if (hipStatus != hipSuccess) {
  //   fprintf(stderr, "hipMalloc failed!");
  //   goto Error1;
  // }
  // hipStatus = hipMalloc(&dev_alpha, l * sizeof(float));
  // if (hipStatus != hipSuccess) {
  //   fprintf(stderr, "hipMalloc failed!");
  //   goto Error2;
  // }
  // hipStatus = hipMalloc(&dev_sigma, l * sizeof(float));
  // if (hipStatus != hipSuccess) {
  //   fprintf(stderr, "hipMalloc failed!");
  //   goto Error3;
  // }
  // hipStatus = hipMalloc(&dev_K, l * l * sizeof(float));
  // if (hipStatus != hipSuccess) {
  //   fprintf(stderr, "hipMalloc failed!");
  //   goto Error4;
  // }
  // hipStatus = hipMalloc(&dev_y, l * sizeof(int));
  // if (hipStatus != hipSuccess) {
  //   fprintf(stderr, "hipMalloc failed!");
  //   goto Error5;
  // }

  // Copy input vectors from host memory to GPU buffers.
  // hipStatus =
  //     hipMemcpy(dev_K, K, l * l * sizeof(float), hipMemcpyHostToDevice);
  // if (hipStatus != hipSuccess) {
  //   fprintf(stderr, "hipMemcpy failed!");
  //   goto Error5;
  // }
  // hipStatus = hipMemcpy(dev_y, y, l * sizeof(int), hipMemcpyHostToDevice);
  // if (hipStatus != hipSuccess) {
  //   fprintf(stderr, "hipMemcpy failed!");
  //   goto Error5;
  // }

  // printf("READY TO CALL KERNEL\n");
  hipStatus = hipLaunchKernel((void *)kernel_minibatch, grid, block, args,
                                sizeof(float), hipStream_t(0));
  if (hipStatus != hipSuccess) {
    fprintf(stderr, "kernel_minibatch launch failed: %s\n",
            hipGetErrorString(hipStatus));
    goto Error5;
  }

  // Check for any errors launching the kernel
  hipStatus = hipGetLastError();
  if (hipStatus != hipSuccess) {
    fprintf(stderr, "kernel_minibatch_g launch failed: %s\n",
            hipGetErrorString(hipStatus));
    goto Error5;
  }

  // hipDeviceSynchronize waits for the kernel to finish, and returns
  // any errors encountered during the launch.
  hipStatus = hipDeviceSynchronize();
  if (hipStatus != hipSuccess) {
    fprintf(stderr,
            "hipDeviceSynchronize returned error code %d after launching "
            "addKernel!\n",
            hipStatus);
    goto Error5;
  }

// Copy output vector from GPU buffer to host memory.
// hipStatus =
//     hipMemcpy(iters, dev_iters, sizeof(int), hipMemcpyDeviceToHost);
// if (hipStatus != hipSuccess) {
//   fprintf(stderr, "hipMemcpy failed!");
//   goto Error5;
// }

// hipStatus =
//     hipMemcpy(alpha, dev_alpha, l * sizeof(float),
//     hipMemcpyDeviceToHost);
// if (hipStatus != hipSuccess) {
//   fprintf(stderr, "hipMemcpy failed!");
//   goto Error5;
// }

// hipStatus =
//     hipMemcpy(sigma, dev_sigma, l * sizeof(float),
//     hipMemcpyDeviceToHost);
// if (hipStatus != hipSuccess) {
//   fprintf(stderr, "hipMemcpy failed!");
//   goto Error5;
// }
Error5:
  //   hipFree(dev_y);
  // Error4:
  //   hipFree(dev_K);
  // Error3:
  //   hipFree(dev_sigma);
  // Error2:
  //   hipFree(dev_alpha);
  // Error1:
  //   hipFree(dev_iters);
  // Error0:
  return hipStatus;
}
