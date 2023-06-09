/*
by Qin Yu, Apr 2019
*/

#include <hip/hip_runtime.h>
#include <stdio.h>

#define CUPRINTF(fmt, ...)                                                     \
  printf("[%d, %d]:\t" fmt, blockIdx.y *gridDim.x + blockIdx.x,                \
         threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +    \
             threadIdx.x,                                                      \
         __VA_ARGS__) // Idiom, not used, put here for convenient debugging.

__global__ void kernel_kernel_matrix(float *B, float *K, uint8_t *X, int l) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < l && j < l) {
    const int offset_i = i * 785;
    const int offset_j = j * 785;
    float squared_euclidean_distance = 0;
    for (int k = 0; k < 785; k++) {
      squared_euclidean_distance += B[X[offset_i + k] * 256 + X[offset_j + k]];
    }
    K[i * l + j] = expf(-0.015f * squared_euclidean_distance);
  }
}

__global__ void kernel_kernel_matrix_ts(float *B, float *K, uint8_t *X1,
                                        uint8_t *X2, int l1, int l2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < l1 && j < l2) {
    const int offset_i = i * 785;
    const int offset_j = j * 785;
    float squared_euclidean_distance = 0;
    for (int k = 0; k < 785; k++) {
      squared_euclidean_distance +=
          B[X1[offset_i + k] * 256 + X2[offset_j + k]];
    }
    K[i * l2 + j] = expf(-0.015f * squared_euclidean_distance);
  }
}

hipError_t compute_kernel_matrix(uint32_t n_features,
                                  uint32_t number_of_data_1v1, uint8_t *dev_X,
                                  float *K, float *B) {
  // GPU kernel launch - set device variables:
  // float *dev_B = 0;
  // float *dev_K = 0;
  // uint8_t *dev_X = 0;
  // If use `hipLaunchKernel()`, need `args[]` here:
  // void  *args[4] = { &dev_B, &dev_K, &dev_X, &number_of_data_1v1 };

  // GPU kernel launch - kernel dimensions:
  dim3 block(32, 32);
  dim3 grid;
  grid.x = (number_of_data_1v1 - 1) / block.x + 1;
  grid.y = (number_of_data_1v1 - 1) / block.y + 1;

  hipError_t hipStatus;

  // GPU kernel launch - allocate device variables:
  // hipStatus = hipMalloc((void **)&dev_B, 256 * 256 * sizeof(float));
  // if (hipStatus != hipSuccess) {
  //   fprintf(stderr, "hipMalloc failed!");
  //   goto Error1;
  // }
  // hipStatus = hipMalloc(
  //     (void **)&dev_K, number_of_data_1v1 * number_of_data_1v1 *
  //     sizeof(float));
  // if (hipStatus != hipSuccess) {
  //   fprintf(stderr, "hipMalloc failed!");
  //   goto Error2;
  // }
  // hipStatus = hipMalloc((void **)&dev_X,
  //                         number_of_data_1v1 * n_features * sizeof(uint8_t));
  // if (hipStatus != hipSuccess) {
  //   fprintf(stderr, "hipMalloc failed!");
  //   goto Error3;
  // }

  // GPU kernel launch - copy to device variables:
  // hipStatus =
  //     hipMemcpy(dev_B, B, 256 * 256 * sizeof(float),
  //     hipMemcpyHostToDevice);
  // if (hipStatus != hipSuccess) {
  //   fprintf(stderr, "hipMemcpy failed!");
  //   goto Error3;
  // }
  // for (uint32_t i = 0; i < number_of_data_1v1; i++) {
  //   hipStatus =
  //       hipMemcpy(dev_X + i * n_features, train_images_1v1[i],
  //                  n_features * sizeof(uint8_t), hipMemcpyHostToDevice);
  //   if (hipStatus != hipSuccess) {
  //     fprintf(stderr, "hipMemcpy failed!");
  //     goto Error3;
  //   }
  // }

  // GPU kernel launch:
  kernel_kernel_matrix<<<grid, block>>>(B, K, dev_X, number_of_data_1v1);
  hipStatus = hipGetLastError();
  if (hipStatus != hipSuccess) {
    fprintf(stderr, "kernel_kernel_matrix launch failed: %s\n",
            hipGetErrorString(hipStatus));
    goto Error3;
  }

  hipStatus = hipDeviceSynchronize();
  if (hipStatus != hipSuccess) {
    fprintf(stderr,
            "hipDeviceSynchronize returned error code %d after launching "
            "addKernel!\n",
            hipStatus);
    goto Error3;
  }

// GPU kernel launch - copy from device variables:
// hipStatus = hipMemcpy(K, dev_K, number_of_data_1v1 * number_of_data_1v1 *
//                                       sizeof(float),
//                         hipMemcpyDeviceToHost);
// if (hipStatus != hipSuccess) {
//   fprintf(stderr, "hipMemcpy failed!");
//   goto Error3;
// }

// GPU kernel launch - free device variables:
Error3:
  // hipFree(dev_X);
  // Error2:
  // hipFree(dev_K);
  // Error1:
  // hipFree(dev_B);
  // Error0:
  return hipStatus;
}

hipError_t compute_kernel_matrix_ts(uint32_t n_features,
                                     uint32_t number_of_data_1v1_ts,
                                     uint8_t *dev_X1,
                                     uint32_t number_of_data_1v1,
                                     uint8_t *dev_X2, float *K, float *B) {
  // GPU kernel launch - set device variables:
  // float *dev_B = 0;
  // float *dev_K = 0;
  // uint8_t *dev_X1 = 0;
  // uint8_t *dev_X2 = 0;

  // GPU kernel launch - kernel dimensions:
  dim3 block(32, 32);
  dim3 grid;
  grid.x = (number_of_data_1v1_ts - 1) / block.x + 1;
  grid.y = (number_of_data_1v1 - 1) / block.y + 1;

  hipError_t hipStatus;

  // GPU kernel launch - allocate device variables:
  // hipStatus = hipMalloc((void **)&dev_B, 256 * 256 * sizeof(float));
  // if (hipStatus != hipSuccess) {
  //   fprintf(stderr, "hipMalloc failed!");
  //   goto Error1;
  // }
  // hipStatus =
  //     hipMalloc((void **)&dev_K,
  //                number_of_data_1v1_ts * number_of_data_1v1 * sizeof(float));
  // if (hipStatus != hipSuccess) {
  //   fprintf(stderr, "hipMalloc failed!");
  //   goto Error2;
  // }
  // hipStatus = hipMalloc((void **)&dev_X1,
  //                         number_of_data_1v1_ts * n_features *
  //                         sizeof(uint8_t));
  // if (hipStatus != hipSuccess) {
  //   fprintf(stderr, "hipMalloc failed!");
  //   goto Error3;
  // }
  // hipStatus = hipMalloc((void **)&dev_X2,
  //                         number_of_data_1v1 * n_features * sizeof(uint8_t));
  // if (hipStatus != hipSuccess) {
  //   fprintf(stderr, "hipMalloc failed!");
  //   goto Error4;
  // }

  // GPU kernel launch - copy to device variables:
  // hipStatus =
  //     hipMemcpy(dev_B, B, 256 * 256 * sizeof(float),
  //     hipMemcpyHostToDevice);
  // if (hipStatus != hipSuccess) {
  //   fprintf(stderr, "hipMemcpy failed!");
  //   goto Error4;
  // }
  // for (uint32_t i = 0; i < number_of_data_1v1_ts; i++) {
  //   hipStatus =
  //       hipMemcpy(dev_X1 + i * n_features, test_images_1v1[i],
  //                  n_features * sizeof(uint8_t), hipMemcpyHostToDevice);
  //   if (hipStatus != hipSuccess) {
  //     fprintf(stderr, "hipMemcpy failed!");
  //     goto Error4;
  //   }
  // }
  // for (uint32_t i = 0; i < number_of_data_1v1; i++) {
  //   hipStatus =
  //       hipMemcpy(dev_X2 + i * n_features, train_images_1v1[i],
  //                  n_features * sizeof(uint8_t), hipMemcpyHostToDevice);
  //   if (hipStatus != hipSuccess) {
  //     fprintf(stderr, "hipMemcpy failed!");
  //     goto Error4;
  //   }
  // }

  // GPU kernel launch:
  kernel_kernel_matrix_ts<<<grid, block>>>(
      B, K, dev_X1, dev_X2, number_of_data_1v1_ts, number_of_data_1v1);
  hipStatus = hipGetLastError();
  if (hipStatus != hipSuccess) {
    fprintf(stderr, "kernel_kernel_matrix_ts launch failed: %s\n",
            hipGetErrorString(hipStatus));
    goto Error4;
  }

  hipStatus = hipDeviceSynchronize();
  if (hipStatus != hipSuccess) {
    fprintf(stderr,
            "hipDeviceSynchronize returned error code %d after launching "
            "addKernel!\n",
            hipStatus);
    goto Error4;
  }

// GPU kernel launch - copy from device variables:
// hipStatus = hipMemcpy(K, dev_K, number_of_data_1v1_ts *
// number_of_data_1v1
// *
//                                       sizeof(float),
//                         hipMemcpyDeviceToHost);
// if (hipStatus != hipSuccess) {
//   fprintf(stderr, "hipMemcpy failed!");
//   goto Error4;
// }

// GPU kernel launch - free device variables:
Error4:
  //   hipFree(dev_X2);
  // Error3:
  // //   hipFree(dev_X1);
  // Error2:
  // //   hipFree(dev_K);
  // Error1:
  // // hipFree(dev_B);
  // Error0:
  return hipStatus;
}
