#include "hip/hip_runtime.h"
#include "ArffImporter.h"
#include "Helper.h"

#include <hipblas/hipblas.h>

#define MAX_ITER 1000
#define LEARNING_RATE 10.0f

Node initNode(unsigned int numFeatures) {
  Node node;
  node.numFeatures = numFeatures;
  //   node.weights = (float *)malloc(numFeatures * sizeof(float));
  hipMallocManaged(&node.weights, numFeatures * sizeof(float));
  memset(node.weights, 0, numFeatures * sizeof(float));

  return node;
}

__global__ void ComputeCost(float *__restrict__ dCostArr,
                            const unsigned short *__restrict__ dClassArr,
                            const unsigned int numInstances) {
  unsigned int instanceId = blockIdx.x * blockDim.x + threadIdx.x;
  if (instanceId >= numInstances)
    return;

  float cost = dCostArr[instanceId];
  cost = 1.0f / (1.0f + expf(-cost)) - (float)dClassArr[instanceId];
  dCostArr[instanceId] = cost;
}

inline void hipErrorCheck(hipError_t hipStatus) {
  if (hipStatus != hipSuccess)
    printf("kernel launch failed with error \"%s\".\n",
           hipGetErrorString(hipStatus));
}

inline void cublasErrorCheck(hipblasStatus_t hipblasStatus) {
  if (hipblasStatus != HIPBLAS_STATUS_SUCCESS) {
    printf("CuBLAS launch failed with error\n");
    switch (hipblasStatus) {
    case HIPBLAS_STATUS_NOT_INITIALIZED:
      printf("HIPBLAS_STATUS_NOT_INITIALIZED\n");

    case HIPBLAS_STATUS_ALLOC_FAILED:
      printf("HIPBLAS_STATUS_ALLOC_FAILED\n");

    case HIPBLAS_STATUS_INVALID_VALUE:
      printf("HIPBLAS_STATUS_INVALID_VALUE\n");

    case HIPBLAS_STATUS_ARCH_MISMATCH:
      printf("HIPBLAS_STATUS_ARCH_MISMATCH\n");

    case HIPBLAS_STATUS_MAPPING_ERROR:
      printf("HIPBLAS_STATUS_MAPPING_ERROR\n");

    case HIPBLAS_STATUS_EXECUTION_FAILED:
      printf("HIPBLAS_STATUS_EXECUTION_FAILED\n");

    case HIPBLAS_STATUS_INTERNAL_ERROR:
      printf("HIPBLAS_STATUS_INTERNAL_ERROR\n");

    case HIPBLAS_STATUS_NOT_SUPPORTED:
      printf("HIPBLAS_STATUS_NOT_SUPPORTED\n");

    case HIPBLAS_STATUS_INVALID_ENUM:
      printf("HIPBLAS_STATUS_INVALID_ENUM\n");

    case HIPBLAS_STATUS_UNKNOWN:
      printf("HIPBLAS_STATUS_UNKNOWN\n");

    case HIPBLAS_STATUS_HANDLE_IS_NULLPTR:
      printf("HIPBLAS_STATUS_HANDLE_IS_NULLPTR\n");

    case HIPBLAS_STATUS_SUCCESS:
      printf("HIPBLAS_STATUS_SUCCESS\n"); // Not possible, but throws a warning if it's not here
    }
  }
}

int main() {
  ArffImporter trainSetImporter;
  trainSetImporter.Read("Dataset/train/train-first1000.arff");

  // ArffImporter testSetImporter;
  // testSetImporter.Read( "Dataset/test/dev-first1000.arff" );

  // Init host data
  float *featureMatTrans = trainSetImporter.GetFeatureMatTrans();
  unsigned short *classArr = trainSetImporter.GetClassIndex();
  unsigned int numInstances = trainSetImporter.GetNumInstances();
  unsigned int numFeatures = trainSetImporter.GetNumFeatures();
  Node node = initNode(numFeatures);

  // Init device data
  float *dCostArr = nullptr;
  float *dWeightArr = node.weights;
  float *dFeaCostProdArr = nullptr;

  hipErrorCheck(hipMallocManaged(&dCostArr, numInstances * sizeof(float)));

  hipErrorCheck(
      hipMallocManaged(&dFeaCostProdArr, numFeatures * sizeof(float)));

  /* Determine block and grid size of ComputeCost kernel */
  dim3 ccBlockDim;
  dim3 ccGridDim;
  if (numInstances > 128) {
    ccBlockDim.x = 128;
    ccGridDim.x = (numInstances + 127) / 128;
  } else
    ccBlockDim.x = numInstances;

  // Init CuBLAS
  hipblasHandle_t cublasHandle;
  cublasErrorCheck(hipblasCreate(&cublasHandle));

  // Gradient descent params
  float updateWParam = -LEARNING_RATE / (float)numInstances;
  unsigned int iter = 0;

  time_t start, end;
  float dif;
  time(&start);

  printf("\nStart gradient descent...\n");

  float default_alpha = 1.0f;
  float default_beta = 0.0f;
  // Gradient descent
  while (iter++ < MAX_ITER) {
    // Classify
    cublasErrorCheck(hipblasSgemv(cublasHandle, HIPBLAS_OP_N, numInstances,
                                 numFeatures, &default_alpha, featureMatTrans,
                                 numInstances, dWeightArr, 1, &default_beta,
                                 dCostArr, 1));
    ComputeCost<<<ccGridDim, ccBlockDim>>>(dCostArr, classArr, numInstances);
    hipErrorCheck(hipGetLastError());
    // Cost vec dot FeaMat-Transpose
    cublasErrorCheck(hipblasSgemv(cublasHandle, HIPBLAS_OP_T, numInstances,
                                 numFeatures, &default_alpha, featureMatTrans,
                                 numInstances, dCostArr, 1, &default_beta,
                                 dFeaCostProdArr, 1));
    // Update weights
    cublasErrorCheck(hipblasSaxpy(cublasHandle, numFeatures, &updateWParam,
                                 dFeaCostProdArr, 1, dWeightArr, 1));
  }
  hipErrorCheck(hipDeviceSynchronize());

  cublasErrorCheck(hipblasDestroy(cublasHandle));
  // hipErrorCheck(hipMemcpy(node.weights, dWeightArr,
  //                           numFeatures * sizeof(float),
  //                           hipMemcpyDeviceToHost));

  time(&end);
  dif = difftime(end, start);
  printf("Time taken is %.2lf seconds.\n", dif);

  printf("Updating weights completed, weight: %f\n", node.weights[0]);

  //   hipFree(dFeatureMatTrans);
  //   hipFree(dClassArr);
  //   hipFree(dWeightArr);
  hipFree(dCostArr);
  hipFree(dFeaCostProdArr);
  hipFree(node.weights);

  return 0;
}
