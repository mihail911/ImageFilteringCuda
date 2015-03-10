#include "ImageCleaner.h"
#include <math.h>

#ifndef SIZEX
#error Please define SIZEX.
#endif
#ifndef SIZEY
#error Please define SIZEY.
#endif
#define PI 3.14159265

//----------------------------------------------------------------
// TODO:  CREATE NEW KERNELS HERE.  YOU CAN PLACE YOUR CALLS TO
//        THEM IN THE INDICATED SECTION INSIDE THE 'filterImage'
//        FUNCTION.
//
// BEGIN ADD KERNEL DEFINITIONS
//----------------------------------------------------------------


__global__ void fftx(float *device_real, float *device_imag, int size_x, int size_y)
{
  //draw into shared memory from global
  //__shared__ float realOutBuffer[SIZEX];
  //__shared__ float imagOutBuffer[SIZEX];
  float realOutVal;
  float imagOutVal;
  float threadDeviceReal[size_x * size_y];
  float threadDeviceImag[size_x * size_y];

  __shared__ float fft_real[SIZEY];
  __shared__ float fft_imag[SIZEY];

  for (int n = 0; n < size_y; n++) {
    float term = -2 * PI * threadIdx.x * n / size_y;
    fft_real[n] = cos(term);
    fft_imag[n] = sin(term);
  }

  //realOutBuffer[threadIdx.x] = 0.0f; //we can make this local to each thread--no need for shared!
  //imagOutBuffer[threadIdx.x] = 0.0f;

  realOutVal = 0.0f;
  imagOutVal = 0.0f;
  for (int n = 0; n < size_y; n++) {
    realOutVal += (device_real[blockIdx.x*size_y + n] * fft_real[n]) - (device_imag[blockIdx.x*size_y + n] * fft_imag[n]);
    imagOutVal += (device_imag[blockIdx.x*size_y + n] * fft_real[n]) + (device_real[blockIdx.x*size_y + n] * fft_imag[n]);
  }

  //__syncthreads(); //we don't need to sync because writing to different spots
  // threadDeviceReal[blockIdx.x*size_y + threadIdx.x] = realOutVal;
  // threadDeviceImag[blockIdx.x*size_y + threadIdx.x] = imagOutVal;

  device_real[blockIdx.x*size_y + threadIdx.x] = realOutVal;;
  device_imag[blockIdx.x*size_y + threadIdx.x] = imagOutVal;
}

__global__ void ifftx(float *device_real, float *device_imag, int size_x, int size_y)
{
  //__shared__ float realOutBuffer[SIZEX];
  //__shared__ float imagOutBuffer[SIZEX];
  float realOutVal;
  float imagOutVal;
  // float threadDeviceReal[size_x * size_y];
  // float threadDeviceImag[size_x * size_y];
  __shared__ float fft_real[SIZEY];
  __shared__ float fft_imag[SIZEY];

  for (int n = 0; n < size_y; n++) {
    float term = 2 * PI * threadIdx.x * n / size_y;
    fft_real[n] = cos(term);
    fft_imag[n] = sin(term);
  }

  realOutVal = 0.0f;
  imagOutVal = 0.0f;
  for (int n = 0; n < size_y; n++) {
    realOutVal += (device_real[blockIdx.x*size_y + n] * fft_real[n]) - (device_imag[blockIdx.x*size_y + n] * fft_imag[n]);
    imagOutVal += (device_imag[blockIdx.x*size_y + n] * fft_real[n]) + (device_real[blockIdx.x*size_y + n] * fft_imag[n]);
  }
  
  realOutVal /= size_y;
  imagOutVal /= size_y;

  //__syncthreads();

  // threadDeviceReal[blockIdx.x*size_y + threadIdx.x] = realOutVal;
  // threadDeviceImag[blockIdx.x*size_y + threadIdx.x] = imagOutVal;

  device_real[blockIdx.x*size_y + threadIdx.x] = realOutVal;
  device_imag[blockIdx.x*size_y + threadIdx.x] = imagOutVal;
}

__global__ void ffty(float *device_real, float *device_imag, int size_x, int size_y)
{
  //__shared__ float realOutBuffer[SIZEY];
  //__shared__ float imagOutBuffer[SIZEY];
  float realOutVal;
  float imagOutVal;
  __shared__ float fft_real[SIZEX];
  __shared__ float fft_imag[SIZEX];

  for (int n = 0; n < size_x; n++) {
    float term = -2 * PI * threadIdx.x * n / size_x;
    fft_real[n] = cos(term);
    fft_imag[n] = sin(term);
  }

  realOutVal = 0.0f;
  imagOutVal = 0.0f;
  for (int n = 0; n < size_x; n++) {
    realOutVal += (device_real[n*size_x + blockIdx.x] * fft_real[n]) - (device_imag[n*size_x + blockIdx.x] * fft_imag[n]);
    imagOutVal += (device_imag[n*size_x + blockIdx.x] * fft_real[n]) + (device_real[n*size_x + blockIdx.x] * fft_imag[n]);
  }

  //__syncthreads();
  device_real[threadIdx.x*size_x + blockIdx.x] = realOutVal;
  device_imag[threadIdx.x*size_x + blockIdx.x] = imagOutVal;
}

__global__ void iffty(float *device_real, float *device_imag, int size_x, int size_y)
{
  //__shared__ float realOutBuffer[SIZEY];
  //__shared__ float imagOutBuffer[SIZEY];
  float realOutVal;
  float imagOutVal;
  __shared__ float fft_real[SIZEX];
  __shared__ float fft_imag[SIZEX];

  for (int n = 0; n < size_x; n++) {
    float term = 2 * PI * threadIdx.x * n / size_x;
    fft_real[n] = cos(term);
    fft_imag[n] = sin(term);
  }

  realOutVal = 0.0f;
  imagOutVal = 0.0f;
  for (int n = 0; n < size_x; n++) {
    realOutVal += (device_real[n*size_x + blockIdx.x] * fft_real[n]) - (device_imag[n*size_x + blockIdx.x] * fft_imag[n]);
    imagOutVal += (device_imag[n*size_x + blockIdx.x] * fft_real[n]) + (device_real[n*size_x + blockIdx.x] * fft_imag[n]);
  }

  realOutVal /= size_x;
  imagOutVal /= size_x;

  //__syncthreads();
  device_real[threadIdx.x*size_x + blockIdx.x] = realOutVal;
  device_imag[threadIdx.x*size_x + blockIdx.x] = imagOutVal;
}

__global__ void filter(float *device_real, float *device_imag, int size_x, int size_y)
{
  int eightX = size_x/8;
  int eight7X = size_x - eightX;
  int eightY = size_y/8;
  int eight7Y = size_y - eightY;
  if(!(threadIdx.x < eightX && blockIdx.x < eightY) &&
     !(threadIdx.x < eightX && blockIdx.x >= eight7Y) &&
     !(threadIdx.x >= eight7X && blockIdx.x < eightY) &&
     !(threadIdx.x >= eight7X && blockIdx.x >= eight7Y))
  {
    // Zero out these values
    device_real[threadIdx.x*size_y + blockIdx.x] = 0;
    device_imag[threadIdx.x*size_y + blockIdx.x] = 0;
  }
}



//----------------------------------------------------------------
// END ADD KERNEL DEFINTIONS
//----------------------------------------------------------------

__host__ float filterImage(float *real_image, float *imag_image, int size_x, int size_y)
{
  // check that the sizes match up
  assert(size_x == SIZEX);
  assert(size_y == SIZEY);

  int matSize = size_x * size_y * sizeof(float);

  // These variables are for timing purposes
  float transferDown = 0, transferUp = 0, execution = 0;
  cudaEvent_t start,stop;

  CUDA_ERROR_CHECK(cudaEventCreate(&start));
  CUDA_ERROR_CHECK(cudaEventCreate(&stop));

  // Create a stream and initialize it
  cudaStream_t filterStream;
  CUDA_ERROR_CHECK(cudaStreamCreate(&filterStream));

  // Alloc space on the device
  float *device_real, *device_imag;
  CUDA_ERROR_CHECK(cudaMalloc((void**)&device_real, matSize));
  CUDA_ERROR_CHECK(cudaMalloc((void**)&device_imag, matSize));

  // Start timing for transfer down
  CUDA_ERROR_CHECK(cudaEventRecord(start,filterStream));
  
  // Here is where we copy matrices down to the device 
  CUDA_ERROR_CHECK(cudaMemcpy(device_real,real_image,matSize,cudaMemcpyHostToDevice));
  CUDA_ERROR_CHECK(cudaMemcpy(device_imag,imag_image,matSize,cudaMemcpyHostToDevice));
  
  // Stop timing for transfer down
  CUDA_ERROR_CHECK(cudaEventRecord(stop,filterStream));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&transferDown,start,stop));

  // Start timing for the execution
  CUDA_ERROR_CHECK(cudaEventRecord(start,filterStream));

  //----------------------------------------------------------------
  // TODO: YOU SHOULD PLACE ALL YOUR KERNEL EXECUTIONS
  //        HERE BETWEEN THE CALLS FOR STARTING AND
  //        FINISHING TIMING FOR THE EXECUTION PHASE
  // BEGIN ADD KERNEL CALLS
  //----------------------------------------------------------------

  // This is an example kernel call, you should feel free to create
  // as many kernel calls as you feel are needed for your program
  // Each of the parameters are as follows:
  //    1. Number of thread blocks, can be either int or dim3 (see CUDA manual)
  //    2. Number of threads per thread block, can be either int or dim3 (see CUDA manual)
  //    3. Always should be '0' unless you read the CUDA manual and learn about dynamically allocating shared memory
  //    4. Stream to execute kernel on, should always be 'filterStream'
  //
  // Also note that you pass the pointers to the device memory to the kernel call
  fftx <<<size_x,size_y,0,filterStream>>> (device_real,device_imag,size_x,size_y);
  ffty <<<size_x,size_y,0,filterStream>>> (device_real,device_imag,size_x,size_y);
  filter <<<size_x,size_y,0,filterStream>>> (device_real,device_imag,size_x,size_y);
  ifftx <<<size_x,size_y,0,filterStream>>> (device_real,device_imag,size_x,size_y);
  iffty <<<size_x,size_y,0,filterStream>>> (device_real,device_imag,size_x,size_y);


  //---------------------------------------------------------------- 
  // END ADD KERNEL CALLS
  //----------------------------------------------------------------

  // Finish timimg for the execution 
  CUDA_ERROR_CHECK(cudaEventRecord(stop,filterStream));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&execution,start,stop));

  // Start timing for the transfer up
  CUDA_ERROR_CHECK(cudaEventRecord(start,filterStream));

  // Here is where we copy matrices back from the device 
  CUDA_ERROR_CHECK(cudaMemcpy(real_image,device_real,matSize,cudaMemcpyDeviceToHost));
  CUDA_ERROR_CHECK(cudaMemcpy(imag_image,device_imag,matSize,cudaMemcpyDeviceToHost));

  // Finish timing for transfer up
  CUDA_ERROR_CHECK(cudaEventRecord(stop,filterStream));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&transferUp,start,stop));

  // Synchronize the stream
  CUDA_ERROR_CHECK(cudaStreamSynchronize(filterStream));
  // Destroy the stream
  CUDA_ERROR_CHECK(cudaStreamDestroy(filterStream));
  // Destroy the events
  CUDA_ERROR_CHECK(cudaEventDestroy(start));
  CUDA_ERROR_CHECK(cudaEventDestroy(stop));

  // Free the memory
  CUDA_ERROR_CHECK(cudaFree(device_real));
  CUDA_ERROR_CHECK(cudaFree(device_imag));

  // Dump some usage statistics
  printf("CUDA IMPLEMENTATION STATISTICS:\n");
  printf("  Host to Device Transfer Time: %f ms\n", transferDown);
  printf("  Kernel(s) Execution Time: %f ms\n", execution);
  printf("  Device to Host Transfer Time: %f ms\n", transferUp);
  float totalTime = transferDown + execution + transferUp;
  printf("  Total CUDA Execution Time: %f ms\n\n", totalTime);
  // Return the total time to transfer and execute
  return totalTime;
}

