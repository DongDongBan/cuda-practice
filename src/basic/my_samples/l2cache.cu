#include <iostream>  
#include "../tutor_book/error.cuh"

__global__ void kernel(int *data_persistent, int *data_streaming, int dataSize, int freqSize) { 
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    
    /*Each CUDA thread accesses one element in the persistent data section
      and one element in the streaming data section.
      Because the size of the persistent memory region (freqSize * sizeof(int) bytes) is much 
      smaller than the size of the streaming memory region (dataSize * sizeof(int) bytes), data 
      in the persistent region is accessed more frequently*/

    if (freqSize > 0)
      data_persistent[tid % freqSize] = 2 * data_persistent[tid % freqSize]; 
    data_streaming[tid % dataSize] = 2 * data_streaming[tid % dataSize];
}

int main(int argc, char const *argv[])
{
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf("cudaGetDeviceCount returned %d\n-> %s\n",
           static_cast<int>(error_id), cudaGetErrorString(error_id));
    printf("Result = FAIL\n");
    exit(EXIT_FAILURE);
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    printf("There are no available device(s) that support CUDA\n");
  } else {
    printf("Detected %d CUDA Capable device(s)\n", deviceCount);
  }


  


  for (size_t device_id = 0; device_id < deviceCount; device_id++)
  {
    CHECK(cudaSetDevice(device_id));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, device_id));    
    CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, prop.persistingL2CacheMaxSize)); /* Set aside max possible size of L2 cache for persisting accesses */ 
    // std::cout << device_id << ":persistingL2CacheMaxSize " << prop.persistingL2CacheMaxSize << std::endl;

    int *p_global, *p_streaming;
    cudaMalloc((void**)&p_global, 1024 * 1024 * 1024 * sizeof(int));
    // Warm Up
    for (size_t i = 0; i < 10; i++)
      kernel<<<1024 * 1024, 1024>>>(p_global, p_global, 1024 * 1024 * 1024, 0);
    CHECK(cudaGetLastError());

    for (size_t freqSize = 0; freqSize < prop.persistingL2CacheMaxSize * 2; freqSize+=1024*1024)
    {
      cudaStream_t stream;
      CHECK(cudaStreamCreate(&stream));        
      cudaStreamAttrValue stream_attribute;                                         // Stream level attributes data structure
      stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(p_global); 
      stream_attribute.accessPolicyWindow.num_bytes = freqSize * sizeof(int);   //Number of bytes for persisting accesses in range 10-60 MB
      stream_attribute.accessPolicyWindow.hitRatio  = 1.0;                      //Hint for cache hit ratio. Fixed value 1.0    
      stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
      stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // Type of access property on cache miss.  
      
      // //Set the attributes to a CUDA stream of type cudaStream_t
      CHECK(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute));        

      p_streaming = p_global + freqSize;
      cudaEvent_t start, stop;
      float time;

      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      cudaEventRecord( start, 0 );      
      kernel<<<1024 * 1024, 1024, 0, stream>>>(p_global, p_streaming, 1024 * 1024 * 1024 - freqSize, freqSize);
      cudaEventRecord( stop, 0 );
      cudaEventSynchronize( stop );

      cudaEventElapsedTime( &time, start, stop ); 
      std::cout << "Device" << device_id << ": " << freqSize * sizeof(int) << '/' << prop.persistingL2CacheMaxSize << " Used " << time << "ms\n";
      cudaEventDestroy( start );
      cudaEventDestroy( stop );

      CHECK(cudaGetLastError());
      
      stream_attribute.accessPolicyWindow.num_bytes = 0;                                          // Setting the window size to 0 disable it
      CHECK(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute));   // Overwrite the access policy attribute to a CUDA Stream
      CHECK(cudaCtxResetPersistingL2Cache());                                                            // Remove any persistent lines in L2    

      CHECK(cudaStreamDestroy(stream));  
    }

    error_id = cudaGetLastError();
    if (error_id != cudaSuccess) {
      printf("HitRatio==1.0 returned %d\n-> %s\n",
            static_cast<int>(error_id), cudaGetErrorString(error_id));
      printf("Result = FAIL\n");
      exit(EXIT_FAILURE);
    }    

    for (size_t freqSize = 0; freqSize < prop.persistingL2CacheMaxSize * 2; freqSize+=1024*1024)
    {
      cudaStream_t stream;
      CHECK(cudaStreamCreate(&stream));          
      cudaStreamAttrValue stream_attribute;                                         // Stream level attributes data structure
      stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(p_global); 
      stream_attribute.accessPolicyWindow.num_bytes = freqSize * sizeof(int);   //Number of bytes for persisting accesses in range 10-60 MB
      if (freqSize * sizeof(int) <= prop.persistingL2CacheMaxSize)
        stream_attribute.accessPolicyWindow.hitRatio  = 1.0f;
      else
        stream_attribute.accessPolicyWindow.hitRatio  = (float)prop.persistingL2CacheMaxSize / (freqSize * sizeof(int)); //Hint for cache hit ratio. Fixed value 1.0    
      stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
      stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // Type of access property on cache miss. 

      // //Set the attributes to a CUDA stream of type cudaStream_t
      CHECK(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute));        

      p_streaming = p_global + freqSize;
      cudaEvent_t start, stop;
      float time;

      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      cudaEventRecord( start, 0 );        
      kernel<<<1024 * 1024, 1024, 0, stream>>>(p_global, p_streaming, 1024 * 1024 * 1024 - freqSize, freqSize);
      cudaEventRecord( stop, 0 );
      cudaEventSynchronize( stop );

      cudaEventElapsedTime( &time, start, stop ); 
      std::cout << "Device" << device_id << ": " << freqSize * sizeof(int) << '/' << prop.persistingL2CacheMaxSize << "Flexible Used " << time << "ms\n";
      cudaEventDestroy( start );
      cudaEventDestroy( stop );

      CHECK(cudaGetLastError());

      stream_attribute.accessPolicyWindow.num_bytes = 0;                                          // Setting the window size to 0 disable it
      CHECK(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute));   // Overwrite the access policy attribute to a CUDA Stream
      CHECK(cudaCtxResetPersistingL2Cache());                                                            // Remove any persistent lines in L2    

      CHECK(cudaStreamDestroy(stream));  
    }    

    error_id = cudaGetLastError();
    if (error_id != cudaSuccess) {
      printf("Flexible HitRatio returned %d\n-> %s\n",
            static_cast<int>(error_id), cudaGetErrorString(error_id));
      printf("Result = FAIL\n");
      exit(EXIT_FAILURE);
    }    

    cudaFree(p_global);
  }
  


  return 0;
}
