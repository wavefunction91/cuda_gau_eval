#include <gpu_alloc_wrapper.hpp>

void* cuda_malloc_wrapper(size_t n) {

  void * ptr = nullptr;
  cudaMalloc( &ptr, n );
  return ptr;

}

void cuda_free_wrapper( void* ptr ) {
  cudaFree( ptr );
}
