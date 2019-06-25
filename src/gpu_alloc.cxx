#include <gpu_alloc.hpp>

void * cuda_memory_resource::do_allocate(std::size_t bytes, std::size_t alignment) {

  return cuda_malloc_wrapper( bytes );

}


void cuda_memory_resource::do_deallocate(void * p, std::size_t bytes, std::size_t alignment) {

  cuda_free_wrapper( p );

}

bool cuda_memory_resource::do_is_equal( const memory_resource& other ) const noexcept {
  return this == &other;
}
