#pragma once
#include <gpu_alloc_wrapper.hpp>
#include <memory>
#include <vector>
#include <experimental/memory_resource>

class cuda_memory_resource : public std::experimental::pmr::memory_resource {

  void * do_allocate(std::size_t bytes, std::size_t alignment);
  void do_deallocate(void * p, std::size_t bytes, std::size_t alignment);
  bool do_is_equal( const memory_resource& ) const noexcept;

};


template <typename T, typename base = std::experimental::pmr::polymorphic_allocator<T> >
class device_allocator : public base {

  static_assert( std::is_trivial_v<T>, "MUST BE TRIVIAL" );

public:

  device_allocator() = delete;
  device_allocator( cuda_memory_resource* r ) : base( r ) { };
  device_allocator( const device_allocator& )     = default;
  device_allocator( device_allocator&& ) noexcept = default;
  ~device_allocator() noexcept = default;  


  template <typename... Args>
  void construct( T* p, Args&& ...args) { };

  void destroy( T* p ) { };

};

template <typename T>
using device_vector = std::vector< T, device_allocator<T> >;
