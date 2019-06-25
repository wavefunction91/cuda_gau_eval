#pragma once
#include <cstddef>
void* cuda_malloc_wrapper(size_t n);
void cuda_free_wrapper(void*);
