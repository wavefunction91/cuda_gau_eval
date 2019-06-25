cmake_minimum_required( VERSION 3.14 FATAL_ERROR )
project( GauEval CXX C CUDA )

find_package( LibInt2  REQUIRED )
find_package( gau2grid REQUIRED )

find_package( CUDA REQUIRED )

set( GE_CXX_SRCS  src/gau2grid_driver.cxx src/gpu_alloc.cxx )
set( GE_CUDA_SRCS src/gpu_driver.cu src/gpu_alloc_wrapper.cu )

add_library( gau_eval_cxx  ${GE_CXX_SRCS}  )
add_library( gau_eval_cuda ${GE_CUDA_SRCS} )

target_link_libraries     ( gau_eval_cuda PUBLIC Libint2::cxx )
target_include_directories( gau_eval_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include )
target_compile_features   ( gau_eval_cuda PRIVATE cxx_std_14 )

target_link_libraries     ( gau_eval_cxx PUBLIC Libint2::cxx gau2grid::gg gau_eval_cuda )
target_include_directories( gau_eval_cxx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include )
target_compile_features   ( gau_eval_cxx PRIVATE cxx_std_17 )

add_executable         ( gau_eval src/main.cxx )
target_link_libraries  ( gau_eval PUBLIC gau_eval_cxx gau_eval_cuda )
target_compile_features( gau_eval PRIVATE cxx_std_17 )
