cmake_minimum_required( VERSION 3.14 FATAL_ERROR )
project( GauEval CXX C CUDA )

find_package( LibInt2  REQUIRED )
find_package( gau2grid REQUIRED )


add_executable( gau_eval
  src/gau2grid_driver.cxx
  src/gpu_driver.cu
  src/main.cxx
)

target_link_libraries( gau_eval PUBLIC Libint2::cxx gau2grid::gg )
target_include_directories( gau_eval PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include )
target_compile_features( gau_eval PUBLIC cxx_std_14 )
