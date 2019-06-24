#include <iostream>
#include <vector>
#include <array>
#include <chrono>
#include <random>
#include <algorithm>

#include <libint2/shell.h>
#include <gau2grid_driver.hpp>
#include <gpu_driver.hpp>
#include <timer.hpp>

using cart_t = std::array<double,3>;

int main() {

  wakeup_gpu();

  // Random
  std::default_random_engine gen;
  std::uniform_real_distribution<> dist_real(0.,.5);
  std::uniform_int_distribution<>  dist_int(1,8);

  auto rrand_gen = [&](){ return dist_real(gen); };
  auto irand_gen = [&](){ return dist_int(gen); };

  // Construct shells
  //const size_t nShells = 1000;
  const size_t nShells = 17;
  std::vector< libint2::Shell > shells;
  shells.reserve(nShells);

  for( auto i = 0ul; i < nShells; ++i ) {
    //const cart_t cen = { rrand_gen(), rrand_gen(), rrand_gen() };
    const cart_t cen = { 0., 0., 0. };
    const int    l     = 0;
    //const int    nprim = irand_gen();
    const int    nprim = 1;

    std::vector<double> coef(nprim);
    std::vector<double> alpha(nprim);

    std::generate( coef.begin(),  coef.end(),  rrand_gen );
    std::generate( alpha.begin(), alpha.end(), rrand_gen );

    shells.push_back(
      libint2::Shell{ alpha, {{l, false, coef}}, cen }
    );
  }

  const size_t nBF = std::accumulate( shells.begin(), shells.end(), size_t(0),
                       [](const auto& a, const auto& b){ return a + b.size(); } );


  // Generate grid
  //const size_t nGrid = 5000;
  const size_t nGrid = 17;
  std::vector<double> gX(nGrid), gY(nGrid), gZ(nGrid);

  std::generate( gX.begin(), gX.end(), rrand_gen );
  std::generate( gY.begin(), gY.end(), rrand_gen );
  std::generate( gZ.begin(), gZ.end(), rrand_gen );

  Timer timer;

  // Gau2Grid
  std::vector<double> phi_g2g( nBF * nGrid );

  timer.time( std::string("g2g execution"), [&]() {
    gau2grid_driver( nGrid, gX.data(), gY.data(), gZ.data(), shells, phi_g2g.data(), timer );
  } );

  //std::chrono::duration<double> g2g_dur = g2g_en - g2g_st;
  //std::cout << "G2G " << g2g_dur.count() << std::endl;

  std::vector<double> phi_gpu( nBF * nGrid );
  timer.time( std::string("gpu execution"), [&]() {
    gpu_driver( nGrid, gX.data(), gY.data(), gZ.data(), shells, phi_gpu.data(), timer );
  });

  std::cout << timer << std::endl;

  std::vector<double> diff( nBF * nGrid ); 
  for(auto i = 0; i < diff.size(); ++i) diff[i] = std::abs( phi_g2g[i] - phi_gpu[i] );


//  for(auto i = 0; i < diff.size(); ++i)
//    std::cout << phi_g2g[i] << ", " << phi_gpu[i] << ", " << diff[i] << std::endl;

  std::cout << "MAX DIFF " << *std::max_element( diff.begin(), diff.end() ) << std::endl;
}
