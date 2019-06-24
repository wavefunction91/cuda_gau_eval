#include <gpu_driver.hpp>
#include <gau_kernel.hcu>
#include <algorithm>
#include <numeric>
#include <chrono>

// Serialize in host memory
std::vector<char> generate_shell_buffer( const std::vector<libint2::Shell>& shells ) {

  // Get size of the buffer
  size_t nBuff = std::accumulate( shells.begin(), shells.end(), size_t(0), 
    [](const auto& a, const auto& b){ return a + 2*sizeof(int) + 2*sizeof(double)*b.nprim();} );

  std::vector<char> buff(nBuff);

  char* buff_ptr = buff.data();
  for( const auto& shell : shells ) {

    const auto& contraction = shell.contr.at(0);
    const int l = contraction.l;
    const int nprim = shell.nprim();    
    const double* coef = contraction.coeff.data();
    const double* alph = shell.alpha.data();


    std::memcpy(buff_ptr, &l,     sizeof(int)); buff_ptr += sizeof(int);
    std::memcpy(buff_ptr, &nprim, sizeof(int)); buff_ptr += sizeof(int);
    std::memcpy(buff_ptr, coef,   nprim*sizeof(double)); buff_ptr += nprim*sizeof(double);
    std::memcpy(buff_ptr, alph,   nprim*sizeof(double)); buff_ptr += nprim*sizeof(double);

  }
  

  return buff;
}

std::vector<Shell_device> generate_device_shells( const std::vector<libint2::Shell>& shells , char* buff ) { 

  size_t nShells = shells.size();
  std::vector<Shell_device> dev_shells(nShells);

  for( auto i = 0; i < nShells; ++i ) {

    const auto& shell = shells[i];
    const auto& contraction = shell.contr.at(0);
    const int l = contraction.l;
    const int nprim = shell.nprim();    

    char* l_ptr  = buff;
    char* np_ptr = l_ptr  + sizeof(int);
    char* co_ptr = np_ptr + sizeof(int);
    char* al_ptr = co_ptr + nprim*sizeof(double);

    dev_shells[i].l     = l;
    dev_shells[i].nprim = nprim;
    dev_shells[i].coef  = reinterpret_cast<double*>(co_ptr);
    dev_shells[i].alpha = reinterpret_cast<double*>(al_ptr);

    buff += 2*sizeof(int) + 2*nprim*sizeof(double);

  }
  
  return dev_shells;

} 

void print_host(int64_t nShells, const std::vector<libint2::Shell>& shells) {

  printf("Hello from Host\n");

  for(auto i = 0; i < nShells; ++i) {
    const auto& shell = shells[i];
    const auto& contraction = shell.contr.at(0);
    const int l = contraction.l;
    const int nprim = shell.nprim();    
    printf("%d, %d\n",l, nprim);
  }

}

void print_host(int64_t nShells, Shell_device* shells) {

  printf("Hello from host\n");

  for(auto i = 0; i < nShells; ++i)
    printf("%p, %d, %d\n", shells+i, shells[i].l, shells[i].nprim);

}
__global__ void print_kern(int64_t nShells, Shell_device* shells) {

  printf("Hello from device\n");

  for(auto i = 0; i < nShells; ++i)
    printf("%p, %d, %d\n", shells+i, shells[i].l, shells[i].nprim);

}






__global__ void gau_eval_kernel( const int64_t npts, const int64_t nShell, const double* gX, const double *gY, const double *gZ,
                                 Shell_device* shells, double *eval ) {


  const int sh_indx = blockIdx.x * blockDim.x + threadIdx.x;
  const int gp_indx = blockIdx.y * blockDim.y + threadIdx.y;


  //printf("In kernel: {%d,%d,%d, %d}, {%d,%d,%d, %d}\n" ,
  //       blockIdx.x, blockDim.x, threadIdx.x, sh_indx,
  //       blockIdx.y, blockDim.y, threadIdx.y, gp_indx);

  if( sh_indx < nShell and gp_indx < npts ) {

    const Shell_device* shell_ptr = shells + sh_indx;
    const double xc = gX[gp_indx];
    const double yc = gY[gp_indx];
    const double zc = gZ[gp_indx];

    const double rsq = xc*xc + yc*yc + zc*zc;
    
    const int nprim = shell_ptr->nprim;

    double tmp = 0.;
    for( int i = 0; i < nprim; ++i )
      tmp += shell_ptr->coef[i] * std::exp( -shell_ptr->alpha[i]*rsq );

    //printf("In kernel: %d, %d, %d, %.8f\n", sh_indx, gp_indx, nprim, float(tmp));
    //printf("In gpu: %d, %d, %d, {%.8f,%.8f,%.8f}, %.8f, %.8f, %.8f\n", 
    //       sh_indx, gp_indx, nprim, 
    //       gX[gp_indx], gY[gp_indx], gZ[gp_indx],
    //       shell_ptr->alpha[0], shell_ptr->coef[0],
    //       float(tmp));
    eval[ gp_indx + sh_indx*npts ] = tmp;

  }

}









void wakeup_gpu() {

  cudaDeviceProp prop;
  cudaGetDeviceProperties( &prop, 0 );

  double* tmp;
  auto err_t = cudaMalloc( &tmp, 1 );
  cudaFree( tmp );
}

void gpu_driver(  int64_t npts,  const double* gX,  const double* gY,  const double *gZ, 
                 const std::vector<libint2::Shell>& shells, double *eval, Timer& timer ) {


  double *g_d = nullptr;
  double *gX_d = nullptr, *gY_d = nullptr, *gZ_d = nullptr, *eval_d = nullptr;

  timer.time( "cuda malloc",[&]() {
    //cudaMalloc( &gX_d,   npts * sizeof(double) );
    //cudaMalloc( &gY_d,   npts * sizeof(double) );
    //cudaMalloc( &gZ_d,   npts * sizeof(double) );
    //cudaMalloc( &eval_d, npts * sizeof(double) );
    cudaMalloc( &g_d, (3*npts + shells.size()*npts)*sizeof(double) );
    gX_d = g_d;
    gY_d = gX_d + npts;
    gZ_d = gY_d + npts;
    eval_d = gX_d + npts;
  } );


  timer.time( "send grid host -> device",[&]() {
    cudaMemcpy( gX_d, gX, npts * sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy( gY_d, gY, npts * sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy( gZ_d, gZ, npts * sizeof(double), cudaMemcpyHostToDevice );
  });


  // Serialize and transfer shells
  std::vector<char> shell_buff;
  timer.time( "generate shell buffer (host)",[&]() {
    shell_buff = generate_shell_buffer( shells );
  });

  char* shell_buff_d = nullptr;
  timer.time( "send shell buffer host -> device",[&]() {
    cudaMalloc( &shell_buff_d, shell_buff.size() );
    cudaMemcpy( shell_buff_d, shell_buff.data(), shell_buff.size(), cudaMemcpyHostToDevice );
  });

  // Get device data ptr array on host and send to device
  std::vector<Shell_device> dev_shells;
  timer.time( "generate device shells (host)",[&]() {
    dev_shells = generate_device_shells( shells, shell_buff_d );
  });

  Shell_device* dev_shells_d;
  timer.time( "send device shells host -> device",[&]() {
    cudaMalloc( &dev_shells_d, dev_shells.size() * sizeof(Shell_device) );
    cudaMemcpy( dev_shells_d, dev_shells.data(), dev_shells.size() * sizeof(Shell_device), cudaMemcpyHostToDevice );
  });


  //std::cout << (shells.size()+1)/16 << ", " << (npts+1)/16 << std::endl;
  //dim3 block_sz(16,16,1);
  //dim3 grid_sz( (shells.size()+1)/16, (npts+1)/16,1);


  dim3 block_sz, grid_sz;
  block_sz.x = 16;
  block_sz.y = 16;
  grid_sz.x = std::max(1u,int(shells.size()+1)/block_sz.x);
  grid_sz.y = std::max(1u,int(npts+1)/block_sz.y);

  //std::cout << "grid: " << grid_sz.x << ", " << grid_sz.y << std::endl;
  //std::cout << "block: " << block_sz.x << ", " << block_sz.y << std::endl;
  timer.time( "gpu kernel eval (device)",[&]() {
    gau_eval_kernel<<<grid_sz, block_sz>>>( npts, shells.size(), gX_d, gY_d, gZ_d, dev_shells_d, eval_d );
  });

  // call kernel
  //print_host(shells.size(), shells);
  //print_host(shells.size(), dev_shells.data());
  //cudaDeviceSynchronize();
  //print_kern<<<1,1>>>(shells.size(), dev_shells_d);

  cudaDeviceSynchronize();
  timer.time( "recv basis eval device -> host",[&]() {
    cudaMemcpy( eval, eval_d, shells.size()*npts * sizeof(double), cudaMemcpyDeviceToHost );
  });

  cudaFree( g_d );
  cudaFree( shell_buff_d );
  cudaFree( dev_shells_d );

  cudaDeviceSynchronize();

}
