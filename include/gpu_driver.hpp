#include <libint2/shell.h>
#include <timer.hpp>

struct Shell_device {

  int l;
  int nprim;

  double* coef;
  double* alpha;

};

void wakeup_gpu();

std::vector<char>         generate_shell_buffer( const std::vector<libint2::Shell>& shells );
std::vector<Shell_device> generate_device_shells( const std::vector<libint2::Shell>& shells , char* ); 


void gpu_driver(  int64_t npts,  const double* gX,  const double* gY,  const double *gZ, 
                 const std::vector<libint2::Shell>& shells, double *eval, Timer& timer );

