#include <libint2/shell.h>
#include <timer.hpp>

void gau2grid_driver( const int64_t npts, const double* gX, const double* gY, const double *gZ, 
                      const std::vector<libint2::Shell>& shells, double *eval, Timer& timer);
