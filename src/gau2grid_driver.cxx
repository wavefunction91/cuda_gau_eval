#include <gau2grid_driver.hpp>
#include <gau2grid/gau2grid.h>

void gau2grid_driver( const int64_t npts, const double* gX, const double* gY, const double *gZ, 
                      const std::vector<libint2::Shell>& shells, double *eval, Timer& timer ) {


  int64_t nComponents = 0;
  for( int64_t iShell = 0; iShell < shells.size(); ++iShell ) {

    const auto& shell = shells[iShell];

    for( int64_t j = 0; j < shell.ncontr(); ++j ) {
    
      const auto& contraction = shell.contr.at(j);
      
      gg_collocation(contraction.l,
                     npts, gX, gY, gZ,
                     shell.nprim(), &contraction.coeff.at(0), &shell.alpha.at(0),
                     shell.O.data(), contraction.pure,
                     eval + nComponents*npts);
      
    
      //auto eval_loc = eval +nComponents*npts;
      //int sh_indx = iShell;
      //for( int gp_indx = 0; gp_indx < npts; gp_indx++ )
      //  printf("In g2g: %d, %d, %d, {%.8f,%.8f,%.8f}, %.8f, %.8f, %.8f\n", 
      //         sh_indx, gp_indx, shell.nprim(), 
      //         gX[gp_indx], gY[gp_indx], gZ[gp_indx],
      //         shell.alpha[0], contraction.coeff[0],
      //         float(eval_loc[gp_indx]));
      nComponents += contraction.size();

    }

  }

}
