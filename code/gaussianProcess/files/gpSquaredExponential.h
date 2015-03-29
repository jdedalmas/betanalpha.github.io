#ifndef _GPSQUAREDEXPONENTIAL_

#include "gpKernel.h"

///
/// \author Michael Betancourt,
/// Massachusetts Institute of Technology
///
/// Implementation of a squared exponential covariance function,
///
/*!
 \f[
 C(x_{i}, x_{j}) = \theta_{1}^{2} \exp \left(
 \sum_{k = 1}^{n} \rho_{k}^{2} | \vec{x}_{i}
 - \vec{x}_{j} |_{k}^{2} \right)
 + \theta_{2}^{2} + \theta_{3}^{2} \delta_{ij}
 \f]
*/

class gpSquaredExponential: public gpKernel
{

    public:
        gpSquaredExponential(int dim);
        ~gpSquaredExponential();

        double kernel(const double *x1, const double *x2, bool same);
        double dKernel(const double *x1, const double *x2, bool same, const int n);
        
        void displayHyperParameters(const char* prefix = "");
        TString displayRelevance() { return "r_{i} = rho_{i}^{2}"; }
        
    private:

};

#define _GPSQUAREDEXPONENTIAL_
#endif
