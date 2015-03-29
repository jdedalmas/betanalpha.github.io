#ifndef _GPNEURALNETWORK_

#include "gpKernel.h"

///
/// \author Michael Betancourt,
/// Massachusetts Institute of Technology
///
/// Implementation of a neural network covariance function
///
/*! 
 \f[
 C(i, j) = \theta_{1}^{2} \frac{2}{\pi}
 \arcsin \left( \frac{2 X_{i}^{T}
 \Sigma X_{j} } { \sqrt{ (1 + 2 X_{i}^{T} 
 \Sigma X_{i} ) (1 + 2 X_{j}^{T} \Sigma X_{j} 
  ) } } \right) + \theta_{2}^{2}
 + \theta_{3}^{2} * \delta_{ij}
 \f]
*/
/// where \f$X_{i} = (1, x_{i})\f$
/// and \f$ \Sigma = \mathrm{diag} \left(\sigma_{0}^{2},\sigma_{i}^{2}\right) \f$

class gpNeuralNetwork: public gpKernel
{

    public:
        gpNeuralNetwork(int dim);
        ~gpNeuralNetwork();

        double kernel(const double *x1, const double *x2, bool same);
        double dKernel(const double *x1, const double *x2, bool same, const int n);
        
        void displayHyperParameters(const char* prefix = "");
        TString displayRelevance() { return "r_{i} = sigma_{i}^{2}"; }
        
    private:

        
};

#define _GPNEURALNETWORK_
#endif
