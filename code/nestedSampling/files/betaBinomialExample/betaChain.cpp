#include "betaChain.h"

#include "math.h"


/// Constructor
/// \param alpha \f$\alpha\f$ parameter of the Beta distribution
/// \param beta \f$\beta\f$ parameter of the Beta distribution 

betaChain::betaChain(double alpha, double beta): baseChain(1) 
{
    mAlpha = alpha;
    mBeta = beta;
}

/// Check if the current point is within the allowed support
/// \return Is the appropriate support violated?

bool betaChain::supportViolated()
{
    if(*mPoint < 0 || *mPoint > 1) return true;
    return false;
}

/// Compute the normal to the support constraint

double* betaChain::supportNormal()
{

    if(*mPoint < 0) *mNormal = 1;
    else if(*mPoint > 1) *mNormal = -1;
    
    return mNormal;
    
}

/// Compute the potential energy at the current point 

void betaChain::fComputeE()
{
    mE = (1 - mAlpha) * log(*mPoint) + (1 - mBeta) * log(1 - *mPoint);
}

/// Compute the gradient of the potential at the current point

void betaChain::fComputeGradE()
{
    *mGradE = (1 - mAlpha) / *mPoint - (1 - mBeta) / (1 - *mPoint);
}
