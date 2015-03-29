#include "math.h"
#include <iostream>

#include "gpSquaredExponential.h"

/// Constructor
/// \param dim Dimension of desired feature space

gpSquaredExponential::gpSquaredExponential(int dim): gpKernel(true, "squared exponential", "SquaredExponential")
{
    mDim = dim;
    mNumHyperParameters = mDim + 3;
    mHyperParameters = new double[mNumHyperParameters];
}

/// Destructor
gpSquaredExponential::~gpSquaredExponential()
{
    delete[] mHyperParameters;
}

double gpSquaredExponential::kernel(const double *x1, const double *x2, bool same)
{

    double arg = 0;
    double C = 0;

    const double *xOnePtr = x1;
    const double *xTwoPtr = x2;
    double *hyperPtr = mHyperParameters + 3;    
    for(int m = 0; m < mDim; ++m)
    {
        arg = (*hyperPtr) * (*xOnePtr - *xTwoPtr);
        C -= arg * arg;
        ++xOnePtr;
        ++xTwoPtr;
        ++hyperPtr; 
    }
    
    C = exp(C);
    C *= mHyperParameters[0] * mHyperParameters[0];
    C += mHyperParameters[1] * mHyperParameters[1];
    if(same) C += mHyperParameters[2] * mHyperParameters[2];
    
    return C;
}

double gpSquaredExponential::dKernel(const double *x1, const double *x2, bool same, const int n)
{
    
    double dC = 0;
    double C = 0;
    
    // Avoid unnecessary computations for the fast derivatives
    switch(n)
    {
        case 1:
            dC = 2 * mHyperParameters[1];
            return dC;
        case 2:
            if(same) dC = 2.0 * mHyperParameters[2];
            return dC;
        default:
            break;
    }
    
    double arg = 0;
    double argN = 0;

    const double *xOnePtr = x1;
    const double *xTwoPtr = x2;
    double *hyperPtr = mHyperParameters + 3;
    for(int m = 0; m < mDim; ++m)
    {
        arg = (*xOnePtr - *xTwoPtr);
        if(m == n - 3) argN = arg;
        arg *= *hyperPtr;
        C -= arg * arg;
        ++xOnePtr;
        ++xTwoPtr;
        ++hyperPtr;
    }
    
    C = exp(C);
    
    switch(n)
    {
        case 0:
            dC = 2.0 * mHyperParameters[0] * C;
            break;
        default:
            argN *= mHyperParameters[0];
            dC = - 2.0 * argN * argN * mHyperParameters[n] * C;
            break;
    }
    
    return dC;
    
}

void gpSquaredExponential::displayHyperParameters(const char* prefix)
{
    
    // Constants and coefficients
    cout << prefix << "theta1 = " << mHyperParameters[0] << endl;
    cout << prefix << "theta2 = " << mHyperParameters[1] << endl;
    cout << prefix << "theta3 = " << mHyperParameters[2] << endl;
    
    // Lengthscales
    for(int n = 0; n < mDim; ++n)
    {
        cout << prefix << "rho" << n + 1 << " = " << mHyperParameters[n + 3] << endl;
    }
        
}
