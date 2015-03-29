#include "math.h"
#include <iostream>

#include "gpNeuralNetwork.h"

/// Constructor
/// \param dim Dimension of desired feature space

gpNeuralNetwork::gpNeuralNetwork(int dim): gpKernel(true, "neural network", "NeuralNetwork")
{
    mDim = dim;
    mNumHyperParameters = mDim + 4;
    mHyperParameters = new double[mNumHyperParameters];
}

/// Destructor
gpNeuralNetwork::~gpNeuralNetwork()
{
    delete[] mHyperParameters;
}

double gpNeuralNetwork::kernel(const double *x1, const double *x2, bool same)
{

    double square = 0;
    double dotCross = 0;
    double dotOne = 0;
    double dotTwo = 0;
    double C = 0;
    
    const double *xOnePtr = x1;
    const double *xTwoPtr = x2;
    double *hyperPtr = mHyperParameters + 4;
    for(int m = 0; m < mDim; ++m)
    {
        square = (*hyperPtr) * (*hyperPtr);
        dotCross += square * (*xOnePtr) * (*xTwoPtr);
        dotOne += square * (*xOnePtr) * (*xOnePtr);
        dotTwo += square * (*xTwoPtr) * (*xTwoPtr);
        ++xOnePtr;
        ++xTwoPtr;
        ++hyperPtr;
    }
    
    dotCross += mHyperParameters[3] * mHyperParameters[3];
    dotOne += mHyperParameters[3] * mHyperParameters[3];
    dotTwo += mHyperParameters[3] * mHyperParameters[3];
    
    dotOne *= 2.0;
    dotTwo *= 2.0;
    
    dotOne += 1.0;
    dotTwo += 1.0;
    
    C = 2 * dotCross / sqrt(dotOne * dotTwo);
    C = 2.0 * asin(C) / 3.14159265; 
    C *= mHyperParameters[0] * mHyperParameters[0];
    C += mHyperParameters[1] * mHyperParameters[1];
    if(same) C += mHyperParameters[2] * mHyperParameters[2];
    
    return C;    
    
}

double gpNeuralNetwork::dKernel(const double *x1, const double *x2, bool same, const int n)
{
    
    double square = 0;
    double dotCross = 0;
    double dotOne = 0;
    double dotTwo = 0;
    double y = 0;
    double dCdh = 0;
    
    switch(n)
    {
        case 1:
            dCdh = 2.0 * mHyperParameters[1];
            return dCdh;
        case 2:
            if(same) dCdh = 2.0 * mHyperParameters[2];
            return dCdh;
        default:
            break;
    }
    
    const double *xOnePtr = x1;
    const double *xTwoPtr = x2;
    double *hyperPtr = mHyperParameters + 4;
    for(int m = 0; m < mDim; ++m)
    {
        square = (*hyperPtr) * (*hyperPtr);
        dotCross += square * (*xOnePtr) * (*xTwoPtr);
        dotOne += square * (*xOnePtr) * (*xOnePtr);
        dotTwo += square * (*xTwoPtr) * (*xTwoPtr);
        ++xOnePtr;
        ++xTwoPtr;
        ++hyperPtr;
    }
    
    dotCross += mHyperParameters[3] * mHyperParameters[3];
    dotOne += mHyperParameters[3] * mHyperParameters[3];
    dotTwo += mHyperParameters[3] * mHyperParameters[3];
    
    dotOne *= 2.0;
    dotTwo *= 2.0;
    
    dotOne += 1.0;
    dotTwo += 1.0;
    
    y = 2.0 * dotCross / sqrt(dotOne * dotTwo);
    
    switch(n)
    {
        case 0:
            dCdh = 2.0 * asin(y) / 3.14159265; 
            dCdh *= 2.0 * mHyperParameters[0]; 
            return dCdh; 
        case 3:
            dCdh = 1.0 / dotCross;
            dCdh -= 1.0 / dotOne;
            dCdh -= 1.0 / dotTwo;
            dCdh *= mHyperParameters[3];
            break;
        default:
            double x1n = *(x1 + n - 4);
            double x2n = *(x2 + n - 4);
            dCdh = x1n * x2n /  dotCross;
            dCdh -= x1n * x1n / dotOne;
            dCdh -= x2n * x2n / dotTwo;
            dCdh *= mHyperParameters[n];
            break;
    }
    
    dCdh *= 4.0 * y / 3.14159265;
    dCdh /= sqrt(1 - y * y);
    dCdh *= mHyperParameters[0] * mHyperParameters[0];
    
    return dCdh;
    
}

void gpNeuralNetwork::displayHyperParameters(const char* prefix)
{

    // Constants and coefficients
    cout << prefix << "theta1 = " << mHyperParameters[0] << endl;
    cout << prefix << "theta2 = " << mHyperParameters[1] << endl;
    cout << prefix << "theta3 = " << mHyperParameters[2] << endl;
    cout << prefix << "sigma0 = " << mHyperParameters[3] << endl;
    
    // Lengthscales
    for(int n = 0; n < mDim; ++n)
    {
        cout << prefix << "sigma" << n + 1 << " = " << mHyperParameters[n + 4] << endl;
    }
        
}
