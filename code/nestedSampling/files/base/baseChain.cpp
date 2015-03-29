#include "baseChain.h"

/// Constructor             
/// \param dim The dimension of the feature space
/// \see ~baseChain()    

baseChain::baseChain(int dim)
{

    mDim = dim;
    
    mNumAccept = 0;
    mNumReject = 0;
    
    // Allocate memory for local arrays
    mPoint = new double[mDim];
    mStorePoint = new double[mDim];
    mBurnedPoint = new double[mDim];
    mMomentum = new double[mDim];
    mInverseMass = new double[mDim];
    
    mGradE = new double[mDim];
    
    mNormal = new double[mDim];
    
    mBurnSum = new double[mDim];
    mBurnSum2 = new double[mDim];
    
    // Initialize variables
    mE = 0;
    for(double *ptr = mPoint; ptr != mPoint + mDim; ++ptr) *ptr = 0;
    for(double *ptr = mStorePoint; ptr != mStorePoint + mDim; ++ptr) *ptr = 0;
    for(double *ptr = mBurnedPoint; ptr != mBurnedPoint + mDim; ++ptr) *ptr = 0;
    for(double *ptr = mMomentum; ptr != mMomentum + mDim; ++ptr) *ptr = 0;
    for(double *ptr = mInverseMass; ptr != mInverseMass + mDim; ++ptr) *ptr = 1.0;
    
    for(double *ptr = mGradE; ptr != mGradE + mDim; ++ptr) *ptr = 0;
    
    for(double *ptr = mNormal; ptr != mNormal + mDim; ++ptr) *ptr = 0;
    
    mNumBurnCheck = 0;
    for(double *ptr = mBurnSum; ptr != mBurnSum + mDim; ++ptr) *ptr = 0;
    for(double *ptr = mBurnSum2; ptr != mBurnSum2 + mDim; ++ptr) *ptr = 0;

}

/// Destructor     
/// \see baseChain()            

baseChain::~baseChain()
{

    delete[] mPoint;
    delete[] mStorePoint;
    delete[] mBurnedPoint;
    delete[] mMomentum;
    delete[] mInverseMass;
    
    delete[] mGradE;
    
    delete[] mNormal;
    
    delete[] mBurnSum;
    delete[] mBurnSum2;

}

/// Copy Constructor   
/// \param chain The baseChain to copy
/// \see operator= (const baseChain& object)

baseChain::baseChain(const baseChain& chain)
{

    mE = chain.E();

    for(int i = 0; i < mDim; ++i)
    {
        mPoint[i] = chain.x()[i];
        mMomentum[i] = chain.p()[i];
        mGradE[i] = chain.gradE()[i];
    }
    
}

/// Operator =    
/// \param chain The baseChain to copy
/// \see baseChain(const baseChain& object)              

baseChain& baseChain::operator = (const baseChain& chain)
{

    if(this == &chain) return *this;

    mE = chain.E();

    for(int i = 0; i < mDim; ++i)
    {
        mPoint[i] = chain.x()[i];
        mMomentum[i] = chain.p()[i];
        mGradE[i] = chain.gradE()[i];
    }
    
    return *this;
    
}

/// Return squared norm of the momentum   

double baseChain::pMp() 
{ 

    double temp = 0;
    
    double *mPtr = mInverseMass;
    for(double *ptr = mMomentum; ptr != mMomentum + mDim; ++ptr, ++mPtr) 
    {
        temp += (*ptr) * (*ptr) * (*mPtr);
    }
    
    return temp; 
    
}

/// Return energy, updating calculation if desired 
/// \param recompute Boolean switch for recomputing the energy
/// \return baseChain::mE   

double baseChain::E(bool recompute) 
{ 
    if(recompute) fComputeE();
    return mE; 
}

/// Return the log of the probability distribution, updating calculation if desired 
/// \param recompute Boolean switch for recomputing the log probability
/// \return The log probability distribution at the current point, -E()
/// \see E()

double baseChain::logP(bool recompute) 
{ 
    if(recompute) fComputeE();
    return -E(); 
}

/// Return gradient of the energy, updating calculation if desired   
/// \param recompute Boolean switch for recomputing the energy gradient
/// \return baseChain::mGradE
        
double* baseChain::gradE(bool recompute) 
{ 
    if(recompute) fComputeGradE();
    return mGradE; 
}

/// Clear all convergence diagnostic statistics 
/// \see incrementChainSums()

void baseChain::clearChainStats()
{
    mNumBurnCheck = 0;
    mNumAccept = 0;
    mNumReject = 0;
    for(double *ptr = mBurnSum; ptr != mBurnSum + mDim; ++ptr) *ptr = 0;
    for(double *ptr = mBurnSum2; ptr != mBurnSum2 + mDim; ++ptr) *ptr = 0;
}

/// Increment convergence diagnostic statistics 
/// \see clearChainStats()

void baseChain::incrementChainSums()
{

    ++mNumBurnCheck;
    
    for(int i = 0; i < mDim; ++i)
    {
        mBurnSum[i] += mPoint[i];
        mBurnSum2[i] += mPoint[i] * mPoint[i];
    }

}

/// Store current point, used for Metropolis-Hastings rejection 
/// \see restoreStorePoint()   

void baseChain::storeCurrentPoint()
{
    for(double *ptrOne = mPoint, *ptrTwo = mStorePoint; ptrOne != mPoint + mDim; ++ptrOne, ++ptrTwo) 
    {
        *ptrTwo = *ptrOne;
    }
}

/// Restore stored point, used for Metropolis-Hastings rejection   
/// \see storeCurrentPoint() 

void baseChain::restoreStoredPoint()
{
    for(double *ptrOne = mPoint, *ptrTwo = mStorePoint; ptrOne != mPoint + mDim; ++ptrOne, ++ptrTwo) 
    {
        *ptrOne = *ptrTwo;
    }
}

/// Store current point, used for restarting markov chain 
/// \see restoreBurnedPoint()       

void baseChain::storeBurnedPoint()
{
    for(double *ptrOne = mPoint, *ptrTwo = mBurnedPoint; ptrOne != mPoint + mDim; ++ptrOne, ++ptrTwo) 
    {
        *ptrTwo = *ptrOne;
    }
}

/// Restore stored point, used for restarting markov chain
/// \see storeBurnedPoint()

void baseChain::restoreBurnedPoint()
{
    for(double *ptrOne = mPoint, *ptrTwo = mBurnedPoint; ptrOne != mPoint + mDim; ++ptrOne, ++ptrTwo) 
    {
        *ptrOne = *ptrTwo;
    }
}
