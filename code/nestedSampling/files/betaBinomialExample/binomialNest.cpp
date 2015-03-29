#include <iostream>
#include <math.h>

#include "TRandom3.h"

#include "binomialNest.h"
#include "gaussMixer.h"

/// Constructor
/// \see ~binomialNest()
/// \param k Number of successful trials in data
/// \param N Total number of trials in data
/// \param random Pointer to external random number generator

binomialNest::binomialNest(double k, double N, TRandom3 *random): chainNest(random)
{

    mK = k;
    mN = N;
    mMu = 0;
    
    mGradLogL = new double[1];

}

/// Destructor
/// \see binomialNest()

binomialNest::~binomialNest()
{
    delete[] mGradLogL;
}

/// Infer the modal structure of the posterior samples
/// generated from nested sampling.  Note that great care
/// must be taken with this approach: nested sampling
/// will quickly pass through the posterior modes, 
/// the effective number of samples will be small,
/// and the gaussian mixture model inference will be
/// dominated by the priors.  If the posterior is critical
/// then run a HMC chain sampling from the posterior
/// directly (possibly seeded from the nested samples with
/// highest weight) and use the resulting unweighted samples
/// to determine expectations or as input to the gaussian
/// mixture model inference.  

void binomialNest::inferPosteriorModes()
{

    if(!mLogLStore.size())
    {
        cout << "chainNest::computeMeanExpectations() - No stored samples, expectations cannot be calculated!" << endl;
        return;
    }
    
    double nInverse = 1.0 / (double)mChains.size();
    double logWidth = log(1.0 - exp(-nInverse) );
    
    double mu[mLogLStore.size()];
    double weight[mLogLStore.size()];
    
    // First nested sample
    vector<double>::iterator lIt = mLogLStore.begin();
    vector<double>::iterator sIt = mSampleStore.begin();
    
    double logWeight = logWidth + *lIt;

    mu[0] = (*sIt); weight[0] = exp(logWeight - mLogZ);
    double maxWeight = weight[0];
    
    ++lIt; ++sIt;
    
    // Loop over remaining nested samples
    for(int i = 1; i < mLogLStore.size(); ++i, ++lIt, ++sIt)
    {

        logWidth -= nInverse;
        logWeight = logWidth + *lIt;
        
        mu[i] = (*sIt); weight[i] = exp(logWeight - mLogZ);
        
        if(weight[i] > maxWeight) maxWeight = weight[i];
        
    }
        
    for(int i = 0; i < mLogLStore.size(); ++i) weight[i] /= maxWeight;
    
    gaussMixer posteriorModes(mu, mLogLStore.size(), 6);
    posteriorModes.cluster(weight);
  
    for(int k = 0; k < posteriorModes.nComponents(); ++k)
    {
        
        cout << "\t" << k << "\t" << posteriorModes.pi()[k] << "\t" << posteriorModes.N()[k] << endl;
        cout << "Mixture Component " << k << endl;
        cout << "\tpi_{k} = " << posteriorModes.pi()[k]
             << ", N_{k} = "  << posteriorModes.N()[k]
             << ", mu_{k} = " << posteriorModes.m()[k]
             << endl;
    }
    
    return;

}

/// Calculate the binomial log likelihood at the given mu
/// \param Pointer to the current mu
/// \return Binomial log likelihood at the given mu

double binomialNest::fLogL(double *point)
{
    
    double logL = mK * log(*point) + (mN - mK) * log(1 - *point);
    for(int i = mN - mK + 1; i < mN + 1; ++i) logL += log(i);
    for(int i = 1; i < mK + 1; ++i) logL -= log(i);
    
    return logL;
    
}

/// Calculate the gradient of binomial the log likelihood
/// \param Pointer to the current mu
/// \return Pointer to the newly computed chainNest::mGradLogL

double* binomialNest::fGradLogL(double *point)
{
    *mGradLogL = mK / *point - (mN - mK) / (1 - *point);
    return mGradLogL;
}

