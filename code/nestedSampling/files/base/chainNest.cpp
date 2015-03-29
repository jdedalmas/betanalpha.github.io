// C++ stdlib
#include <iostream>
#include <vector>
#include "math.h"

// ROOT Libraries
#include "TRandom3.h"

// Local Libraries
#include "baseChain.h"
#include "chainNest.h"

using namespace std;


/// Constructor
/// \param random Pointer to external random number generator

chainNest::chainNest(TRandom3 *random): chainBundle(random) 
{

    mGoLikelihood = false;

    mNumNests = 200;
    mDisplayFrequency = 10;
    mTerm = 1e-6;

    mMaxNumSamples = 1e6;

    mMinLogL = 0;
    mLogZ = 0;
    mGradLogL = 0;
}

/// Compute the logarithm of the sum of two exponentials,
/// \f$ \log( \exp(a) + \exp(b) ) \f$
/// param a Argument of one exponential
/// param b Argument of the second
/// return logarithm of the summed exponentials

double fLogSumExp(double a, double b)
{

    if(a > b) 
    {
        return a + log( 1.0 + exp(b - a) );
    }
    else
    {      
        return b + log( 1.0 + exp(a - b) );
    }
    
}

/// Sample from the posterior distribution with nested sampling
/// See Betancourt, M. J., arXiv:1005.0157 (http://arxiv.org/abs/1005.0157)

void chainNest::nestedSample()
{

    // Initialize variables
    int nCompleteNests = 0;

    double nObjects = mChains.size();
    double nObjectsInverse = 1.0 / (double)nObjects;
    double logWidth = log(1.0 - exp(-nObjectsInverse) );
    double HSumOne = 0;
    double logHSumTwo = 0;
    double NSumOne = 0;
    double logNSumTwo = 0;
    double logNEff = 0;
    
    // Compute the likelihood of the initial samples
    double lArray[mChains.size()];
    
    for(int i = 0; i < nObjects; ++i) lArray[i] = fLogL(mChains.at(i)->x());
    
    if(mGoVerbose && mDisplayFrequency > 0) 
    {
        cout << "Beginning nested sampling," << endl;
        cout << "\tContinuing until (N + N_{i}^{eff}) * L_{i}^{max} * width_{i} / Z_{i} < " << mTerm << endl;
        cout << "\tAll diagnostic statistics assume the mean of the shrinkage distributions" << endl;
        cout << endl;
    }
    
    // Draw (x_{i}, L_{i}) samples
    for(unsigned int n = 0; n < mNumNests; ++n)
    {
    
        // Display current samples
        if(mGoVerbose && mDisplayFrequency > 0 && n != 0)
        {
        
            if(n % mDisplayFrequency == 0 || n == 1)
            {
        
                cout << "\tIteration " << n << endl;
                cout << "\t\t"
                     << "logZ = " << mLogZ 
                     << ", logWidth = " << logWidth
                     << ", H = " << mH 
                     << ", logNEff = " << logNEff << endl;
                
                cout << "\t\tStep size = " << mStepSize 
                     << ", Local metropolis accept rate = " << mXBar << endl;
                
                cout << "\t\tlogL : " << flush;
                for(int i = 0; i < nObjects; ++i) cout << lArray[i] << "\t" << flush;
                cout << endl << endl;
                
            }
            
        }
    
        // Find sample with smallest likelihood
        int minIndex = 0;
        mMinLogL = lArray[minIndex];
        double maxLogL = mMinLogL;
        
        double *lPtr = lArray + 1;
        for(int i = 1; i < nObjects; ++i, ++lPtr)
        {
        
            if(*lPtr < mMinLogL)
            {
                mMinLogL = *lPtr;
                minIndex = i;
            }
            
            if(*lPtr > maxLogL) maxLogL = *lPtr;
                
        }
        
        baseChain* minObject = mChains.at(minIndex);
        double logL = mMinLogL;
        double logWeight = logWidth + logL;
        
        // Update evidence and auxiliary entropies
        if(!n) 
        {
            mLogZ = logWeight;
            HSumOne = exp(logWeight) * logL;
            logHSumTwo = logWeight;
            NSumOne = exp(logWeight) * logWeight;
            logNSumTwo = logWeight;
        }
        else   
        {
            mLogZ = fLogSumExp(mLogZ, logWeight);
            HSumOne += exp(logWeight) * logL;
            logHSumTwo = fLogSumExp(logHSumTwo, logWeight);
            NSumOne += exp(logWeight) * logWeight;
            logNSumTwo = fLogSumExp(logNSumTwo, logWeight);
        }
        
        mH = exp(-mLogZ) * HSumOne - mLogZ * exp(-mLogZ + logHSumTwo);
        logNEff = - exp(-mLogZ) * NSumOne + mLogZ * exp(-mLogZ + logNSumTwo);
        
        // Store sample
        fStore(logL, minObject->x());

        // Break if Skilling criteria is satisfied
        if( exp(log(exp(logNEff) + nObjects) + maxLogL + logWidth - mLogZ) < mTerm )
        {
        
            if(mGoVerbose)
            {
                cout << "\tNested sampling finished early after " << nCompleteNests << " iterations" << endl;
            }
            
            break;
        
        }
        
        // Break if too many samples have been drawn
        if(mNumSamples > mMaxNumSamples)
        {

            if(mGoVerbose)
            {
             	cout << "Nested sampling terminating because of too many total "
                     << "samples after " << nCompleteNests << " nested samples." << endl;
            }

            break;

        }
        
        // Randomly select a new seed from the current samples
        int seed = minIndex;
        while(seed == minIndex) seed = floor(nObjects * mRandom->Rndm());

        // Draw new samples until likelihood bound is exceeded
        while(1)
        {
            *minObject = *(mChains.at(seed));
            if(sample(minObject)) break;
            if(mNumSamples > mMaxNumSamples) break;
        }
        
        lArray[minIndex] = fLogL(minObject->x());
        
        // Shrink sample width
        logWidth -= nObjectsInverse;
        
        ++nCompleteNests;
        
    }
    
    if(mGoVerbose)
    {
        if(nCompleteNests == mNumNests) cout << "\tTerminating nested sampling with" << endl;
        cout << "\t\t logZ = " << mLogZ << " +/- " << sqrt(mH * nObjectsInverse) << endl;
        cout << "\t\t H = " << mH << endl;
        cout << endl;
    }

    return;
    
}

/// Check for constraint violation.  If the prior distribution has finite support then
/// this includes a support constraint, and if chainNest::mGoLikelihood is set then the
/// likelihood constraint is also included.  Note that, because it is discontinuous,
/// the support constraint takes precedence over the likelihood constraint.
/// \param chain The chain currently being sampled
/// \return Is the constraint currently violated?

bool chainNest::fConstraint(baseChain *chain) 
{ 

    // Check for support violation
    if(chain->supportViolated())
    {
        return false;
    }
    // Check for likelihood constraint
    else if(mGoLikelihood)
    { 
        return fLogL(chain->x()) > mMinLogL;
    }
    
    return true;
    
}

/// Compute the normal of the constraint boundary.  Note that, because the support
/// constraint is discontinuous, the gradient will be infinite and any finite
/// likelihood constraint will be a negligle constribution. 
/// \param chain The chain currently being sampled
/// \return Pointer to the newly computed chainNest::mNormal

double* chainNest::fNormal(baseChain *chain)
{

    double* normal = chain->supportNormal();

    // Because a support violation results in an infinite
    // gradient, a support violation will take precedence
    // over a likelihood violation
    if(chain->supportViolated())
    {
        return normal;
    }
    // If the sample is within the appropriate support,
    // return the normal of the likelihood boundary
    else
    {
        double *gPtr = fGradLogL(chain->x());
        for(double *nPtr = normal; nPtr != normal + chain->dim(); ++nPtr, ++gPtr) *nPtr = *gPtr;
    }

    // Normalize
    double norm = 0;
    double *nPtr = normal;
    for(int i = 0; i < chain->dim(); ++i, ++nPtr) norm += (*nPtr) * (*nPtr);
    
    if(norm)
    {
    
        norm = sqrt(norm);
    
        nPtr = normal;
        for(int i = 0; i < chain->dim(); ++i, ++nPtr) *nPtr /= norm;
    
    }
    
    return normal;
    
}

/// Compute Monte Carlo posterior expectations assuming the means of the 
/// shrinkage distributions. Note that this virtual implementation computes 
/// the posterior expectation of only the first component of the feature space
/// \see computeSampledExpectations()

void chainNest::computeMeanExpectations()
{

    if(!mLogLStore.size())
    {
        cout << "chainNest::computeMeanExpectations() - No stored samples, expectations cannot be calculated!" << endl;
        return;
    }
    
    double mean = 0;
    double nInverse = 1.0 / (double)mChains.size();
    double logWidth = log(1.0 - exp(-nInverse) );
    
    // First nested sample
    vector<double>::iterator lIt = mLogLStore.begin();
    vector<double>::iterator sIt = mSampleStore.begin();
    
    double logWeight = logWidth + *lIt;
    mean += (*sIt) * exp(logWeight - mLogZ);
    
    ++lIt;
    ++sIt;
    
    // Loop over remaining nested samples
    for(int i = 1; i < mLogLStore.size(); ++i, ++lIt, ++sIt)
    {

        logWidth -= nInverse;
        logWeight = logWidth + *lIt;
        
        mean += (*sIt) * exp(logWeight - mLogZ);
        
    }
        
    cout << "<x> = " << mean << endl;
    cout << "logZ = " << mLogZ << " +/- " << sqrt(mH / (double)mChains.size() ) << endl;
    
    return;

}
        
/// Compute Monte Carlo posterior expectations, marginalizing over the 
/// shrinkage distributions. Note that this virtual implementation computes 
/// the posterior expectation of only the first component of the feature space.
/// \see computeMeanExpectations()
/// \param nSamples Number of Monte Carlo samples for each shrinkage distribution
        
void chainNest::computeSampledExpectations(int nSamples)
{

    if(!mLogLStore.size())
    {
        cout << "chainNest::computeSampledExpectations() - No samples stored, expectations cannot be calculated!" << endl;
        return;
    }

    double mean = 0;
    mH = 0;
    double nInverse = 1.0 / (double)mChains.size();
    
    vector<double>::iterator lIt;
    vector<double>::iterator sIt;
    vector<double> logWeightStore;
        
    for(int n = 0; n < nSamples; ++n)
    {
    
        // Sample x for each nested sample, compute corresponding logZ and H
        double xOld = 1;
        double xNew = 0;
        double logZ = 0;
        double HSumOne = 0;
        double logHSumTwo = 0;
        
        lIt = mLogLStore.begin();
        
        for(int i = 0; i < mLogLStore.size(); ++i, ++lIt)
        {
        
            // Sample x for the new nested sample, compute integral contribution
            double tau = - nInverse * log(1.0 - mRandom->Rndm());
            xNew = exp(-tau) * xOld;
    
            double logWeight = log(xOld - xNew) + (*lIt);
            
            // Increment evidence integral, information sums
            if(!i) 
            {
                logZ = logWeight;
                HSumOne = exp(logWeight) * (*lIt);
                logHSumTwo = logWeight;
            }
            else   
            {
                logZ = fLogSumExp(logZ, logWeight);
                HSumOne += exp(logWeight) * (*lIt);
                logHSumTwo = fLogSumExp(logHSumTwo, logWeight);
            }
            
            // Store weight for posterior expectations
            logWeightStore.push_back(logWeight);
            
            // Prepare for next iteration
            xOld = xNew;
    
        }
        
        mLogZ += logZ;
        mH += exp(-logZ) * HSumOne - logZ * exp(-logZ + logHSumTwo);
    
        // Compute posterior expectations for the sampled x
        double sampleMean = 0;
        
        sIt = mSampleStore.begin();

        for(vector<double>::iterator wIt = logWeightStore.begin(); wIt != logWeightStore.end(); ++wIt, ++sIt)
        {
            sampleMean += (*sIt) * exp(*wIt - logZ);
        }
        
        mean += sampleMean;
        
        // Clean up
        logWeightStore.clear();
    
    }
    
    mLogZ /= (double)nSamples;
    mH /= (double)nSamples;
    mean /= (double)nSamples;
    
    cout << "<x> = " << mean << endl;
    cout << "logZ = " << mLogZ << " +/- " << sqrt(mH / (double)mChains.size() ) << endl;
    
    return;

}

/// Store given sample.  Note that this virtual implementation stores only 
/// the first component of the sampled point
/// \param logL log likelihood of the nested sample
/// \param sample Pointer to the nested sample position

void chainNest::fStore(double logL, double *sample)
{

    mLogLStore.push_back(logL);
    mSampleStore.push_back(*sample);

    return;

}
        
