// C++ standard library
#include "math.h"
#include <iostream>

// ROOT libraries
#include "TRandom3.h"

// Local libraries
#include "baseChain.h"
#include "chainBundle.h"


/// Constructor
/// \param random Pointer to external random number generator
/// \see ~chainBundle()

chainBundle::chainBundle(TRandom3 *random)
{

    mGoVerbose = false;
    mGoBurn = false;
    mChar = false;

    mRandom = random;

    mNumDump = 5;
    mNumLeapfrog = 150;
    mStepSize = 0.1;
    mStepSizeJitter = 0.001;
    mMaxStepSize = 1.0;
    mMinStepSize = 0.0001;
    
    mTargetAcceptRate = 0.7;
    mXBar = 0;
    mN = 0;
    mAdaptStepSize = true;
    mAlpha = 0.95;
    mLambda = 1.0 / 15.0;
    
    mNumSamples = 0;
    
    mNumBurn = 0;
    mNumCheck = 0;
    mMinR = 0;
    
}


/// Destructor
/// \see chainBundle()

chainBundle::~chainBundle()
{
    for(mIt = mChains.begin(); mIt != mChains.end(); ++mIt) delete (*mIt);
    mChains.clear();
}

/// Set valid step size jitter          
///                                              
/// Note that the jitter is defined as a fraction of the current step size,
/// so that the jitter will scale as the step size is adaptively updated          
/// \param j Stepsize jitter                                              


void chainBundle::setStepSizeJitter(double j)
{

    if(j > 1)
    {
        cout << "chainBundle::setStepSizeJitter() - Step size jitter cannot exceed 1!" << endl;
        return;
    }
    
    mStepSizeJitter = j;

}

/// Set burn in parameters          
/// \param nBurn Desired chainBundle::mNumBurn
/// \param nCheck Desired chainBundle::mNumCheck
/// \param minR Desired chainBundle::mMinR

void chainBundle::setBurnParameters(int nBurn, int nCheck, double minR)
{

    mNumBurn = nBurn;
    mNumCheck = nCheck;
    mMinR = minR;
    mGoBurn = true;

}

/// Sample all chains

void chainBundle::sample()
{
    for(mIt = mChains.begin(); mIt != mChains.end(); ++mIt) sample(*mIt);
}

/// Sample from the ith chain
/// \param i Index of chain to be sampled

void chainBundle::sample(int i)
{

    if(i < 0 || i >= mChains.size())
    {
        cout << "chainBundle::sample() - Bad chain index " << i << "!" << endl;
        return;
    }

    sample(mChains.at(i));
    
    return;
    
}

/// Draw a baseNesteObject::nNumDump consecutive samples from the given Markov chain
/// using Constrained Hamiltonian Monte Carlo, reserving the final state as a sample candidate
///
/// In anticipation of nested sampling, the leapfrog stepsize is adaptively updated, scaled
/// by exp( mLambda / (mTargetAcceptRate - 1.0) ) after each rejection, and
/// exp( mLambda / mTargetAcceptRate ) after each acceptance.  In equilibrium, the expected
/// Metropolis accept rate converges to mTargetAcceptRate and the expected steps size
/// converges to a constant value.
///
/// mLambda should be kept small in order to avoid awkward behavior and biased samples.
/// 1 / 15 or smaller has proven successful in practice.

/// \param chain The given Markov chain
/// \return Was the final Metropolis proposal accepted?

bool chainBundle::sample(baseChain *chain)
{

    // Check that the chain was properly seeded
    if(mNumSamples == 0)
    {
    
        if(chain->supportViolated())
        {
            cout << "chainBundle::sample() - Initial chain parameters in violation "
                 << "of support constraint, aborting sample!" << endl;
            return false;
        }
    
    }

    int dim = chain->dim();
    bool constraint = true;
    bool nanFlag = false;

    for(int n = 0; n < mNumDump; ++n)
    {

        double dH = 0;
        chain->storeCurrentPoint();
        
        // Add random jitter to the step size to avoid closed loops
        double stepSize = mStepSize * (1.0 + mStepSizeJitter * (2.0 * mRandom->Rndm() - 1.0) );
        
        // Sample momenta
        double *mPtr = chain->mInv();
        for(double *ptr = chain->p(); ptr != chain->p() + dim; ++ptr, ++mPtr)
        {
            *ptr = mRandom->Gaus(0, sqrt(1.0 / *mPtr));
        }

        // Calculate Hamiltonian at the sample to be replaced
        double H = 0.5 * chain->pMp() + chain->E(1);
        
        //////////////////////////////////////////////////
        //        Evolve through the feature space      //
        //          for mNumLeapfrog iterations         //
        //////////////////////////////////////////////////
        
        // First momentum half step
        double *xPtr = chain->x();
        double *pPtr = chain->p();
        double *gPtr = chain->gradE(1);
        double *nPtr;
        
        for(int i = 0; i < dim; ++i, ++pPtr, ++gPtr) *pPtr += - 0.5 * stepSize * (*gPtr);
        
        for(unsigned int t = 0; t < mNumLeapfrog - 1; ++t)
        {

            // Check for any NANs that have sneaked in,
            // restarting if any are found
            nanFlag = false;
            for(int i = 0; i < dim; ++i) if(chain->p()[i] != chain->p()[i]) nanFlag |= true;
            if(nanFlag) break;

            //  Full spatial step
            xPtr = chain->x();
            pPtr = chain->p();
            mPtr = chain->mInv();
            for(int i = 0; i < dim; ++i, ++xPtr, ++pPtr, ++mPtr) 
            {
                *xPtr += stepSize * (*mPtr) * (*pPtr);
            }

            // Check for constraint violation, reflecting the momenta as necessary
            if(!fConstraint(chain))
            {

                double *normal = fNormal(chain);
                
                // Inner product of normal and momentum
                double dot = 0;
                
                nPtr = normal;
                pPtr = chain->p();
                for(int i = 0; i < dim; ++i, ++nPtr, ++pPtr) dot += (*nPtr) * (*pPtr);
                
                // Bounce
                nPtr = normal;
                pPtr = chain->p();
                for(int i = 0; i < dim; ++i, ++pPtr, ++nPtr) *pPtr = *pPtr - 2.0 * dot * (*nPtr);            

                constraint = false;     
                
            }
            else
            {
                constraint = true;
            }
            
            // Following two momenta half steps
            if(constraint)
            {
            
                pPtr = chain->p();
                gPtr = chain->gradE(1);
                
                for(int i = 0; i < dim; ++i, ++pPtr, ++gPtr)
                {
                    *pPtr += - stepSize * (*gPtr);
                }
                        
            }
        
        }
        
        // Restart if loop ended with a NAN
        if(nanFlag)
        {
            cout << "chainBundle::sample() - Restarting after encountering a NAN..." << endl;
            chain->restoreStoredPoint();
            n--;
            continue;
        }
        
        // Last full spatial step
        xPtr = chain->x();
        pPtr = chain->p();
        mPtr = chain->mInv();
        for(int i = 0; i < dim; ++i, ++xPtr, ++pPtr, ++mPtr) *xPtr += stepSize * (*mPtr) * (*pPtr);

        // And last momentum half step
        pPtr = chain->p();
        gPtr = chain->gradE(1);
        for(int i = 0; i < dim; ++i, ++pPtr, ++gPtr) *pPtr += - 0.5 * stepSize * (*gPtr);
        
        // Calculate change the in the Hamiltonian
        dH = 0.5 * chain->pMp() + chain->E(1);
        dH -= H;
        
        // Determinte accept/reject
        
        // Reject if constraint is not satisfied,
        // corresponding to dH = - \infty
        if(!fConstraint(chain))
        {
            chain->restoreStoredPoint();
            chain->incrementReject();
            constraint = false;
        }
        else
        {
           
            // Accept with probability exp(-dH)
            double ratioP = exp(-dH);
            
            if( ratioP > 1 ) 
            {
                chain->incrementAccept();
                constraint = true;
            }
            else if(mRandom->Rndm() > ratioP ) 
            {
                chain->restoreStoredPoint();
                chain->incrementReject();
                constraint = false;
            }
            else
            {
                chain->incrementAccept();
                constraint = true;
            }
        
        }
        
        // Update moving average
        mXBar = mAlpha * mN * mXBar + (double)constraint;
        mN = mAlpha * mN + 1.0;
        mXBar /= mN;

        // Update step size
        if(mAdaptStepSize)
        {
        
            if(constraint)
            {
                mStepSize *= exp(mLambda / mTargetAcceptRate);
                
            }
            else
            {
                mStepSize *= exp( mLambda / (mTargetAcceptRate - 1.0) );
            }
            
            mStepSize = mStepSize > mMaxStepSize ? mMaxStepSize : mStepSize;
            mStepSize = mStepSize < mMinStepSize ? mMinStepSize : mStepSize;
        
        }
        
    }
    
    ++mNumSamples;
    
    return constraint;

}

/// Seed all chains, using the same bounds for all components
/// \param min Minimum bound for all components
/// \param max Maximum bound for all components

void chainBundle::seed(double min, double max)
{
    for(int i = 0; i < mChains.at(0)->dim(); ++i) seed(i, min, max);
}

/// Seed the ith component of the feature space for all chains
/// \param i The selected component of the feature space
/// \param min Minimum bound for the ith component
/// \param max Maximum bound for the ith component

void chainBundle::seed(int i, double min, double max)
{

    if(i < 0 || i >= mChains.at(0)->dim())
    {
        cout << "chainBundle::seed() - Bad parameter index!" << endl;
        return;
    }
    
    for(mIt = mChains.begin(); mIt != mChains.end(); ++mIt) 
    {
        (*mIt)->x()[i] = (max - min) * mRandom->Rndm() + min;
    }

}

/// Burn in all chains until the Gelman convergence critieria has been satisfied
///
/// Gelman, A.
/// Inference and monitoring convergence,
/// "Markov Chain Monte Carlo in Practice"
/// (1996) Chapman & Hall/CRC, New York
///

void chainBundle::burn()
{

    int dim = mChains.at(0)->dim();
    double m = (double)mChains.size();
      
    // burn in markov chain
    if(mGoVerbose) cout << "Burn, baby, burn" << flush;
    for(int i = 0; i < mNumBurn; ++i) 
    {
        sample();
        if(mGoVerbose) if(i % 10 == 0) cout << "." << flush;
    }
    if(mGoVerbose) cout << endl;
    
    // Check for convergence
    double B[dim];                 // Between chain variance
    double W[dim];                 // Within chain variance
    double R[dim];                 // R something potential something
    
    double sampleMean = 0;
    double ensembleSum[dim];
    double ensembleSum2[dim];
    
    bool burning = true;
    
    if(mGoVerbose) cout << "Computing diagnostics:" << endl;
    
    while(burning)
    {
    
        // Clear diagnostic statistics
        for(mIt = mChains.begin(); mIt != mChains.end(); ++mIt) (*mIt)->clearChainStats();

        // burn through diagnostic samples
        for(int i = 0; i < mNumCheck; ++i) 
        {
        
            for(mIt = mChains.begin(); mIt != mChains.end(); ++mIt) 
            {
                sample(*mIt);
                (*mIt)->incrementChainSums();
            }
            
        }
        
        // Compute diagnostic statistics
        for(double *ptr = B; ptr != B + dim; ++ptr) *ptr = 0;
        for(double *ptr = W; ptr != W + dim; ++ptr) *ptr = 0;
        for(double *ptr = ensembleSum; ptr != ensembleSum + dim; ++ptr) *ptr = 0;
        for(double *ptr = ensembleSum2; ptr != ensembleSum2 + dim; ++ptr) *ptr = 0;
        
        double n = mChains.at(0)->nBurn();

        for(mIt = mChains.begin(); mIt != mChains.end(); ++mIt) 
        {
            for(int i = 0; i < dim; ++i)
            {
                W[i] += ( (*mIt)->burnSum2()[i] - (*mIt)->burnSum()[i] * (*mIt)->burnSum()[i] / n ) / (n - 1);
                sampleMean = (*mIt)->burnSum()[i] / n;
                ensembleSum[i] += sampleMean;
                ensembleSum2[i] += sampleMean * sampleMean;
            }
        }
        
        burning = false;
        for(int i = 0; i < dim; ++i)
        {
            W[i] /= m;
            B[i] = ( ensembleSum2[i] - ensembleSum[i] * ensembleSum[i] / m ) / (m - 1);
            R[i] = (n - 1) / n + B[i] / W[i];
            burning |= (R[i] > mMinR);
        }
        
        if(mGoVerbose && burning)
        {
            cout << "\tDiagnostic test failed," << endl;
            
            int nPerLine = 5;
            int nLines = dim / nPerLine;
            
            int k = 0;
            for(int i = 0; i < nLines; ++i)
            {
            
                cout << "\t\t" << flush;
            
                for(int j = 0; j < nPerLine; ++j, ++k)
                {
                    cout << "R_{" << k << "} = " << R[k] << ", " << flush;
                }
                
                cout << endl;
            
            }
            
            if(dim % nPerLine > 0)
            {
            
                cout << "\t\t" << flush;
                for(int i = 0; i < dim % nPerLine; ++i, ++k)
                {
                    cout << "R_{" << k << "} = " << R[k] << ", " << flush;
                }
                cout << endl;
            
            }
            
        }
    
    }

    // Display convergence details if desired
    if(mGoVerbose)
    {
        cout << "\tMarkov chains converged with" << endl;
        
        int nPerLine = 5;
        int nLines = dim / nPerLine;
        
        int k = 0;
        for(int i = 0; i < nLines; ++i)
        {
        
            cout << "\t\t" << flush;
        
            for(int j = 0; j < nPerLine; ++j, ++k)
            {
                cout << "R_{" << k << "} = " << R[k] << ", " << flush;
            }
            
            cout << endl;
        
        }
        
        if(dim % nPerLine > 0)
        {
        
            cout << "\t\t" << flush;
            for(int i = 0; i < dim % nPerLine; ++i, ++k)
            {
                cout << "R_{" << k << "} = " << R[k] << ", " << flush;
            }
            cout << endl;
        
        }
        
        cout << endl;
        cout << "\tIndividual Metropolis accept rates:" << endl;
        int i = 0;
        for(mIt = mChains.begin(); mIt != mChains.end(); ++mIt, ++i) 
        {
            cout << "\t\t Chain_{" << i << "}, P(accept) = " << (*mIt)->acceptRate() << endl;
        }
        cout << endl;
        
    }
    
    mChar = true;
    
    // Compute Metropolis statistics
    if(mGoVerbose)
    {
        cout << "\tLocal Metropolis accept rate = " << mXBar << endl;
        cout << "\tStep size = " << mStepSize << endl;
        cout << endl;
    }
    
    return;

}

/// Store current state of all chains in the bundle
/// \see restoreBurnedPoints()

void chainBundle::storeBurnedPoints()
{
    for(mIt = mChains.begin(); mIt != mChains.end(); ++mIt) (*mIt)->storeBurnedPoint();
}


/// Restore stored state of all chains in the bundle
/// \see storedBurnedPoints()

void chainBundle::restoreBurnedPoints()
{
    for(mIt = mChains.begin(); mIt != mChains.end(); ++mIt) (*mIt)->restoreBurnedPoint();

}

