#include <iostream>
#include "math.h"

#include "TRandom3.h"
#include "TMath.h"

#include "gaussMixer.h"
#include "gaussMixerMatrix.h"

using namespace std;

/// Constructor
/// \param x Array of input data, the values of each sample concatenated into a single array
/// \param nData Number of samples
/// \param dim Dimensionality of each sample
/// \param nComponents Number of mixture components
/// \see ~gaussMixer

gaussMixer::gaussMixer(double *x, int nData, int dim, int nComponents)
{

    mData = x;
    mNumData = nData;
    mDim = dim;
    mNumComponents = nComponents;
    
    // Defaults
    mNumMaxIterations = 2500;
    mEpsilon = 1e-8;
    
    mAlpha0 = 1e-3;
    mBeta0 = 1;
    mNu0 = mDim - 1 + 1e-3;
    
    mM0 = new double[mDim];
    for(double* ptr = mM0; ptr != mM0 + mDim; ++ptr) *ptr = 0;
    
    mW0inverse = new double[mDim * mDim];
    for(double* ptr = mW0inverse; ptr != mW0inverse + mDim * mDim; ++ptr) *ptr = 0;
    for(int i = 0; i < mDim; ++i) mW0inverse[i * mDim + i] = 1e-3;
    
    mLogEvidence = 0;
    
    mN = new double[mNumComponents];
    for(double* ptr = mN; ptr != mN + mNumComponents; ++ptr) *ptr = 0;
    
    mM = new double[mNumComponents * mDim];
    for(double* ptr = mM; ptr != mM + mNumComponents * mDim; ++ptr) *ptr = 0;
    
    mSigma = new double[mNumComponents * mDim * mDim];
    for(double* ptr = mSigma; ptr != mSigma + mNumComponents * mDim * mDim; ++ptr) *ptr = 0;
    
    mPi = new double[mNumComponents];
    for(double* ptr = mPi; ptr != mPi + mNumComponents; ++ptr) *ptr = 0;
    
    mR = new double[mNumComponents * mNumData];
    for(double* ptr = mR; ptr != mR + mNumComponents * mNumData; ++ptr) *ptr = 0;

}

/// Destructor
/// \see gaussMixer(double *x, int nData, int dim, int nComponents)

gaussMixer::~gaussMixer()
{

    delete[] mM0;
    delete[] mW0inverse;
    delete[] mN;
    delete[] mM;
    delete[] mSigma;
    delete[] mPi;
    delete[] mR;

}

/// Compute the variational solution to the approximate mixture model posterior

void gaussMixer::cluster()
{

    //////////////////////////////////////////////////
    //             Instantiate Variables            //
    //////////////////////////////////////////////////

    double alphaHat = 0;

    double xBar[mDim * mNumComponents];
    double S[mDim * mDim * mNumComponents];
    double alpha[mNumComponents];
    double beta[mNumComponents];
    double nu[mNumComponents];
    double W[mDim * mDim * mNumComponents];
    double Winverse[mDim * mDim * mNumComponents];
    double L[mDim * mDim];
    double LT[mDim * mDim];
    double logPiTwidle[mNumComponents];
    double logLambdaTwidle[mNumComponents];
    double logW[mNumComponents];
    
    // Avoid casting
    double auxDim = (double)mDim;
    double auxNumComponents = (double)mNumComponents;
        
    // Fire up the random number generator
    TRandom3 random(0);
        
    //////////////////////////////////////////////////
    //        Compute Variational Distribution      //
    //////////////////////////////////////////////////
    
    // Compute constant terms in the logEvidence lower bound
    
    fCholesky(mW0inverse, L, mDim);
    double logDetW0 = 0;
    for(int i = 0; i < mDim; ++i) logDetW0 -= 2.0 * log(L[i * mDim + i]);

    double logEvidence0 = TMath::LnGamma( auxNumComponents * mAlpha0 ) - auxNumComponents * TMath::LnGamma(mAlpha0);

    double minusLogB = 0.5 * mNu0 * (logDetW0 + auxDim * log(2) ) + 0.25 * auxDim * (auxDim - 1) * log(3.141592653589793);
    for(int i = 0; i < mDim; ++i) minusLogB += TMath::LnGamma( 0.5 * (mNu0 - (double)i) );

    logEvidence0 += - auxNumComponents * minusLogB;
    
    mLogEvidence = 0;
    
    // Initialize the responsibilities to random variables
    for(int n = 0; n < mNumData; ++n)
    {
    
        double norm = 0;
        
        for(int k = 0; k < mNumComponents; ++k)
        {
            mR[n * mNumComponents + k] = random.Rndm();
            norm += mR[n * mNumComponents + k];
        }
        
        for(int k = 0; k < mNumComponents; ++k) mR[n * mNumComponents + k] /= norm;
        
    }

    // Compute variational updates
    double dLogEvidence = 0;
    
    for(int l = 0; l < mNumMaxIterations; ++l)
    {

        // Store previous lower bound
        double logEvidenceOld = mLogEvidence;
    
        //////////////////////////////////////////////////
        //      Compute the sufficient statistics       //
        //           of the responsibilities            //
        //////////////////////////////////////////////////
    
        for(int k = 0; k < mNumComponents; ++k)
        {
        
            // Local pointers
            double *xBarPtr = xBar + k * mDim;
            double *SPtr = S + k * mDim * mDim;
            double *dataPtr = 0;
        
            // Effective occupancies N_{k} and effective sample mean \bar{x}_{k}
            mN[k] = 0;
            
            for(int i = 0; i < mDim; ++i) xBarPtr[i] = 0;
            
            for(int n = 0; n < mNumData; ++n)
            {
            
                mN[k] += mR[n * mNumComponents + k];
            
                dataPtr = mData + n * mDim;
                
                for(int i = 0; i < mDim; ++i)
                {
                    xBarPtr[i] += mR[n * mNumComponents + k] * dataPtr[i];
                }
            
            }
            
            for(int i = 0; i < mDim; ++i) if(mN[k]) xBarPtr[i] /= mN[k];

            // Effective sample covariance S_{k}
            for(int i = 0; i < mDim; ++i)
            {
                for(int j = 0; j < mDim; ++j)
                {
                    SPtr[i * mDim + j] = 0;
                }
            }
            
            for(int n = 0; n < mNumData; ++n)
            {
            
                dataPtr = mData + n * mDim;
            
                for(int i = 0; i < mDim; ++i)
                {
                    for(int j = 0; j < mDim; ++j)
                    {
                        SPtr[i * mDim + j] += mR[n * mNumComponents + k] * (dataPtr[i] - xBarPtr[i]) * (dataPtr[j] - xBarPtr[j]);
                    }
                }
            
            }
            
            for(int i = 0; i < mDim; ++i)
            {
                for(int j = 0; j < mDim; ++j)
                {
                    if(mN[k]) SPtr[i * mDim + j] /= mN[k];
                }
            }
            
        }
        
        //////////////////////////////////////////////////
        //        Compute variational updates to        //
        //         the mixture model parameters         //
        //////////////////////////////////////////////////
        
        // \alpha_{k}, \beta_{k}, and \nu_{k}
        alphaHat = 0;
        
        for(int k = 0; k < mNumComponents; ++k)
        {
            alpha[k] = mAlpha0 + mN[k];
            alphaHat += alpha[k];
            beta[k] = mBeta0 + mN[k];
            nu[k] = mNu0 + mN[k] + 1;
        }
        
        // Compute m_{k}
        for(int k = 0; k < mNumComponents; ++k)
        {
            for(int i = 0; i < mDim; ++i)
            {
                mM[k * mDim + i] = (mBeta0 * mM0[i] + mN[k] * xBar[k * mDim + i]) / beta[k];
            }
        }
        
        // Compute W_{k}^{-1} and W_{k}
        for(int k = 0; k < mNumComponents; ++k)
        {
        
            double *WInversePtr = Winverse + k * mDim * mDim;
            double *WPtr = W + k * mDim * mDim;
            
            double *xBarPtr = xBar + k * mDim;
            double *SPtr = S + k * mDim * mDim;
        
            double c = mBeta0 * mN[k] / (mBeta0 + mN[k]);
        
            for(int i = 0; i < mDim; ++i)
            {
                for(int j = 0; j < mDim; ++j)
                {
                    WInversePtr[i * mDim + j] = mW0inverse[i * mDim + j] + mN[k] * SPtr[i * mDim + j];
                    WInversePtr[i * mDim + j] += c * (xBarPtr[i] - mM0[i]) * (xBarPtr[j] - mM0[j]);
                }
            }
        
            fInvert(WInversePtr, WPtr, L, LT, mDim);

            // Compute log \tilde{\lambda}_{k}
            //     Note the minus sign! A sum would compute the log determinent of W_{k}^{-1}
            logW[k] = 0;
            for(int i = 0; i < mDim; ++i) logW[k] -= 2.0 * log(L[i * mDim + i]);
        
            logLambdaTwidle[k] = mDim * log(2) + logW[k];
            for(int i = 0; i < mDim; ++i) logLambdaTwidle[k] += fDigamma( 0.5 * (nu[k] - (double)i) );
        
            // Compute log \tilde{\pi}_{k}
            logPiTwidle[k] = fDigamma(alpha[k]) - fDigamma(alphaHat);
            
        }
        
        //////////////////////////////////////////////////
        //     Compute the logEvidence lower bound      //
        //////////////////////////////////////////////////
            
        mLogEvidence = 0;
            
        for(int k = 0; k < mNumComponents; ++k)
        {
            
            double *mPtr = mM + k * mDim;
            double *WPtr = W + k * mDim * mDim;
            
            double *xBarPtr = xBar + k * mDim;
            double *SPtr = S + k * mDim * mDim;
            
            double tempOne = 0;
            double tempTwo = 0;
            
            for(int i = 0; i < mDim; ++i)
            {
                for(int j = 0; j < mDim; ++j)
                {
                    tempOne += (xBarPtr[i] - mPtr[i]) * WPtr[i * mDim + j] * (xBarPtr[j] - mPtr[j]);
                    tempOne += SPtr[i * mDim + j] * WPtr[j * mDim + i];
                    tempTwo += mBeta0 * (mPtr[i] - mM0[i]) * WPtr[i * mDim + j] * (mPtr[j] - mM0[j]);
                    tempTwo += mW0inverse[i * mDim + j] * WPtr[j * mDim + i];
                }
            }
            
            mLogEvidence += - 0.5 * mN[k] * (auxDim / beta[k] + nu[k] * tempOne - logLambdaTwidle[k] + auxDim * log(6.283185307179586));
            mLogEvidence += - 0.5 * nu[k] * tempTwo;
            mLogEvidence += (mAlpha0 - alpha[k] + mN[k]) * logPiTwidle[k] + 0.5 * (mNu0 - auxDim - 1) * logLambdaTwidle[k] + TMath::LnGamma(alpha[k]);
            mLogEvidence += 0.5 * auxDim * ( 1 - mBeta0 / beta[k] + log(mBeta0 / beta[k]) );
            
            double minusLogB = 0.5 * nu[k] * (logW[k] + auxDim * log(2) ) + 0.25 * auxDim * (auxDim - 1) * log(3.141592653589793);
            for(int i = 0; i < mDim; ++i) minusLogB += TMath::LnGamma(0.5 * (nu[k] - (double)i));
            
            double H = minusLogB - 0.5 * (nu[k] - auxDim - 1) * logLambdaTwidle[k] + 0.5 * auxDim * nu[k];
            
            mLogEvidence += H;
              
        }

        // alphaHat contribution
        mLogEvidence += - TMath::LnGamma(alphaHat);

        // r_{nk} entropy contribution
        for(int n = 0; n < mNumData; ++n)
        {
            double *rPtr = mR + n * mNumComponents;
            for(int k = 0; k < mNumComponents; ++k) if(rPtr[k]) mLogEvidence += - rPtr[k] * log(rPtr[k]);
        }
        
        //////////////////////////////////////////////////
        //             Check for convergence            //
        //////////////////////////////////////////////////
        
        dLogEvidence = mLogEvidence - logEvidenceOld;

        if(fabs(dLogEvidence) < mEpsilon) break;
        
        //////////////////////////////////////////////////
        //         Compute variational updates          //
        //           to the responsibilities            //
        //////////////////////////////////////////////////
        
        // Unnormalized responsibilities
        for(int k = 0; k < mNumComponents; ++k)
        {
        
            double *WPtr = W + k * mDim * mDim;            
            double *mPtr = mM + k * mDim;
        
            for(int n = 0; n < mNumData; ++n)
            {
            
                double *rPtr = mR + n * mNumComponents + k;
                double *dataPtr = mData + n * mDim;
                
                *rPtr = 0;
                
                for(int i = 0; i < mDim; ++i)
                {
                    for(int j = 0; j < mDim; ++j)
                    {
                        *rPtr += (dataPtr[i] - mPtr[i]) * WPtr[i * mDim + j] * (dataPtr[j] - mPtr[j]);
                    }
                }
                
                *rPtr *= - 0.5 * nu[k];
                *rPtr += - 0.5 * auxDim / beta[k];
             
                *rPtr = logPiTwidle[k] + 0.5 * logLambdaTwidle[k] + *rPtr;
                      
            }
        
        }
        
        // Normalize
        for(int n = 0; n < mNumData; ++n)
        {
        
            double norm = 0;
            double *rPtr = mR + n * mNumComponents;
            double max = *rPtr;
        
            for(int k = 0; k < mNumComponents; ++k) max = rPtr[k] > max ? rPtr[k] : max;
            for(int k = 0; k < mNumComponents; ++k) 
            {
                rPtr[k] = exp(rPtr[k] - max);
                norm += rPtr[k];
            }

            for(int k = 0; k < mNumComponents; ++k) rPtr[k] /= norm;
            
        }

    
    }
    
    //////////////////////////////////////////////////
    //          Compute and Display Results         //
    //////////////////////////////////////////////////  
    
    mLogEvidence += logEvidence0;
    
    cout << "Variational updates finished with lower bound" << endl;
    cout << "\tL = " << mLogEvidence << endl;
    cout << "\tdL = " << dLogEvidence << endl;
    cout << endl;
    
    for(int k = 0; k < mNumComponents; ++k)
    {
        
        mPi[k] = (mAlpha0 + mN[k]) / ( auxNumComponents * mAlpha0 + (double)mNumData );
        
        for(int i = 0; i < mDim; ++i)
        {
            for(int j = 0; j < mDim; ++j)
            {
                int index = k * mDim * mDim + i * mDim + j;
                mSigma[index] = Winverse[index] / nu[k];
            }
        }

    }
    
}

/// Compute the variational solution to the approximate mixture model posterior
/// where the input samples are weighted
/// \param weight Array of weights ordered the same as the data

void gaussMixer::cluster(double* weight)
{

    //////////////////////////////////////////////////
    //             Instantiate Variables            //
    //////////////////////////////////////////////////

    double alphaHat = 0;

    double xBar[mDim * mNumComponents];
    double S[mDim * mDim * mNumComponents];
    double alpha[mNumComponents];
    double beta[mNumComponents];
    double nu[mNumComponents];
    double W[mDim * mDim * mNumComponents];
    double Winverse[mDim * mDim * mNumComponents];
    double L[mDim * mDim];
    double LT[mDim * mDim];
    double logPiTwidle[mNumComponents];
    double logLambdaTwidle[mNumComponents];
    double logW[mNumComponents];
    
    // Avoid casting
    double auxDim = (double)mDim;
    double auxNumComponents = (double)mNumComponents;
        
    // Fire up the random number generator
    TRandom3 random(0);
        
    //////////////////////////////////////////////////
    //        Compute Variational Distribution      //
    //////////////////////////////////////////////////
    
    // Compute constant terms in the logEvidence lower bound
    
    fCholesky(mW0inverse, L, mDim);
    double logDetW0 = 0;
    for(int i = 0; i < mDim; ++i) logDetW0 -= 2.0 * log(L[i * mDim + i]);

    double logEvidence0 = TMath::LnGamma( auxNumComponents * mAlpha0 ) - auxNumComponents * TMath::LnGamma(mAlpha0);

    double minusLogB = 0.5 * mNu0 * (logDetW0 + auxDim * log(2) ) + 0.25 * auxDim * (auxDim - 1) * log(3.141592653589793);
    for(int i = 0; i < mDim; ++i) minusLogB += TMath::LnGamma( 0.5 * (mNu0 - (double)i) );

    logEvidence0 += - auxNumComponents * minusLogB;
    
    mLogEvidence = 0;
    
    // Initialize the responsibilities to random variables
    for(int n = 0; n < mNumData; ++n)
    {
    
        double norm = 0;
        
        for(int k = 0; k < mNumComponents; ++k)
        {
            mR[n * mNumComponents + k] = random.Rndm();
            norm += mR[n * mNumComponents + k];
        }
        
        for(int k = 0; k < mNumComponents; ++k) mR[n * mNumComponents + k] /= norm;
        
    }

    // Compute variational updates
    double dLogEvidence = 0;
    
    for(int l = 0; l < mNumMaxIterations; ++l)
    {

        // Store previous lower bound
        double logEvidenceOld = mLogEvidence;
    
        //////////////////////////////////////////////////
        //      Compute the sufficient statistics       //
        //           of the responsibilities            //
        //////////////////////////////////////////////////
    
        for(int k = 0; k < mNumComponents; ++k)
        {
        
            // Local pointers
            double *xBarPtr = xBar + k * mDim;
            double *SPtr = S + k * mDim * mDim;
            double *dataPtr = 0;
        
            // Effective occupancies N_{k} and effective sample mean \bar{x}_{k}
            mN[k] = 0;
            
            for(int i = 0; i < mDim; ++i) xBarPtr[i] = 0;
            
            for(int n = 0; n < mNumData; ++n)
            {
            
                mN[k] += weight[n] * mR[n * mNumComponents + k];
            
                dataPtr = mData + n * mDim;
                
                for(int i = 0; i < mDim; ++i)
                {
                    xBarPtr[i] += weight[n] * mR[n * mNumComponents + k] * dataPtr[i];
                }
            
            }
            
            for(int i = 0; i < mDim; ++i) if(mN[k]) xBarPtr[i] /= mN[k];

            // Effective sample covariance S_{k}
            for(int i = 0; i < mDim; ++i)
            {
                for(int j = 0; j < mDim; ++j)
                {
                    SPtr[i * mDim + j] = 0;
                }
            }
            
            for(int n = 0; n < mNumData; ++n)
            {
            
                dataPtr = mData + n * mDim;
            
                for(int i = 0; i < mDim; ++i)
                {
                    for(int j = 0; j < mDim; ++j)
                    {
                        SPtr[i * mDim + j] += weight[n] * mR[n * mNumComponents + k] * (dataPtr[i] - xBarPtr[i]) * (dataPtr[j] - xBarPtr[j]);
                    }
                }
            
            }
            
            for(int i = 0; i < mDim; ++i)
            {
                for(int j = 0; j < mDim; ++j)
                {
                    if(mN[k]) SPtr[i * mDim + j] /= mN[k];
                }
            }
            
        }
        
        //////////////////////////////////////////////////
        //        Compute variational updates to        //
        //         the mixture model parameters         //
        //////////////////////////////////////////////////
        
        // \alpha_{k}, \beta_{k}, and \nu_{k}
        alphaHat = 0;
        
        for(int k = 0; k < mNumComponents; ++k)
        {
            alpha[k] = mAlpha0 + mN[k];
            alphaHat += alpha[k];
            beta[k] = mBeta0 + mN[k];
            nu[k] = mNu0 + mN[k] + 1;
        }
        
        // Compute m_{k}
        for(int k = 0; k < mNumComponents; ++k)
        {
            for(int i = 0; i < mDim; ++i)
            {
                mM[k * mDim + i] = (mBeta0 * mM0[i] + mN[k] * xBar[k * mDim + i]) / beta[k];
            }
        }
        
        // Compute W_{k}^{-1} and W_{k}
        for(int k = 0; k < mNumComponents; ++k)
        {
        
            double *WInversePtr = Winverse + k * mDim * mDim;
            double *WPtr = W + k * mDim * mDim;
            
            double *xBarPtr = xBar + k * mDim;
            double *SPtr = S + k * mDim * mDim;
        
            double c = mBeta0 * mN[k] / (mBeta0 + mN[k]);
        
            for(int i = 0; i < mDim; ++i)
            {
                for(int j = 0; j < mDim; ++j)
                {
                    WInversePtr[i * mDim + j] = mW0inverse[i * mDim + j] + mN[k] * SPtr[i * mDim + j];
                    WInversePtr[i * mDim + j] += c * (xBarPtr[i] - mM0[i]) * (xBarPtr[j] - mM0[j]);
                }
            }
        
            fInvert(WInversePtr, WPtr, L, LT, mDim);

            // Compute log \tilde{\lambda}_{k}
            //     Note the minus sign! A sum would compute the log determinent of W_{k}^{-1}
            logW[k] = 0;
            for(int i = 0; i < mDim; ++i) logW[k] -= 2.0 * log(L[i * mDim + i]);
        
            logLambdaTwidle[k] = mDim * log(2) + logW[k];
            for(int i = 0; i < mDim; ++i) logLambdaTwidle[k] += fDigamma( 0.5 * (nu[k] - (double)i) );
        
            // Compute log \tilde{\pi}_{k}
            logPiTwidle[k] = fDigamma(alpha[k]) - fDigamma(alphaHat);
            
        }
        
        //////////////////////////////////////////////////
        //     Compute the logEvidence lower bound      //
        //////////////////////////////////////////////////
            
        mLogEvidence = 0;
            
        for(int k = 0; k < mNumComponents; ++k)
        {
            
            double *mPtr = mM + k * mDim;
            double *WPtr = W + k * mDim * mDim;
            
            double *xBarPtr = xBar + k * mDim;
            double *SPtr = S + k * mDim * mDim;
            
            double tempOne = 0;
            double tempTwo = 0;
            
            for(int i = 0; i < mDim; ++i)
            {
                for(int j = 0; j < mDim; ++j)
                {
                    tempOne += (xBarPtr[i] - mPtr[i]) * WPtr[i * mDim + j] * (xBarPtr[j] - mPtr[j]);
                    tempOne += SPtr[i * mDim + j] * WPtr[j * mDim + i];
                    tempTwo += mBeta0 * (mPtr[i] - mM0[i]) * WPtr[i * mDim + j] * (mPtr[j] - mM0[j]);
                    tempTwo += mW0inverse[i * mDim + j] * WPtr[j * mDim + i];
                }
            }
            
            mLogEvidence += - 0.5 * mN[k] * (auxDim / beta[k] + nu[k] * tempOne - logLambdaTwidle[k] + auxDim * log(6.283185307179586));
            mLogEvidence += - 0.5 * nu[k] * tempTwo;
            mLogEvidence += (mAlpha0 - alpha[k] + mN[k]) * logPiTwidle[k] + 0.5 * (mNu0 - auxDim - 1) * logLambdaTwidle[k] + TMath::LnGamma(alpha[k]);
            mLogEvidence += 0.5 * auxDim * ( 1 - mBeta0 / beta[k] + log(mBeta0 / beta[k]) );
            
            double minusLogB = 0.5 * nu[k] * (logW[k] + auxDim * log(2) ) + 0.25 * auxDim * (auxDim - 1) * log(3.141592653589793);
            for(int i = 0; i < mDim; ++i) minusLogB += TMath::LnGamma(0.5 * (nu[k] - (double)i));
            
            double H = minusLogB - 0.5 * (nu[k] - auxDim - 1) * logLambdaTwidle[k] + 0.5 * auxDim * nu[k];
            
            mLogEvidence += H;
              
        }

        // alphaHat contribution
        mLogEvidence += - TMath::LnGamma(alphaHat);

        // r_{nk} entropy contribution
        for(int n = 0; n < mNumData; ++n)
        {
            double *rPtr = mR + n * mNumComponents;
            for(int k = 0; k < mNumComponents; ++k) if(rPtr[k]) mLogEvidence += - weight[n] * rPtr[k] * log(rPtr[k]);
        }
        
        //////////////////////////////////////////////////
        //             Check for convergence            //
        //////////////////////////////////////////////////
        
        dLogEvidence = mLogEvidence - logEvidenceOld;

        if(fabs(dLogEvidence) < mEpsilon) break;
        
        //////////////////////////////////////////////////
        //         Compute variational updates          //
        //           to the responsibilities            //
        //////////////////////////////////////////////////
        
        // Unnormalized responsibilities
        for(int k = 0; k < mNumComponents; ++k)
        {
        
            double *WPtr = W + k * mDim * mDim;            
            double *mPtr = mM + k * mDim;
        
            for(int n = 0; n < mNumData; ++n)
            {
            
                double *rPtr = mR + n * mNumComponents + k;
                double *dataPtr = mData + n * mDim;
                
                *rPtr = 0;
                
                for(int i = 0; i < mDim; ++i)
                {
                    for(int j = 0; j < mDim; ++j)
                    {
                        *rPtr += (dataPtr[i] - mPtr[i]) * WPtr[i * mDim + j] * (dataPtr[j] - mPtr[j]);
                    }
                }
                
                *rPtr *= - 0.5 * nu[k];
                *rPtr += - 0.5 * auxDim / beta[k];
             
                *rPtr = logPiTwidle[k] + 0.5 * logLambdaTwidle[k] + *rPtr;
                      
            }
        
        }
        
        // Normalize
        for(int n = 0; n < mNumData; ++n)
        {
        
            double norm = 0;
            double *rPtr = mR + n * mNumComponents;
            double max = *rPtr;
        
            for(int k = 0; k < mNumComponents; ++k) max = rPtr[k] > max ? rPtr[k] : max;
            for(int k = 0; k < mNumComponents; ++k) 
            {
                rPtr[k] = exp(rPtr[k] - max);
                norm += rPtr[k];
            }

            for(int k = 0; k < mNumComponents; ++k) rPtr[k] /= norm;
            
        }

    
    }
    
    //////////////////////////////////////////////////
    //          Compute and Display Results         //
    //////////////////////////////////////////////////  
    
    mLogEvidence += logEvidence0;
    
    cout << "Variational updates finished with lower bound" << endl;
    cout << "\tL = " << mLogEvidence << endl;
    cout << "\tdL = " << dLogEvidence << endl;
    cout << endl;
    
    double sumN = 0;
    for(int k = 0; k < mNumComponents; ++k) sumN += mN[k];
    
    for(int k = 0; k < mNumComponents; ++k)
    {
        
        mPi[k] = (mAlpha0 + mN[k]) / ( auxNumComponents * mAlpha0 + sumN );
        
        for(int i = 0; i < mDim; ++i)
        {
            for(int j = 0; j < mDim; ++j)
            {
                int index = k * mDim * mDim + i * mDim + j;
                mSigma[index] = Winverse[index] / nu[k];
            }
        }

    }
    
}

///  Compute digamma(x) = (dGamma(x) / dx) / Gamma(x)
///
/// Original source from http://people.sc.fsu.edu/~burkardt/cpp_src/asa103/asa103.C
/// See Jose Bernardo, Algorithm AS 103: Psi ( Digamma ) Function,
/// Applied Statistics, Volume 25, Number 3, 1976, pages 315-317.
///
/// Modified by Michael Betancourt, November 20 2009
///
/// \param x Input value
/// \return digamma(x)

double gaussMixer::fDigamma(double x)
{

    // Initialize variables
    double y = x;
    double value = 0.0;
    
    double s = 0.00001;
    double d1 = -0.5772156649;
    
    double c = 8.5;
    double r;
    double s3 = 0.08333333333;
    double s4 = 0.0083333333333;
    double s5 = 0.003968253968;

    // Use approximation if argument <= s
    if(y <= s)
    {
        value = d1 - 1.0 / y;
        return value;
    }

    // Reduce to digamma(x + n) where (x + n) >= c
    while(y < c)
    {
        value -= 1.0 / y;
        y += 1.0;
    }

    // Use Stirling's (actually de Moivre's) expansion if argument > c.
    r = 1.0 / y;
    value += log(y) - 0.5 * r;
    r *= r;
    value -= r * ( s3 - r * (s4 - r * s5) );

    return value;

}