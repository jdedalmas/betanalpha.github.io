// C++ Standard Library
#include <iostream>
#include "math.h"

// ROOT Libraries
#include "TCanvas.h"
#include "TGraph.h"
#include "TH1.h"
#include "TF1.h"
#include "TLegend.h"
#include "TMath.h"
#include "TRandom3.h"
#include "TString.h"
#include "TStyle.h"

// Local Libraries
#include "bayesHistComp.h"

using namespace std;

//ClassImp(bayesHistComp);

/// Overloaded constructor allowing customization
/// \param nPiPoints Number of points in the discretization of pi
/// \see ~bayesHistComp()

bayesHistComp::bayesHistComp(int nPiPoints)
{

    mGoVerbose = false;
    mSkipEmptyBins = false;

    mNumPiPoints = nPiPoints;
    mPi = new double[mNumPiPoints];
    mP = new double[mNumPiPoints];
   
    mPiMAP = 0;
    mCredibility = 0;
    mModelOdds = 0;
    
    mNumBins = 0;
    mHistOne = 0;
    mHistOneN = 0;
    mHistTwo = 0;
    mHistTwoN = 0;

    for(int i = 0; i < mNumPiPoints; ++i)
    {
        mPi[i] = (double)i / (double)(mNumPiPoints - 1);
    }

}

/// Destructor
/// \see bayesHistComp()
/// \see bayesHistComp(int nPiPoints, double credibility)

bayesHistComp::~bayesHistComp()
{
    delete[] mPi;
    delete[] mP;
}

/// Compute the probability of same source and 
/// mixture model posterior for the two input histograms
/// \param dataOne Pointer to the first data histogram
/// \param dataTwo Pointer to the second data histogram
/// \see fCompareDataData(TH1 *dataOne, TH1 *dataTwo)

void bayesHistComp::compareDataData(TH1 *dataOne, TH1 *dataTwo)
{
     fCompareDataData(dataOne, dataTwo);
}

/// Compute the probability of same source and 
/// mixture model posterior for the two input histograms,
/// drawing the mixture posterior onto an external TGraph.
/// \param dataOne Pointer to the first data histogram
/// \param dataTwo Pointer to the second data histogram
/// \param posterior Pointer to external TGraph onto which the posterior will be drawn
/// \see fCompareDataData(TH1 *dataOne, TH1 *dataTwo)

void bayesHistComp::compareDataData(TH1 *dataOne, TH1 *dataTwo, TGraph *posterior)
{

    // Compare the two histograms
    fCompareDataData(dataOne, dataTwo);
    
    // Draw mixture posterior
    fDraw(posterior, "");
    
    return;

}

/// Compute the probability of same source and 
/// mixture model posterior for the two input histograms,
/// summarizing the results with an output image.
/// \param dataOne Pointer to the first data histogram
/// \param dataTwo Pointer to the second data histogram
/// \param imageName File name of saved image
/// \see fCompareDataData(TH1 *dataOne, TH1 *dataTwo)

void bayesHistComp::compareDataData(TH1 *dataOne, TH1 *dataTwo, TString imageName)
{
    
    // Compare the two histograms
    fCompareDataData(dataOne, dataTwo);

    // Draw summary results
    fDraw(0, imageName);
    
    return;
    
}

/// Compute the probability of same source and 
/// mixture model posterior for the two input histograms
/// \param dataOne Pointer to the first data histogram
/// \param dataTwo Pointer to the second data histogram

void bayesHistComp::fCompareDataData(TH1 *dataOne, TH1 *dataTwo)
{

    // Reset
    mPiMAP = 0;
    mCredibility = 0;
    mModelOdds = 0;

    // First check for compatibility
    mNumBins = dataOne->GetNbinsX() * dataOne->GetNbinsY() * dataOne->GetNbinsZ();
    
    if(dataTwo->GetNbinsX() * dataTwo->GetNbinsY() * dataTwo->GetNbinsZ() != mNumBins)
    {
        cout << "bayesHistComp::fCompareDataData() - Input histograms"
             << " are not compatible! Check for consistent binning." << endl;
        return;
    }
    
    // Check status of the histograms
    if(dataOne->GetSumw2N())
    {
    
     	bool noWeight = true;

        for(int i = 0; i < mNumBins; ++i)
        {
            noWeight &= fabs(dataOne->GetBinContent(i + 1) - dataOne->GetBinContent(i + 1) * dataOne->GetBinContent(i + 1)) < 1e-12;
        }

        if(noWeight)
        {
            cout << "bayesHistComp::fCompareDataData() - First input data histogram "
                 << "was filled with non-unity weights, a possible mistake?" << endl;
        }

    }
    
    if(dataTwo->GetSumw2N())
    {
       
        bool noWeight = true;
    
        for(int i = 0; i < mNumBins; ++i)
        {
            noWeight &= fabs(dataTwo->GetBinContent(i + 1) - dataTwo->GetBinContent(i + 1) * dataTwo->GetBinContent(i + 1)) < 1e-12;
        }
    
        if(noWeight)
        {
            cout << "bayesHistComp::fCompareDataData() - Second input data histogram "
                 << "was filled with non-unity weights, a possible mistake?" << endl;
        }

    }
    
    // Define model
    histModel model = kDataData;
    
    mHistOne = dataOne;
    mHistTwo = dataTwo;
    
    // Prepare for inference
    for(double *pPtr = mP; pPtr != mP + mNumPiPoints; ++pPtr) *pPtr = 0;
    
    // Compare the two histograms
    mModelOdds = fComputeMixturePosterior(model);
    
    // Compute credibility interval
    fCalcCredibility();
    
    if(mGoVerbose) cout << "bayesHistComp::fCompareDataData() -  pi > 1/2 with probability " << mCredibility << endl;
    if(mGoVerbose) cout << "bayesHistComp::fCompareDataData() - p(Same) = " << mModelOdds << endl;
    
    return;

}

/// Compute the probability of same source and 
/// mixture model posterior for the two input histograms
/// \param data Pointer to the data histogram
/// \param simuWeight Pointer to the simulation histogram filled with weights
/// \param simuNoWeight Pointer to the simulation histogram filled without weights
/// \see fCompareDataSimu(TH1 *data, TH1 *simuWeight, TH1 *simuNoWeight)

void bayesHistComp::compareDataSimu(TH1 *data, TH1 *simuWeight, TH1 *simuNoWeight)
{
    fCompareDataSimu(data, simuWeight, simuNoWeight);
}

/// Compute the probability of same source and 
/// mixture model posterior for the two input histograms,
/// drawing the mixture posterior onto an external TGraph.
/// \param data Pointer to the data histogram
/// \param simuWeight Pointer to the simulation histogram filled with weights
/// \param simuNoWeight Pointer to the simulation histogram filled without weights
/// \param posterior Pointer to external TGraph onto which the posterior will be drawn
/// \see fCompareDataSimu(TH1 *data, TH1 *simuWeight, TH1 *simuNoWeight)

void bayesHistComp::compareDataSimu(TH1 *data, TH1 *simuWeight, TH1 *simuNoWeight, TGraph *posterior)
{

    // Compare the two histograms
    fCompareDataSimu(data, simuWeight, simuNoWeight);
    
    // Draw mixture posterior
    fDraw(posterior, "");
    
    return;

}

/// Compute the probability of same source and 
/// mixture model posterior for the two input histograms,
/// summarizing the results with an output image.
/// \param data Pointer to the data histogram
/// \param simuWeight Pointer to the simulation histogram filled with weights
/// \param simuNoWeight Pointer to the simulation histogram filled without weights
/// \param imageName File name of saved image
/// \see fCompareDataSimu(TH1 *data, TH1 *simuWeight, TH1 *simuNoWeight)

void bayesHistComp::compareDataSimu(TH1 *data, TH1 *simuWeight, TH1 *simuNoWeight, TString imageName)
{
    
    // Compare the two histograms
    fCompareDataSimu(data, simuWeight, simuNoWeight);

    // Draw summary results
    fDraw(0, imageName);
    
    return;
    
}

/// Compute the probability of same source  
/// and mixture model posterior for a data histogram  
/// and one generated from a importance sampled simulation
/// \param data Pointer to the data histogram
/// \param simuWeight Pointer to the simulation histogram filled with weights
/// \param simuNoWeight Pointer to the simulation histogram filled without weights

void bayesHistComp::fCompareDataSimu(TH1 *data, TH1 *simuWeight, TH1 *simuNoWeight)
{
        
    // Reset
    mPiMAP = 0;
    mCredibility = 0;
    mModelOdds = 0;
    
    // First check for compatibility
    mNumBins = data->GetNbinsX() * data->GetNbinsY() * data->GetNbinsZ();
    int nSimuWeight = simuWeight->GetNbinsX() * simuWeight->GetNbinsY() * simuWeight->GetNbinsZ();
    int nSimuNoWeight = simuNoWeight->GetNbinsX() * simuNoWeight->GetNbinsY() * simuNoWeight->GetNbinsZ();
    
    if(nSimuWeight != mNumBins || nSimuNoWeight != mNumBins)
    {
        cout << "bayesHistComp::fCompareDataSimu() - Input histograms"
             << " are not compatible! Check for consistent binning." << endl;
        return;
    }
    
    // Check status of the histograms
    if(data->GetSumw2N())
    {

     	bool noWeight = true;

        for(int i = 0; i < mNumBins; ++i)
        {
            noWeight &= fabs(data->GetBinContent(i + 1) - data->GetBinContent(i + 1) * data->GetBinContent(i + 1)) < 1e-12;
        }

        if(noWeight)
        {
            cout << "bayesHistComp::fCompareDataSimu() - Input data histogram "
                 << "was filled with non-unity weights, a possible mistake?" << endl;
        }

    }
    
    if(!simuWeight->GetSumw2N())
    {
        cout << "bayesHistComp::fCompareDataSimu() - Input weighted simulation histogram "
             << "was not filled with weights, a possible mistake?" << endl;
    }
    
    if(simuNoWeight->GetSumw2N())
    {

     	bool noWeight = true;

        for(int i = 0; i < mNumBins; ++i)
        {
            noWeight &= fabs(simuNoWeight->GetBinContent(i + 1) - simuNoWeight->GetBinContent(i + 1) * simuNoWeight->GetBinContent(i + 1)) < 1e-12;
        }

        if(noWeight)
        {
            cout << "bayesHistComp::fCompareDataSimu() - Input unweighted simulation histogram "
                 << "was filled with non-unity weights, a possible mistake?" << endl;
        }

    }
    
    // Define model
    histModel model = kDataSimu;
    
    mHistOne = data;
    mHistTwo = simuWeight;
    mHistTwoN = simuNoWeight;
    
    // Prepare for inference
    for(double *pPtr = mP; pPtr != mP + mNumPiPoints; ++pPtr) *pPtr = 0;
    
    // Compare the two histograms
    mModelOdds = fComputeMixturePosterior(model);
    
    // Compute credibility interval
    fCalcCredibility();
    
    if(mGoVerbose) cout << "bayesHistComp::fCompareDataSimu() -  pi > 1/2 with probability " << mCredibility << endl;
    if(mGoVerbose) cout << "bayesHistComp::fCompareDataSimu() - p(Same) = " << mModelOdds << endl;
    
    return;

}

/// Compute the probability of same source and 
/// mixture model posterior for the two input histograms
/// \param simuOneW Pointer to the first simulation histogram filled with weights
/// \param simuOneNoW Pointer to the first simulation histogram filled without weights
/// \param simuTwoW Pointer to the second simulation histogram filled with weights
/// \param simuTwoNoW Pointer to the second simulation histogram filled without weights
/// \see fCompareSimuSimu(TH1 *simuOneW, TH1 *simuOneNoW, TH1 *simuTwoW, TH1 *simuTwoNoW)

void bayesHistComp::compareSimuSimu(TH1 *simuOneW, TH1 *simuOneNoW, TH1 *simuTwoW, TH1 *simuTwoNoW)
{
    fCompareSimuSimu(simuOneW, simuOneNoW, simuTwoW, simuTwoNoW);
}

/// Compute the probability of same source and 
/// mixture model posterior for the two input histograms,
/// drawing the mixture posterior onto an external TGraph.
/// \param simuOneW Pointer to the first simulation histogram filled with weights
/// \param simuOneNoW Pointer to the first simulation histogram filled without weights
/// \param simuTwoW Pointer to the second simulation histogram filled with weights
/// \param simuTwoNoW Pointer to the second simulation histogram filled without weights
/// \param posterior Pointer to external TGraph onto which the posterior will be drawn
/// \see fCompareSimuSimu(TH1 *simuOneW, TH1 *simuOneNoW, TH1 *simuTwoW, TH1 *simuTwoNoW)

void bayesHistComp::compareSimuSimu(TH1 *simuOneW, TH1 *simuOneNoW, TH1 *simuTwoW, TH1 *simuTwoNoW, TGraph *posterior)
{

    // Compare the two histograms
    fCompareSimuSimu(simuOneW, simuOneNoW, simuTwoW,simuTwoNoW);
    
    // Draw mixture posterior
    fDraw(posterior, "");
    
    return;

}

/// Compute the probability of same source and 
/// mixture model posterior for the two input histograms,
/// summarizing the results with an output image.
/// \param simuOneW Pointer to the first simulation histogram filled with weights
/// \param simuOneNoW Pointer to the first simulation histogram filled without weights
/// \param simuTwoW Pointer to the second simulation histogram filled with weights
/// \param simuTwoNoW Pointer to the second simulation histogram filled without weights
/// \param imageName File name of saved image
/// \see fCompareSimuSimu(TH1 *simuOneW, TH1 *simuOneNoW, TH1 *simuTwoW, TH1 *simuTwoNoW)

void bayesHistComp::compareSimuSimu(TH1 *simuOneW, TH1 *simuOneNoW, TH1 *simuTwoW, TH1 *simuTwoNoW, TString imageName)
{
    
    // Compare the two histograms
    fCompareSimuSimu(simuOneW, simuOneNoW, simuTwoW, simuTwoNoW);

    // Draw summary results
    fDraw(0, imageName);
    
    return;
    
}

/// Compute the probability of same source  
/// and mixture model posterior for two input histograms  
/// generated from importance sampled simulation
/// \param simuOneW Pointer to the first simulation histogram filled with weights
/// \param simuOneNoW Pointer to the first simulation histogram filled without weights
/// \param simuTwoW Pointer to the second simulation histogram filled with weights
/// \param simuTwoNoW Pointer to the second simulation histogram filled without weights

void bayesHistComp::fCompareSimuSimu(TH1 *simuOneW, TH1 *simuOneNoW, TH1 *simuTwoW, TH1 *simuTwoNoW)
{

    // Reset
    mPiMAP = 0;
    mCredibility = 0;
    mModelOdds = 0;

    // First check for compatibility
    mNumBins = simuOneW->GetNbinsX() * simuOneW->GetNbinsY() * simuOneW->GetNbinsZ();
    int nSimuOneNoW = simuOneNoW->GetNbinsX() * simuOneNoW->GetNbinsY() * simuOneNoW->GetNbinsZ();
    int nSimuTwoW = simuTwoW->GetNbinsX() * simuTwoW->GetNbinsY() * simuTwoW->GetNbinsZ();
    int nSimuTwoNoW = simuTwoNoW->GetNbinsX() * simuTwoNoW->GetNbinsY() * simuTwoNoW->GetNbinsZ();
    
    bool incompatible = nSimuOneNoW != mNumBins;
    incompatible |= nSimuTwoW != mNumBins;
    incompatible |= nSimuTwoNoW != mNumBins;
    
    if(incompatible)
    {
        cout << "bayesHistComp::fCompareSimuSimu() - Input histograms"
             << " are not compatible! Check for consistent binning." << endl;
        return;
    }
    
    // Check status of the histograms
    if(!simuOneW->GetSumw2N())
    {
        cout << "bayesHistComp::fCompareSimuSimu() - First input weighted simulation histogram "
             << "was not filled with weights, a possible mistake?" << endl;
    }
    
    if(simuOneNoW->GetSumw2N())
    {
            
        bool noWeight = true;
         
        for(int i = 0; i < mNumBins; ++i)
        {
            noWeight &= fabs(simuOneNoW->GetBinContent(i + 1) - simuOneNoW->GetBinContent(i + 1) * simuOneNoW->GetBinContent(i + 1)) < 1e-12;
        }
    
    	if(noWeight)
        {
            cout << "bayesHistComp::fCompareSimuSimu() - First input unweighted simulation histogram "
                 << "was filled with non-unity weights, a possible mistake?" << endl;
        }

    }

    if(!simuTwoW->GetSumw2N())
    {
        cout << "bayesHistComp::fCompareSimuSimu() - Second input weighted simulation histogram "
             << "was not filled with weights, a possible mistake?" << endl;
    }
    
    if(simuTwoNoW->GetSumw2N())
    {
    
        bool noWeight = true;
    
    	for(int i = 0; i < mNumBins; ++i)
        {
            noWeight &= fabs(simuTwoNoW->GetBinContent(i + 1) - simuTwoNoW->GetBinContent(i + 1) * simuTwoNoW->GetBinContent(i + 1)) < 1e-12;
 	}

        if(noWeight)
        {
            cout << "bayesHistComp::fCompareSimuSimu() - Second input unweighted simulation histogram "
                 << "was filled with non-unity weights, a possible mistake?" << endl;
        }

    }
    
    // Define model
    histModel model = kSimuSimu;
    
    mHistOne = simuOneW;
    mHistOneN = simuOneNoW;
    mHistTwo = simuTwoW;
    mHistTwoN = simuTwoNoW;
    
    // Prepare for inference
    for(double *pPtr = mP; pPtr != mP + mNumPiPoints; ++pPtr) *pPtr = 0;
    
    // Compare the two histograms
    mModelOdds = fComputeMixturePosterior(model);
    
    // Compute credibility interval
    fCalcCredibility();
    
    if(mGoVerbose) cout << "bayesHistComp::fCompareSimuSimu() -  pi > 1/2 with probability " << mCredibility << endl;
    if(mGoVerbose) cout << "bayesHistComp::fCompareSimuSimu() - p(Same) = " << mModelOdds << endl;
    
    return;

}

/// Compute the log evidence ratio and mixture model posterior
/// for the two input histograms and given model
/// \param model Enumeration of possible data/simulation models
/// \return Probability of the same model, Z(same) / [ Z(same) + Z(!same) ]

double bayesHistComp::fComputeMixturePosterior(histModel model)
{

    double s = 0;
    double nS = 0;
    double r = 0;
    double nR = 0;
    
    double logEvidenceRatio = 0;
    double *piPtr = mPi;
    
    double pSame = 0;
    double logOdds = 1;
    double localOdds = 0;

    // Loop over histogram bins
    for(int i = 0; i < mNumBins; ++i)
    {
    
        s = mHistOne->GetBinContent(i + 1);
        r = mHistTwo->GetBinContent(i + 1);
    
        if(!s && !r && mSkipEmptyBins) continue;
    
        switch(model)
        {
        
            case kDataData:
            
                logEvidenceRatio = fComputeLogEvidenceRatio(s, r);
                    
                break;
                
            case kDataSimu:

                nR = mHistTwoN->GetBinContent(i + 1);
                
                logEvidenceRatio = fComputeLogEvidenceRatio(s, r, nR);
                
                break;
                
            case kSimuSimu:

                nS = mHistOneN->GetBinContent(i + 1);
                nR = mHistTwoN->GetBinContent(i + 1);
                
                logEvidenceRatio = fComputeLogEvidenceRatio(s, nS, r, nR);
                
                break;
                                
        }
       
    	if(s < 0 || r < 0)
        {
            cout << "bayesHistComp::fComputeMixturePosterior() - Skipping bin " << i + 1
                 << " due to negative bin contents..." << endl;
            continue;
        }

        if(nS < 0 || nR < 0)
        {
            cout << "bayesHistComp::fComputeMixturePosterior() - Skipping bin " << i + 1
    	         << " due to negative weight multplicities..." << endl;
            continue;
        }

          
        // Loop over pi points, incrementing the log mixture posterior
        if(logEvidenceRatio > 0)
        { 
            
            localOdds = exp(-logEvidenceRatio);
            
            piPtr = mPi;
            for(double *pPtr = mP; pPtr != mP + mNumPiPoints; ++pPtr, ++piPtr)
            {
                *pPtr += log( localOdds * (*piPtr) + (1 - (*piPtr)) );
            }
            
        }
        else
        {
        
            localOdds = exp(logEvidenceRatio);
        
            piPtr = mPi;
            for(double *pPtr = mP; pPtr != mP + mNumPiPoints; ++pPtr, ++piPtr)
            {
                *pPtr += log( (*piPtr) + (1 - (*piPtr) ) * localOdds);
            }
            
        }
        
        logOdds += logEvidenceRatio;
            
    }
    
    // Subtract away maximum log probability before exponentiating
    double maxLogP = *mP;
    
    for(double* pPtr = mP + 1; pPtr != mP + mNumPiPoints; ++pPtr)
    {
        if(*pPtr > maxLogP) maxLogP = *pPtr;
    }
    
    // Exponentiate and normalize
    double norm = 0;
    double dPi = mPi[1] - mPi[0];
    
    for(double* pPtr = mP; pPtr != mP + mNumPiPoints; ++pPtr)
    {
        *pPtr = exp(*pPtr - maxLogP);
        norm += (*pPtr) * dPi;
    }
    
    for(double* pPtr = mP; pPtr != mP + mNumPiPoints; ++pPtr) *pPtr /= norm;
    
    // Compute probability of the same model
    if(logOdds > 0)
    {
        pSame = exp(-logOdds);
        pSame = pSame / (pSame + 1);
    }
    else
    {
        pSame = exp(logOdds);
        pSame = 1.0 / (1.0 + pSame);
    }

    return pSame;

}

/// Compute the log ratio of the evidences for the 
/// two distribution models, log [ Z(!same) / Z(same) ]
/// The prior support is dynamically adjusted to the smallest
/// range that doesn't clip the posterior normalization,
/// implicitly performing a crude model selection.  In order
/// to improve the accuracy of the numerical integrations, the
/// ranges are also dynamically selected given Gaussian
/// approximations to the distributions.
/// \param m Counts in the first histogram
/// \param n Counts in the second histogram

double bayesHistComp::fComputeLogEvidenceRatio(double m, double n)
{

    //////////////////////////////////////////////////
    //      Dynamically calculate prior support     //
    //////////////////////////////////////////////////
    
    // First Poisson Model
    double mAlpha = 0;
    double mBeta = 0;
    
    if(m < 10)
    {
        mAlpha = 0;
        mBeta = 45;
    }
    else
    {
        mAlpha = m - 11.0 * sqrt(m);
        mAlpha = mAlpha > 0 ? mAlpha : 0;
        mBeta = m + 11.0 * sqrt(m);
    }

    // Second Poisson Model
    double nAlpha = 0;
    double nBeta = 0;
    
    if(n < 10)
    {
        nAlpha = 0;
        nBeta = 45;
    }
    else
    {
        nAlpha = n - 11.0 * sqrt(n);
        nAlpha = nAlpha > 0 ? nAlpha : 0;
        nBeta = n + 11.0 * sqrt(n);
    }
    
    // Poisson-Poisson Model
    double mnAlpha = 0;
    double mnBeta = 0;
    
    if(m == 0)
    {
        mnAlpha = nAlpha;
        mnBeta = 0.5 * nBeta;
    }
    else if(n == 0)
    {
        mnAlpha = mAlpha;
        mnBeta = 0.5 * mBeta;
    }
    else
    {
        mnAlpha = 0.5 * (m + n) - 10.0 * sqrt(0.25 * (m + n));
        mnAlpha = mnAlpha > 0 ? mnAlpha : 0;
        mnBeta = 0.5 * (m + n) + 10.0 * sqrt(0.25 * (m + n));
    }
    
    //double alpha = mAlpha < nAlpha ? mAlpha : nAlpha;
    //double beta = mBeta > nBeta ? mBeta : nBeta;
    
    //////////////////////////////////////////////////
    //           Compute evidence ratio             //
    //////////////////////////////////////////////////
    
    double logZratio = (n + m + 1);
    logZratio *= TMath::Gamma(m + 1, mBeta) - TMath::Gamma(m + 1, mAlpha);
    logZratio *= TMath::Gamma(n + 1, nBeta) - TMath::Gamma(n + 1, nAlpha);
    logZratio /= TMath::Gamma(m + n + 1, 2.0 * mnBeta) - TMath::Gamma(m + n + 1, 2.0 * mnAlpha);
    logZratio = log(logZratio) + TMath::LnGamma(m + 1) + TMath::LnGamma(n + 1) - TMath::LnGamma(n + m + 2);
    logZratio += - log( (mBeta - mAlpha) * (nBeta - nAlpha) / (mnBeta - mnAlpha) ) + (n + m + 1) * log(2);
    
    return logZratio;

    //return TMath::LnGamma(m) + TMath::LnGamma(n) - TMath::LnGamma(m + n) + (m + n) * log(2);

}

/// Compute the log ratio of the evidences for the 
/// two distribution models, log [ Z(!same) / Z(same) ]
/// The prior support is dynamically adjusted to the smallest
/// range that doesn't clip the posterior normalization,
/// implicitly performing a crude model selection.  In order
/// to improve the accuracy of the numerical integrations, the
/// ranges are also dynamically selected given Gaussian
/// approximations to the distributions.
/// \param m Counts in the first histogram
/// \param s Summed weights of the second histogram
/// \param nS Total number of weighted events in the second histogram

double bayesHistComp::fComputeLogEvidenceRatio(double m, double s, double nS)
{

    double sBar = nS == 0 ? 1 : s / nS;

    //////////////////////////////////////////////////
    //      Dynamically calculate prior support     //
    //////////////////////////////////////////////////
    
    // Poisson Model
    double mAlpha = 0;
    double mBeta = 0;
    
    if(m < 10)
    {
        mAlpha = 0;
        mBeta = 45;
    }
    else
    {
        mAlpha = m - 11.0 * sqrt(m);
        mAlpha = mAlpha > 0 ? mAlpha : 0;
        mBeta = m + 11.0 * sqrt(m);
    }
    
    // Gaussian Model
    double sMean = 0;
    double sSigma = 0;
    double sAlpha = 0;
    double sBeta = 0;
    
    if(s == 0)
    {
     	sSigma = 1;
        sAlpha = 0;
        sBeta = 50;
    }
    else
    {
    
    	sMean = 0.5 * (sqrt(sBar * sBar + 4 * s * s) - sBar);
        sSigma = sqrt(2.0 * (sBar * sMean * sMean * sMean) / (2 * s * s - sBar * sMean));
    
        sAlpha = sMean - 11.0 * sSigma;
        sAlpha = sAlpha > 0 ? sAlpha : 0;
        sBeta = sMean + 11.0 * sSigma;

    }

    // Poisson-Gaussian Model
    double msMean = 0;
    double msSigma = 0;
    double msAlpha = 0;
    double msBeta = 0;

    if(m == 0)
    {
     	msSigma = sSigma;
        msAlpha = sAlpha;
        msBeta = sBeta;
    }
    else if(s == 0)
    {
     	msSigma = 0.67 * sqrt(m);
        msAlpha = 0.67 * mAlpha;
        msBeta = 0.67 * mBeta;
    }
    else
    {

     	double a = 2.0 * sBar + 1;
        double b = (m - 0.5) * sBar;

        msMean = (sqrt( b * b + a * s * s) + b) / a;
        msSigma = sqrt( (sBar * msMean * msMean * msMean) / (b * msMean + s * s) );

        msAlpha = msMean - 11.0 * msSigma;
        msAlpha = msAlpha > 0 ? msAlpha : 0;
        msBeta = msMean + 11.0 * msSigma;
    
    }
        
    //double alpha = mAlpha < sAlpha ? mAlpha : sAlpha;
    //double beta = mBeta > sBeta ? mBeta : sBeta;
    
    //////////////////////////////////////////////////
    //           Compute evidence ratio             //
    //////////////////////////////////////////////////
    
    double Z1 = TMath::Gamma(m + 1, mBeta) - TMath::Gamma(m + 1, mAlpha);
    
    // Define and integrate the Gaussian integral, expanding the prior support if needed
    TF1 f("f", "exp( - 0.5 * (x - [0]) * (x - [0]) / ([1] * x) ) / sqrt(x)", sAlpha, sBeta);
    f.SetParameter(0, s);
    f.SetParameter(1, sBar);
    
    while(f.Eval(sBeta) > 1e-12) sBeta += sSigma;
    
    double Z2 = f.Integral(sAlpha, sBeta);
    
    // Define and integrate the Poisson-Gaussian integral, expanding the prior support if needed
    TF1 g("g", "exp( ([0] - 0.5) * log(x) - x - 0.5 * (x - [1]) * (x - [1]) / ([2] * x) - TMath::LnGamma([0] + 1) )", msAlpha, msBeta);
    g.SetParameter(0, m);
    g.SetParameter(1, s);
    g.SetParameter(2, sBar);
    
    while(g.Eval(msBeta) > 1e-12) msBeta += msSigma;
    
    double Z3 = g.Integral(msAlpha, msBeta);

    return log(Z1) + log(Z2) - log(Z3) - log( (mBeta - mAlpha) * (sBeta - sAlpha) / (msBeta - msAlpha) );

}

/// Compute the log ratio of the evidences for the 
/// two distribution models, log [ Z(!same) / Z(same) ]
/// The prior support is dynamically adjusted to the smallest
/// range that doesn't clip the posterior normalization,
/// implicitly performing a crude model selection.  In order
/// to improve the accuracy of the numerical integrations, the
/// ranges are also dynamically selected given Gaussian
/// approximations to the distributions.
/// \param s Summed weights of the first histogram
/// \param nS Total number of weighted events in the first histogram
/// \param r Summed weights of the second histogram
/// \param nR Total number of weighted events in the second histogram

double bayesHistComp::fComputeLogEvidenceRatio(double s, double nS, double r, double nR)
{
    
    double sBar = nS == 0 ? 1 : s / nS;
    double rBar = nR == 0 ? 1 : r / nR;
    
    //////////////////////////////////////////////////
    //      Dynamically calculate prior support     //
    //////////////////////////////////////////////////
    
    // First Gaussian Model
    double sMean = 0;
    double sSigma = 0;
    double sAlpha = 0;
    double sBeta = 0;
    
    if(s == 0)
    {
     	sSigma = 1;
        sAlpha = 0;
        sBeta = 50;
    }
    else
    {
    
    	sMean = 0.5 * (sqrt(sBar * sBar + 4 * s * s) - sBar);
        sSigma = sqrt(2.0 * (sBar * sMean * sMean * sMean) / (2 * s * s - sBar * sMean));
    
        sAlpha = sMean - 11.0 * sSigma;
        sAlpha = sAlpha > 0 ? sAlpha : 0;
        sBeta = sMean + 11.0 * sSigma;

    }

    // Second Gaussian Model
    double rMean = 0;
    double rSigma = 0;
    double rAlpha = 0;
    double rBeta = 0;
    
    if(r == 0)
    {
     	rSigma = 1;
        rAlpha = 0;
        rBeta = 50;
    }
    else
    {
    
    	rMean = 0.5 * (sqrt(rBar * rBar + 4 * r * r) - rBar);
        rSigma = sqrt(2.0 * (rBar * rMean * rMean * rMean) / (2 * r * r - rBar * rMean));
    
        rAlpha = rMean - 11.0 * rSigma;
        rAlpha = rAlpha > 0 ? rAlpha : 0;
        rBeta = rMean + 11.0 * rSigma;

    }
    
    // Gaussian-Gaussian Model
    double srMean = 0;
    double srSigma = 0;
    double srAlpha = 0;
    double srBeta = 0;
    
    if(s == 0)
    {
        srSigma = rSigma;
        srAlpha = rAlpha;
        srBeta = rBeta;
    }
    else if(r == 0)
    {
        srSigma = sSigma;
        srAlpha = rAlpha;
        srBeta = rBeta;
    }
    else
    {
    
        double a = sBar + rBar;
        double b = sBar * rBar;
        double c = s * s * rBar + r * r * sBar;
    
        srMean = ( sqrt( b * b + a * c ) - b ) / a;
        srSigma = sqrt( (b * srMean * srMean * srMean) / (c - srMean * b) );
        
        srAlpha = srMean - 10.0 * srSigma;
        srAlpha = srAlpha > 0 ? srAlpha : 0;
        srBeta = srMean + 10.0 * srSigma;
        
    }
            
    //double alpha = sAlpha < rAlpha ? sAlpha : rAlpha;
    //double beta = sBeta > rBeta ? sBeta : rBeta;
    
    //////////////////////////////////////////////////
    //           Compute evidence ratio             //
    //////////////////////////////////////////////////
    
    // // Define and integrate the first Gaussian integral, expanding the prior support if needed
    TF1 f("f", " exp( - 0.5 * (x - [0]) * (x - [0]) / ([1] * x) ) / sqrt(x)", sAlpha, sBeta);
    f.SetParameter(0, s);
    f.SetParameter(1, sBar);
    
    while(f.Eval(sBeta) > 1e-12) sBeta += sSigma;
    
    double Z1 = f.Integral(sAlpha, sBeta);
    
    // Define and integrate the second Gaussian integral, expanding the prior support if needed
    TF1 g("f", " exp( - 0.5 * (x - [0]) * (x - [0]) / ([1] * x) ) / sqrt(x)", rAlpha, rBeta);
    g.SetParameter(0, r);
    g.SetParameter(1, rBar);
    
    while(g.Eval(rBeta) > 1e-12) rBeta += rSigma;
    
    double Z2 = g.Integral(rAlpha, rBeta);
    
    // Define and integrate the Gaussian-Gaussian integral, expanding the prior support if needed
    TF1 h("g", "exp( - 0.5 * ( (x - [0]) * (x - [0]) / [1] + (x - [2]) * (x - [2]) / [3] ) / x ) / x", srAlpha, srBeta);
    h.SetParameter(0, s);
    h.SetParameter(1, sBar);
    h.SetParameter(2, r);
    h.SetParameter(3, rBar);
    
    while(h.Eval(srBeta) > 1e-12) srBeta += srSigma;
    
    double Z3 = h.Integral(srAlpha, srBeta);

    return log(Z1) + log(Z2) - log(Z3) - log( (sBeta - sAlpha) * (rBeta - rAlpha) / (srBeta - srAlpha) );

}

/// Draw the two input histograms along with the comparison results
/// \param name File name where the image will be saved

void bayesHistComp::fDraw(TGraph *externalPosterior, TString imageName)
{

    //////////////////////////////////////////////////
    //            Create mixture posterior          //
    //////////////////////////////////////////////////   
    
    bool cleanPosterior = false;
    TGraph *posterior = externalPosterior;
    
    if(posterior)
    {
        posterior->Set(mNumPiPoints);
        for(int i = 0; i < mNumPiPoints; ++i) posterior->SetPoint(i, mPi[i], mP[i]);
    }
    else
    {
        posterior = new TGraph(mNumPiPoints, mPi, mP);
        posterior->SetLineColor(2);
        cleanPosterior = true;
    }
    
    TString formatCred = "";
    formatCred += 0.001 * floor(1000 * mCredibility);
    formatCred.ReplaceAll(" ", "");
    
    TString formatMAP = "";
    formatMAP += 0.001 * floor(1000 * mPiMAP);
    formatMAP.ReplaceAll(" ", "");
    
    TString formatOdds = "";
    formatOdds += 0.001 * floor(1000 * mModelOdds);
    formatOdds.ReplaceAll(" ", "");
    
    TString title = "p(#pi > 1/2) = ";
    title += formatCred;
    title += ", #pi_{MAP} = ";
    title += formatMAP;
    title += ", p(S| m, n) = ";
    title += formatOdds;
    posterior->SetTitle(title);

    posterior->GetXaxis()->SetTitle("#pi");
    posterior->GetXaxis()->SetRangeUser(0, 1);

    if(!imageName.Length()) return;

    //////////////////////////////////////////////////
    //                Purty Graphics                //
    //////////////////////////////////////////////////

    TStyle* white = gStyle;
    white->SetCanvasBorderMode(0);
    white->SetCanvasColor(0);
    white->SetFrameBorderMode(0);
    white->SetFrameFillColor(0);
    white->SetTitleColor(1);
    white->SetTitleFillColor(0);
    white->SetStatColor(0);
    white->SetPalette(1);
    white->SetOptStat(0000);
    white->SetOptFit(0000);
    
    TCanvas canvas("canvas","Comparison", 0, 0, 900, 400);
    
    //////////////////////////////////////////////////
    //               Draw histograms                //
    //////////////////////////////////////////////////
    
    // Store title of the original histogram
    TString originalTitle(mHistOne->GetTitle());
    
    // Draw histogram comparison only in the one dimensional case
    if(mHistOne->GetDimension() == 1)
    {
    
        canvas.Divide(2, 1);
        canvas.cd(1);
        
        // Draw the two histograms
        mHistOne->SetTitle("Input Histograms");
        mHistOne->Draw("e");
        mHistTwo->Draw("esame");
        
        // Draw legend
        TLegend legend(0.76, 0.82, 0.98, 0.98);
        legend.SetFillColor(0);

        legend.AddEntry(mHistOne, originalTitle, "l");
        legend.AddEntry(mHistTwo, mHistTwo->GetTitle(), "l");
        legend.Draw();
        
        canvas.cd(2);
    
    }
        
    //////////////////////////////////////////////////
    //            Draw mixture posterior            //
    //////////////////////////////////////////////////   

    posterior->Draw("AL");
    
    //////////////////////////////////////////////////
    //            Draw mixture posterior            //
    //        filled from pi = 1/2 to pi = 1        //
    //////////////////////////////////////////////////   
    
    // Find best approximation to pi = 1/2
    int halfIndex = 0;
    
    for(int i = 0; i < mNumPiPoints; ++i)
    {
        if(mPi[i] < 0.5) halfIndex = i + 1;
        else break;
    }
    
    // Note that the filled graph has to be buffered 
    // with mHistTwo points at p = 0 to ensure that the 
    // fill is between the graph and the y axis
    
    TGraph fillPosterior(mNumPiPoints - halfIndex + 2);
    
    fillPosterior.SetPoint(0, mPi[halfIndex], 0); // First buffer
    
    for(int i = halfIndex; i < mNumPiPoints; ++i)
    {
        fillPosterior.SetPoint(i - halfIndex + 1, mPi[i], mP[i]);
    }
    fillPosterior.SetPoint(mNumPiPoints - halfIndex + 1, mPi[mNumPiPoints - 1], 0); // Second buffer
    
    fillPosterior.SetLineColor(2);
    fillPosterior.SetFillColor(2);
    fillPosterior.Draw("Fsame");

    //////////////////////////////////////////////////
    //                  Save image                  //
    //////////////////////////////////////////////////  
    
    canvas.Print(imageName);
    
    // Restore original title
    mHistOne->SetTitle(originalTitle);

    // and clean up
    if(cleanPosterior) posterior->Delete();

}

/// Find the MAP and compute the posterior mass above pi = 1/2

void bayesHistComp::fCalcCredibility()
{

    // Find the mode
    int modeIndex = 0;
    double mode = 0;

    for(int i = 0; i < mNumPiPoints; ++i)
    {
        if(mP[i] > mode)
        {
            mode = mP[i];
            modeIndex = i;
        }
    }
    
    mPiMAP = mPi[modeIndex];
    
    // Integrate the posterior between pi = 1 / 2 and pi = 1
    mCredibility = 0;
    double dPi = mPi[1] - mPi[0];
    
    for(int i = 0; i < mNumPiPoints; ++i)
    {
    
        if(mPi[i] < 0.5) continue;
        
        mCredibility += mP[i] * dPi;
        
    }
    
    return;

}
