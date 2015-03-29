#ifndef _GPCONJUGATEGRADIENT_

#include "math.h"
#include <iostream>

#include "gpKernel.h"

/// Implementation of a nonlinear conjugate gradient algorithm
/// for learning the optimal covariance function mHyperParameters
/// by optimizing the log evidence.
///
/// Note that since we're seeking to maximize the log evidence
/// we use the conjugate gradient algorithm to minimize the negative
/// log evidence

/// \param goVerbose Switch for verbose search output

/// \see fDot(double* g, double* s)
/// \see fInterpolate(double d1, double d2, double f1, double f2, double dfds1, double dfds2)
/// \see fExtrapolate(double d1, double d2, double f1, double f2, double dfds1, double dfds2)

void gpClassifier::fSearch(bool goVerbose)
{

    ///////////////////////////////////
    //     Linesearch Parameters     //
    ///////////////////////////////////
    
    double sigma = 0.50;
    double rho = 0.1;
    
    // Initial step sizes
    double oldSlope = 0;
    double delta1 = 0;
    double delta2 = 0;
    
    // Line search variables
    double f1 = 0;
    double f2 = 0;
    double dfds1 = 0;
    double dfds2 = 0;
    double d1 = 0;
    double d2 = 0;
    
    // Number of matrix inversions per search
    int nEpochs = 0;
    
    ////////////////////////////////////////////////////
    //    Initiate the conjugate gradient descent     //
    ////////////////////////////////////////////////////

    // Store initial point, using mHyperparameters for exploration afterwards
    for(int i = 0; i < mNumHyperParameters; ++i) mH1[i] = mHyperParameters[i];
    mH2 = mHyperParameters;

    // negative log evidence and negative gradient at the initial point
    covariance(); invert();
    f1 = - logEvidence();
    gradLogE(mG1);
    for(double *gradPtr = mG1; gradPtr != mG1 + mNumHyperParameters; ++gradPtr) *gradPtr *= -1;
    ++nEpochs;

    // Initial search direction follows the gradient
    for(int i = 0; i < mNumHyperParameters; ++i) mS[i] = - mG1[i];
    for(int i = 0; i < mNumHyperParameters; ++i) mGi[i] = mG1[i];
    
    // Normalize initial search direction
    double norm = 1 / sqrt(fDot(mS, mS));
    for(int i = 0; i < mNumHyperParameters; ++i)
    {
        mD[i] = mS[i] * norm;
    }
    
    // Initial directional derivative
    dfds1 = fDot(mG1, mD);
    
    // Initial step size guess
    oldSlope = dfds1;
    delta1 = 50.0 / (1 + fabs(dfds1));
    
    //////////////////////////////////////////////////
    //     Start the conjugate gradient descent     //
    //////////////////////////////////////////////////

    int nConsecutiveRestarts = 0;
    bool restart = 0;

    // Perform line searches along conjugate directions
    // until the search criteria have been satisfied
    while(1)
    {

        // Check if a restart has been requested
        if(restart)
        {

            if(goVerbose) cout << "\tRestarting line search..." << endl << endl;

            // Check that the restarted search direction points to a minimum
            if(nConsecutiveRestarts > 1) break;

            ++nConsecutiveRestarts;
            
            // Reset search direction to local gradient
            for(int i = 0; i < mNumHyperParameters; ++i) mS[i] = - mG1[i];
            for(int i = 0; i < mNumHyperParameters; ++i) mGi[i] = mG1[i];
            
            // Normalize the search direction
            norm = 1 / sqrt(fDot(mS, mS));
            for(int i = 0; i < mNumHyperParameters; ++i) mD[i] = mS[i] * norm;
            
            // Directional derivative
            dfds1 = fDot(mG1, mD);
            
            // Check that the restarted search direction points to a minimum
            if(dfds1 > 0) break;

            // New initial step size guess
            delta1 = 50.0 / (1 + fabs(dfds1));
            
            // Reset restart flag
            restart = 0;
            
        }  
        else
        {
            if(goVerbose) cout << "\tBeginning line search along new conjugate gradient direction with " << endl;
            if(goVerbose) cout << "\t\tf1 = " << f1 << ", dfds1 = " << dfds1 << endl << endl;
        }
        
        
        // Restart if the initial slope is positive
        if(dfds1 > 0) 
        {
            restart = 1;
            continue;
        }   
        
        if(goVerbose) cout << "\t\tComputing initial guess..." << endl;

        // Initial guess
        for(int i = 0; i < mNumHyperParameters; ++i) mH2[i] = mH1[i] + delta1 * mD[i];
        
        // Evaluate the minus log evidence
        // and minus gradient at this guess
        covariance(); invert();
        f2 = - logEvidence();
        gradLogE(mG2);
        for(double *gradPtr = mG2; gradPtr != mG2 + mNumHyperParameters; ++gradPtr) *gradPtr *= -1;
        ++nEpochs;
        
        // Calculate the directional derivative at the guess
        dfds2 = fDot(mG2, mD);                    

        if(goVerbose) cout << "\t\t\tf2 = " << f2 << ", dfds2 = " << dfds2 << endl << endl;
        
        //////////////////////////////////////////////////
        //             Start the line search            //
        //////////////////////////////////////////////////

        while(1)
        {

            // Break out of any bad loops
            if(nEpochs > mMaxEpochs) break;

            // Calculate the distances of the inital point and 
            // the guess projected along the line search
            d1 = fDot(mH1, mD);
            d2 = fDot(mH2, mD);

            // If the directional derivative at the guess is too
            // large then the guess has overshot the minimum and
            // an interpolation is necessary 
            bool a2 = dfds2 > - sigma * dfds1;

            // If the function has not sufficiently decreasaed
            // in value at the guess then we've passed the
            // minimum and an interpolation is necessary 
            bool b = f2 > f1 + rho * (d2 - d1) * dfds1;
            
            // Check if an interpolation is necessasry
            while(a2 || b)
            {
                
                if(goVerbose) cout << "\t\tInterpolating..." << endl;

                // fInterpolate between the the points
                double min = fInterpolate(d1, d2, f1, f2, dfds1, dfds2);

                if(goVerbose) cout << "\t\t\tf1 = " <<  f1 << ", f2 = " << f2 << endl;
                if(goVerbose) cout << "\t\t\tdfds1 = " << dfds1 << ", dfds2 = " << dfds2 << endl;
                if(goVerbose) cout << "\t\t\td1 = " << d1 << ", d2 = " << d2 << ", interpolation = " << min << endl;

                // Calculate the new point's location
                // in the full feature space
                for(int i = 0; i < mNumHyperParameters; ++i)
                {
                    mH2[i] = mH1[i] + (min - d1) * mD[i];
                }
                
                // Else go ahead and evaluate all of the
                // relevant functions at this new point
                covariance(); invert();
                f2 = - logEvidence();
                gradLogE(mG2);
                for(double *gradPtr = mG2; gradPtr != mG2 + mNumHyperParameters; ++gradPtr) *gradPtr *= -1;
                ++nEpochs;

                // Project distance and gradient along the line search
                d2 = fDot(mH2, mD);
                dfds2 = fDot(mG2, mD);

                if(goVerbose) cout << "\t\t\tf2 = " << f2 << ", dfds2 = " << dfds2 << endl << endl;                
                
                // Recheck the interpolation conditions at the new guess
                a2 = dfds2 > - sigma * dfds1;
                b = f2 > f1 + rho * (d2 - d1) * dfds1;
                
            }
            
            // If the directional derivative at the new guess
            // is too negative then an extrapolation is necessary 
            bool a1 = dfds2 < sigma * dfds1;
            
            // If the directional derivative is sufficiently close
            // to zero then the line search has finished successfully
            if(!a1) break;

            if(goVerbose) cout << "\t\tExtrapolating..." << endl;

            // Otherwise, go on and fExtrapolate
            double min = fExtrapolate(d1, d2, f1, f2, dfds1, dfds2);

            if(goVerbose) cout << "\t\t\tf1 = " <<  f1 << ", f2 = " << f2 << endl;
            if(goVerbose) cout << "\t\t\tdfds1 = " << dfds1 << ", dfds2 = " << dfds2 << endl;
            if(goVerbose) cout << "\t\t\td1 = " << d1 << ", d2 = " << d2 << ", extrapolation = " << min << endl;

            // Shift the search window over to the new points
            // First with by moving the initial point over to our first guess
            d1 = d2;
            f1 = f2;
            dfds1 = dfds2;
            double hTemp = 0;
            for(int i = 0; i < mNumHyperParameters; ++i)
            {
                hTemp = mH2[i];
                mH2[i] = hTemp + (min - d1) * mD[i];
                mH1[i] = hTemp;
                mG1[i] = mG2[i];
            }
                        
            // And then using the extrapolation as our new guess
            d2 = fDot(mH2, mD);

            if(goVerbose) cout << "\t\t\td1 = " << d1 << ", d2 = " << d2  << endl;
            
            // Else go ahead and evaluate all of the
            // relevant functions at this new point
            covariance(); invert();
            f2 = - logEvidence();
            gradLogE(mG2);
            for(double *gradPtr = mG2; gradPtr != mG2 + mNumHyperParameters; ++gradPtr) *gradPtr *= -1;
            ++nEpochs;

            dfds2 = fDot(mG2, mD);

            if(goVerbose) cout << "\t\t\tf2 = " << f2 << ", dfds2 = " << dfds2 << endl << endl;
            
            // Return to the top of the loop to check that
            // the extrapolation did not go too far (else fInterpolate again)
            // but that it went far enough (else fExtrapolate again)
            
        }

        if(restart) continue;
        nConsecutiveRestarts = 0;
        
        if(nEpochs > mMaxEpochs) break;
        
        // If the gradient at the new point is sufficiently
        // small then end the conjugate gradient search.
        // Continuing with G2 = 0 will just result in another
        // line search along the same direction, while 
        // restarting will lead to a poor search direction
        //if(sqrt(fDot(mG2, mG2)) < 0.01) break;
        if(sqrt(fDot(mG2, mG2)) / (double)mNumHyperParameters < 4) break;
        
        if(goVerbose) 
        {
            cout << "\t\tLine search complete, <mG2_{i}> = " 
                 << sqrt(fDot(mG2, mG2)) / (double)mNumHyperParameters << endl << endl;
        }

        // Calculate new search direction
        // using the Polak-Ribiere approximation
        double numer = 0;
        double denom = 0;
        
        for(int i = 0; i < mNumHyperParameters; ++i)
        {
            numer += (mG2[i] - mGi[i]) * mGi[i];
            denom += mGi[i] * mGi[i];
        }

        for(int i = 0; i < mNumHyperParameters; ++i)
        {
            mS[i] = - mG2[i] + mS[i] * numer / denom;
        }
        
        // Normalize new search direction
        
        double norm = 1 / sqrt(fDot(mS, mS));
        for(int i = 0; i < mNumHyperParameters; ++i) mD[i] = mS[i] * norm;
        
        // Calculate the new directional derivative and initial guess
        dfds2 = fDot(mG2, mD);
        
        double alpha = fabs(oldSlope / dfds2);
        if(alpha < 100)
        {
            delta2 = alpha * delta1;
        }
        else
        {
            delta2 = 100 * delta1;
        }
        oldSlope = dfds2;
        
        // Set current minimum to the inital point of the next iteration
        for(int i = 0; i < mNumHyperParameters; ++i)
        {
            mH1[i] = mH2[i];
            mG1[i] = mG2[i];
            mGi[i] = mG2[i];
        }
        f1 = f2;
        dfds1 = dfds2;
        delta1 = delta2;
        
    }
    
    // Final point
    for(int i = 0; i < mNumHyperParameters; ++i) mH2[i] = mH1[i];
    
    return;

}

/// Calculate the dot product of two arrays
/// \param g Pointer to first array
/// \param s Pointer to second array
/// \return fDot product

double gpClassifier::fDot(double* g, double* s)
{

    double f = 0;
    
    double *gPtr = g;
    double *sPtr = s;
    for(int i = 0; i < mNumHyperParameters; ++i)
    {
        f += (*gPtr) * (*sPtr);
        ++gPtr;
        ++sPtr;
    }
    
    return f;
    
}

/// Interpolate a minimum of a function f between two
/// points, d1 and d2, on a one-dimensional line
/// approximating f as a quadratic

/// \param d1 Position of smaller point on the line
/// \param d2 Position of larger point on the line
/// \param f1 f evaluated at d1
/// \param f2 f evaluated at d2
/// \param dfds1 Derivative of f at d1
/// \param dfds2 Derivative of f at d2

/// \return Position of the interpolation

double gpClassifier::fInterpolate(double d1, double d2, double f1, double f2, double dfds1, double dfds2)
{

    
    double dd = d2 - d1;
    double df = f2 - f1;
    double ddf = dfds2 - dfds1;
    double dmin = 0;

    dmin = (0.5 * ddf * (d1 + d2) - df) / ddf;

    if(dmin - d1 < dd * 0.1)
    {
        dmin = d1 + dd * 0.1;
    }
    
    if(dmin > d2)
    {
        dmin = d2 - dd * 0.1;
    }
    
    return dmin;

}

/// Extrapolate a minimum of a function f from two
/// points, d1 and d2, on a one-dimensional line
/// approximating f first as a quadratic and then
/// as linear if necessary

/// \param d1 Position of smaller point on the line
/// \param d2 Position of larger point on the line
/// \param f1 f evaluated at d1
/// \param f2 f evaluated at d2
/// \param dfds1 Derivative of f at d1
/// \param dfds2 Derivative of f at d2

/// \return Position of the extrapolation

double gpClassifier::fExtrapolate(double d1, double d2, double f1, double f2, double dfds1, double dfds2)
{

    double dd = d2 - d1;
    double df = f2 - f1;
    double ddf = dfds2 - dfds1;
    double dmin = 0;

    // We want to use as much information as possible from
    // the two points, while ensuring that the resulting
    // quadratic opens upwards to ensure a proper extrapolation
    
    // Try using the slopes at each point 
    if(ddf > 0) 
    {
        dmin = (0.5 * ddf * (d1 + d2) - df) / ddf;
    }
    // Otherwise use the slope only at the initial point
    else
    {
    
        double a = df - dfds1 * dd;
        double b = df / (dd * dfds1);
        
        if(a > 0)
        {
            dmin = d1 + 0.5 * dd / (1.0 + b);
        }
        // Otherwise resort to a linear extrapolation
        else
        {
            a = d1 + f1 / fabs(dfds1);
            b = d2 + f2 / fabs(dfds2);
            dmin = (a < b) ? a : b;            
        }
        
    }
    
    // Ensure an extrapolation
    if(dmin < d2) dmin = d2;
    
    return dmin;
        
}


#define _GPCONJUGATEGRADIENT_
#endif