#ifndef _BETA_CHAIN_NEST_

#include "chainBundle.h"

///                                              
///  \author Michael Betancourt,                 
///  Massachusetts Institute of Technology       
///                                              
///  Abstract class defining the nested sampling algorithm.                                  
///                                              


class chainNest: public chainBundle
{

    public:
    
        chainNest(TRandom3 *random);
        
        // Mutators
        void setNumNests(int n) { mNumNests = n; }           ///< Set total number of nested samples
        void setNestThreshold(double t) { mTerm = t; }       ///< Set threshold for early termination threshold
        void setMaxNumSamples(int n) { mMaxNumSamples = n; } ///< Set total number of attempted samples
        
        // Accessors
        double logZ() { return mLogZ; }                      ///< Return expected logZ
        double H() { return mH; }                            ///< Return expected information
        
        // Auxiliary functions
        void nestedSample();
        void setLikelihoodConstraint(bool c) { mGoLikelihood = c; }  ///< Turn on the likelihood constraint
        void setDisplayFrequency(int n) { mDisplayFrequency = n; }   ///< Set display frequency
        
        virtual void computeMeanExpectations();
        virtual void computeSampledExpectations(int nSamples);
        
    protected:
    
        // Nested samping statistics
        double mMinLogL; ///< log likelihood of the nested sample
        double mLogZ;    ///< Current log evidence
        double mH;       ///< Current posterior information 
        
        // Constraint arrays
        
        /// Gradient of the log likelihood constraint surface,
        /// necessary to be memory allocated/freed in a derived implementation.
        double *mGradLogL;         

        // Nested samples
        vector<double> mLogLStore;   ///< log likelihood of the nested samples
        vector<double> mSampleStore; ///< Position in feature space of the nested samples
        
    private:
    
        bool mGoLikelihood; ///< Likelihood constraint flag
    
        int mNumNests;      ///< Maximum number of nested samples
        int mMaxNumSamples; ///< Maximum number of total drawn samples
        
        /// Display intermediate statistics every mDisplayFrequency samples
        /// provided chainBundle::mGoVerbose is set to true.
        /// n = 0 is never displayed, instead the output starts with n = 1.
        /// If mDisplayFrequency <= 0, no statistics are shown.
        int mDisplayFrequency;
        
        /// Set a (lightly tweaked) Skilling termination threshold, 
        /// \f$(N + N_{eff}) \log \mathcal{L}_{\max} * w < t \f$
        ///
        /// See Skilling, J. (2004) Nested Sampling.  In "Maximum Entropy and Bayesian methods in science and engineering"
        /// (ed. G. Erickson, J. T. Rychert, C. R. Smith).  AIP Conf. Proc., 735: 395-405.
        double mTerm;
        
        double fSumLogExp(double a, double b);
        
        virtual bool fConstraint(baseChain *chain);
        virtual double* fNormal(baseChain *chain);
        
        /// Abstract method defining interface for calculating the log likelihood
        virtual double fLogL(double *point) = 0;
        
        /// Abstract method defining interface for calculating the gradient of the log likelihood
        virtual double* fGradLogL(double *point) = 0;
        
        virtual void fStore(double logL, double *sample);
        
};

#define _BETA_CHAIN_NEST_
#endif