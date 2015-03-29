#ifndef _BETA_CHAIN_BUNDLE_
                                                                      
#include <vector>
#include "baseChain.h"

class TRandom3;

using namespace std;

///                                              
///  \author Michael Betancourt,                   
///  Massachusetts Institute of Technology       
///                                              
///  Abstract container class for an ensemble of Markov chains.  
///  Unconstrained and constrained Hamiltonian Monte Carlo         
///  transitions are defined in addition to method for initializing 
///  the chains and diagnosing convergence.                     
///       

class chainBundle
{

    public:
    
        chainBundle(TRandom3 *random);
        ~chainBundle();
    
        // Mutators
        void setVerbosity(bool v) { mGoVerbose = v; }                 ///< Set verbosity
        
        void setNumDump(int n) { mNumDump = n; }                      ///< Set number of HMC iterations per drawn sample
        void setNumLeapfrog(int n) { mNumLeapfrog = n; }              ///< Set number of leapfrog steps
        void setStepSize(double s) { mStepSize = s; }                 ///< Set stepsize
        void setStepSizeJitter(double j);                             //   Requires warning, see cpp
        void setMaxStepSize(double m) { mMaxStepSize = m; }           ///< Set maximum stepsize
        void setMinStepSize(double m) { mMinStepSize = m; }           ///< Set minimum stepsize
        void setTargetAcceptRate(double r) { mTargetAcceptRate = r; } ///< Set target accept rate
        void setAverageDecay(double alpha) { mAlpha = alpha; }        ///< Set moving average decay rate
        void setUpdateLambda(double lambda) { mLambda = lambda; }     ///< Set stepsize update temperating parameter
        void setAdaptStepSize(bool adapt) { mAdaptStepSize = adapt; } ///< Set stepsize adaptation flag
        
        void setBurnParameters(int nBurn, int nCheck, double minR);
            
        /// Add a new baseChain* to the bundle
        void addChain(baseChain *object) { mChains.push_back(object); }
            
        // Accesors
        int nChains() { return mChains.size(); }          ///< Return number of chains
        baseChain* chain(int i) { return mChains.at(i); } ///< Return pointer to ith chain
            
        // Auxiliary functions
        void sample();
        void sample(int i);
        bool sample(baseChain *chain);
        
        void seed(double min, double max);
        void seed(int i, double min, double max);
        void burn();
        
        void storeBurnedPoints();
        void restoreBurnedPoints();

    protected:
    
        // Flags
        bool mGoVerbose;          ///< Verbosity flag
        bool mGoBurn;             ///< Burn in readiness flag
        bool mChar;               ///< Burn in completion flag

        // Random number generator
        TRandom3 *mRandom;        ///< Mersenne twistor pseudo-random number generator

        // Hamiltonian Monte Carlo parameters
        int mNumDump;             ///< Number of HMC steps between each sample
        int mNumLeapfrog;         ///< Number of leapfrog steps
        double mStepSize;         ///< Nominal leapfrog stepsize
        double mStepSizeJitter;   ///< Stepsize jitter, in fraction of chainBundle::mStepSize
        double mMaxStepSize;      ///< Maximum stepsize allowed for adaptive algorithms
        double mMinStepSize;      ///< Minimum stepsize allowed for adaptive algorithms
        
        double mXBar;             ///< Exponential moving average of Metropolis accept rate
        double mN;                ///< Exponential moving average normalization
        double mAlpha;            ///< Exponential moving average decay rate

        bool mAdaptStepSize;      ///< Stepsize adaptation flag
        double mTargetAcceptRate; ///< Desired Metropolis accept rate
        double mLambda;           ///< Stepsize update tempering parameter
        
        int mNumSamples;          ///< Total number of computed samples
        
        // Burn in parameters
        int mNumBurn;             ///< Number of burn in iterations
        int mNumCheck;            ///< Number of samples for diagnosing burn in
        double mMinR;             ///< Minimum R statistic for diagnosing burn in

        // Chain containers
        vector<baseChain*> mChains;       ///< Vector of baseChain pointers
        vector<baseChain*>::iterator mIt; ///< Vector iterator 
        
        // Private functions
        
        /// Abstract method defining the interface for a sampling constraint.
        /// Defaults to any support constraint defined in the input chains.
        virtual bool fConstraint(baseChain *chain) { return !(chain->supportViolated()); }
        
        /// Abstract method defining the interface for a sampling constraint normal.
        /// Defautls to any support constraint normal defined in the input chains.
        virtual double* fNormal(baseChain *chain) { return chain->supportNormal(); }

};

#define _BETA_CHAIN_BUNDLE_
#endif