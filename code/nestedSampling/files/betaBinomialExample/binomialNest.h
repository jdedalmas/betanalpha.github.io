#ifndef _BETA_BINOMIAL_

///
/// \mainpage Nested Sampling with Constrained Hamiltonian Monte Carlo
///
/// A suite of classes facilitating Hamiltonian Monte
/// Carlo (both constrained and unconstrainted) and its
/// use in nested sampling.  
/// 
/// baseChain implements the necessary framework
/// for a HMC Morkov chain, including contrainted HMC
/// when sampling from a distribution with bounded
/// support.
///
/// chainBundle organizes a container of baseChains,
/// including the functionality to sample from the chains,
/// burn in, and diagnose burn in.
/// 
/// chainNest implements nested sampling, including
/// early termination determination and the calculation
/// of posterior expectations.
///
/// An example implementation of a binomial likelihood
/// with beta prior is also included in betaNestedObject
/// and binomialNest.  This implemenation also demonstrates
/// the use of a Variational Bayes Gaussian Mixture Model
/// to infer multiple modes in the posterior samples.
///
/// Note that a ROOT (http://root.cern.ch/drupal/)
/// installation is required for TRandom3, a Mersenne 
/// twistor psuedo-random number generator.  If an additional 
/// generator is provided then the code would be removed 
/// of all dependencies outside of the C++ stdlib
/// (modulo a few mathematical libraries used in gaussMixer).
///

#include "chainNest.h"
#include <vector>

class TRandom3;

using namespace std;

///                                              
///  \author Michael Betancourt,                     
///  Massachusetts Institute of Technology       
///                                              
///  Perform nested sampling with a binomial likelihood
///

class binomialNest: public chainNest
{

    public:
    
        binomialNest(double k, double N, TRandom3 *random);
        ~binomialNest();
        
        void inferPosteriorModes();
                      
    private:
    
        double mK; ///< Number of successful trials
        double mN; ///< Total number of trials
        double mMu; ///< Mean of binomial distribution

        double fLogL(double *point);
        double *fGradLogL(double *point);

};
  
/// \example main.cpp
        
#define _BETA_BINOMIAL
#endif