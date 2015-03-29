#ifndef _BETA_BETACHAIN_

#include "baseChain.h"

///                                              
///  betaChain                            
///                                              
///  \author Michael Betancourt,
///  Massachusetts Institute of Technology       
///                                              
///  Markov chain with the invariant distribution 
///  \f$ \mathrm{Be}(\alpha, \beta)\f$.  Note that \f$x\f$  
///  has the finite support \f$x \in [0, 1] \f$
///  and constrained Hamiltonian Monte Carlo is  
///  necessary to to avoid transitions outside   
///  of this support.                            
///                                              

class betaChain: public baseChain
{

    public:
    
        betaChain(double alpha, double beta);
        ~betaChain() {};
        
        bool supportViolated();
        double* supportNormal();        
        
    private:
    
        double mAlpha; ///< \f$\alpha\f$ parameter of the Beta distribution
        double mBeta;  ///< \f$\beta\f$ parameter of the Beta distribution
            
        void fComputeE();
        void fComputeGradE();

};

#define _BETA_BETACHAIN_
#endif