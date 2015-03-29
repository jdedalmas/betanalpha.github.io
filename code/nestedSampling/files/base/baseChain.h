#ifndef _BETA_BASECHAIN_

///                  
///  \author Michael Betancourt,
///  Massachusetts Institute of Technology       
///                                              
///  Abstract class defining the state of a      
///  Markov chain with invariant distribution    
///  \f$ P \propto \exp(-E)\f$.  Auxiliary data 
///  members allow for Hamiltonian Monte Carlo           
///  transitions and diagnosing convergence.     
///                                              

class baseChain
{

    public:
    
        baseChain(int dim);
        virtual ~baseChain();
        baseChain(const baseChain& chain);
        baseChain& operator = (const baseChain& chain);


        // Accessors
        int dim() { return mDim; } ///< Return baseChain::mDim
        
        double *x() const { return mPoint; }          ///< Return baseChain::mPoint
        double *p() const { return mMomentum; }       ///< Return baseChain::mMomentum
        double *mInv() const { return mInverseMass; } ///< Return baseChain::mInverseMass

        double pMp();
        double E(bool recompute = 0);
        double E() const { return mE; } ///< Const version of E() for copy constructor \sa E()
        double logP(bool recompute = 0);
        double *gradE(bool recompute = 0);
        double *gradE() const { return mGradE; } ///< Const version of gradE() for copy constructor \sa gradE()
        
        double nBurn() { return mNumBurnCheck; } ///< Return baseChain::mNumBurnCheck
        double *burnSum() { return mBurnSum; } ///< Return baseChain::mBurnSum
        double *burnSum2() { return mBurnSum2; } ///< Return baseChain::mBurnSum2
        
        /// \return Metropolis accept rate over history of chain
        double acceptRate() { return mNumAccept / (mNumAccept + mNumReject); } 
        
        // Mutators
        void clearChainStats();
        void incrementChainSums();
        void incrementAccept() { ++mNumAccept; } ///< Increment Metropolis accept counter
        void incrementReject() { ++mNumReject; } ///< Increment Metropolis reject counter
        
        // Auxiliary functions
        
        /// Virtual method defining the interface for checking violation of valid support
        virtual bool supportViolated() { return false; }
        /// Virtual method defining the interface for returning the
        /// normal to the constraint boundary at the current point
        virtual double* supportNormal() { return mNormal; }
        
        void storeCurrentPoint();
        void restoreStoredPoint();
        
        void storeBurnedPoint();
        void restoreBurnedPoint();

    protected:
    
        int mDim;                ///< Dimension of the feature space
    
        double *mPoint;          ///< Current point in feature space
        double *mStorePoint;     ///< Stored point in feature space
        double *mBurnedPoint;    ///< Stored point in feature space
        
        double *mMomentum;       ///< Current momentum
        double *mInverseMass;    ///< Inverse mass of each dimension
        
        double mE;               ///< Current potential energy
        double *mGradE;          ///< Current energy gradient
        
        double *mNormal;         ///< Normal to support (or any auxiliary) constraint surface
        
        // Convergence Diagnostics
        double mNumBurnCheck;    ///< Number of samples currently burned through

        /// Array of position components summed over burn in
        /// \see clearChainStats(), incrementChainSums()        
        double *mBurnSum;
        
        /// Array of squared position components summed over burn in
        /// \see clearChainStats(), incrementChainSums()
        double *mBurnSum2;
        
        double mNumAccept;       ///< Number of Metropolis accepts over the history of the chain
        double mNumReject;       ///< Number of Metropolis rejects over the history of the chain
        
        /// Virtual method defining the interface for
        /// computing the energy at the current point
        virtual void fComputeE() = 0;

        /// Virtual method defining the interface for computing 
        /// the gradient of the energy at the current point
        virtual void fComputeGradE() = 0;

};

#define _BETA_BASECHAIN_
#endif