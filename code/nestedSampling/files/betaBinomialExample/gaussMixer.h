#ifndef _BETA_GAUSSMIXER_

///
/// \author Michael Betancourt,
/// Massachusetts Institute of Technology
///
/// Class implementing a Variational Bayes Gaussian Mixture Model.
/// For a complete description of the algorithm see
/// Bishop, C.M (2007) Pattern Classification and Machine Learning. Springer, New York

class gaussMixer
{

    public:
    
        gaussMixer(double *x, int nData, int dim, int nComponents = 5);
        ~gaussMixer();
        
        // Auxiliary methods
        void cluster();
        void cluster(double* weight);
        
        // Mutators
        void setNumMaxInterations(int n) { mNumMaxIterations = n; } ///< Set maximum number of variational iterations
        void setDeltaLogEvidenceThresh(int t) { mEpsilon = t; }     ///< Set convergence threshold
        void setAlpha0(double a) { mAlpha0 = a; }                   ///< Set mixture Dirichlet parameter
        void setBeta0(double b) { mBeta0 = b; }                     ///< Set Gaussian-Wishart covariance scale
        void setNu0(double n) { mNu0 = n; }                         ///< Set Gaussian-Wishart number of degrees of freedom
        
        // Accessors
        int nComponents() { return mNumComponents; }                ///< Return number of mixture components
        double* N() { return mN; }                                  ///< Return array of effective component multiplicities
        double* m() { return mM; }                                  ///< Return array of mean component positions
        double* sigma() { return mSigma; }                          ///< Return array of mean component covariances
        double* pi() { return mPi; }                                ///< Return array of mean component mixture coefficients
        double* r() { return mR; }                                  ///< Return array of mean responsibilities
        double logEvidence() { return mLogEvidence; }               ///< Return variational bound on mixture log evidence
    
    private:

        double *mData;           ///< Input samples
        int mNumData;            ///< Number of samples
        int mDim;                ///< Dimension of each sample
        int mNumComponents;      ///< Number of gaussian mixture components
        int mNumMaxIterations;   ///< Maximum number of variational interations
        double mEpsilon;         ///< Termination threshold for the change in log evidence
        
        double mAlpha0;          ///< Dirichlet prior parameter (same for each component)
        double mBeta0;           ///< Gaussian-Wishart covariance scale
        double mNu0;             ///< Gaussian-Wishart number of degrees of freedom
        double *mM0;             ///< Gaussian-Wishart mean
        double *mW0inverse;      ///< Gaussian-Wishart covariance
        
        double mLogEvidence;     ///< Lower bound on the model log evidence
        
        double *mN;              ///< Expected occupancy of each mixture component
        double *mM;              ///< Expected mean of each mixture component
        double *mSigma;          ///< Expected covariance of each mixture component
        double *mPi;             ///< Expected weight of each mixture component   
        double *mR;              ///< Expected responsiblity of each sample to each component
    
        // Internal methods
        double fDigamma(double x);
        
        void fCholesky(const double *C, double *L, const int nPoints);
        void fInvert(const double *C, double *Cinverse, double *L, double *LT, const int nPoints);


};

#define _BETA_GAUSSMIXER_
#endif