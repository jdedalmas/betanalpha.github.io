#ifndef _GPKERNEL_

#include "TString.h"

using namespace std;

///
/// \author Michael Betancourt,
/// Massachusetts Institute of Technology
///
/// Abstract class defining the interface and 
/// private data members for covariance function implementations
///

class gpKernel
{

    public:
        gpKernel(bool ARD, TString title, TString name);
        virtual ~gpKernel() {};
        
        /// Compute the covariance function, or kernel at the training points \f$x_{1}\f$ and \f$x_{2}\f$
        /// \param x1 Pointer to the first training point
        /// \param x2 Pointer to the second training point
        /// \param same Flag indicating that \f$x_{1}\f$ and \f$x_{2}\f$ are the same point
        /// \return \f$k\left(x_{1}, x_{2}\right)\f$

        virtual double kernel(const double *x1, const double *x2, bool same) = 0;

        /// Compute the derivative of the kernel at the training points 
        /// \f$x_{1}\f$ and \f$x_{2}\f$ with respect to the nth hyperparameter
        /// \param x1 Pointer to the first training point
        /// \param x2 Pointer to the second training point
        /// \param same Flag indicating that \f$x_{1}\f$ and \f$x_{2}\f$ are the same point
        /// \param n Index of the hyperparameter
        /// \return \f$dk\left(x_{1}, x_{2}\right) / dh_{n}\f$
        
        virtual double dKernel(const double *x1, const double *x2, bool same, const int n) = 0;
        
        /// Display the hyperparameters of the covariance function
        virtual void displayHyperParameters(const char* prefix) = 0;
        
        /// Return the definition of ARD for the covariance function, if applicable
        virtual TString displayRelevance() { return ""; }
        
        double *hyperParameters() { return mHyperParameters; } ///< Return hyperparameter array
        int nHyperParameters() { return mNumHyperParameters; } ///< Return number of hyperparameters
        
        // Accessors
        bool ARD() { return mARD; }        ///< Return ARD flag
        TString title() { return mTitle; } ///< Return title of the covariance function
        TString name() { return mName; }   ///< Return name of the covariance function
        
    protected:

        int mDim;                 ///< Dimension of feature space
        int mNumHyperParameters;  ///< Number of hyperparameters
        double *mHyperParameters; ///< Hyperparameter array
        
    private:
        
        bool mARD; ///< Does the covariance function implement automatic relevance detection?
        
        TString mTitle; ///< Title of the covariance function
        TString mName;  ///< Name of the covariance function
        
};

#define _GPKERNEL_
#endif
