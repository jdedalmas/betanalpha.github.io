#ifndef _GPAUX_

#include <iostream>
#include <Accelerate/Accelerate.h>
#include "math.h"

#include "gpKernel.h"

using namespace std;


/// Compute the lower triangular part of the covariance matrix,
/// using the symmetry of the matrix for the upper triangular components
void gpClassifier::covariance()
{

    double *point = mCinverse;
    double *xOnePtr = mX;
    double *xTwoPtr = mX;
    
    for(int i = 0; i < mNumPoints; ++i, xOnePtr += mDim)
    {
    
        xTwoPtr = mX;
    
        for(int j = 0; j < i + 1; ++j, ++point, xTwoPtr += mDim)
        {
            *point = mKernel->kernel(xOnePtr, xTwoPtr, i == j);
        }
        point += mNumPoints - (i + 1);
        
    }
    
}

/// Compute the derivative of the covariance matrix with respect
/// to the nth hyperparameter.  As in covariance(), only the lower
/// triangular components are calculated (or need be) due to the
/// symmetry properties of the covariance matrix.
/// \param n Index of the hyperparameter

void gpClassifier::dCovariance(const int n)
{

    double *point = mdCdh;    
    double *xOnePtr = mX;
    double *xTwoPtr = mX;
    
    for(int i = 0; i < mNumPoints; ++i, xOnePtr += mDim)
    {
    
        xTwoPtr = mX;
    
        for(int j = 0; j < i + 1; ++j, ++point, xTwoPtr += mDim)
        {
            *point = mKernel->dKernel(xOnePtr, xTwoPtr, i == j, n);
        }
        point += mNumPoints - (i + 1);
        
    }
    
}


/// Calculate the log model evidence from the given the inverse covariance matrix
/// \return log model evidence

double gpClassifier::logEvidence()
{

    cblas_dsymv(CblasRowMajor, CblasLower, mNumPoints, -0.5, mCinverse, mNumPoints, mY, 1, 0.0, mB, 1);
    double term1 = cblas_ddot(mNumPoints, mY, 1, mB, 1);
    
    double term2 = - 0.5 * mLogDet;
    
    double term3 = -0.5 * mNumPoints * log(2 * 3.14159);

    return term1 + term2 + term3;

}

/// Calculate the gradient of the log evidence
/// \param dLogEdh Pointer to gradient array

void gpClassifier::gradLogE(double* dLogEdh)
{


    for(int n = 0; n < mNumHyperParameters; ++n)
    {
        
        // Calculate matrix of derivatives with respect to the nth hyperparameter
        dCovariance(n);
        
        // Calculate the deriviate of the likelihood
        
        // Some slick pointer arithmatic to compute the trace using only
        // the lower triangular elements.  cblas takes care of the vector
        // products.
        double term1 = 0;
        for(int i = 0; i < mNumPoints; ++i)
        {
        
            const double *pointA = mCinverse + i * mNumPoints;
            double *pointB = mdCdh + i * mNumPoints;
            
            for(int j = 0; j < mNumPoints; ++j)
            {
                
                term1 += (*pointA) * (*pointB);
                
                if(j < i)
                {
                    ++pointA;
                    ++pointB;
                }
                else
                {
                    pointA += mNumPoints;
                    pointB += mNumPoints;
                }
                
            }
            
        }
        term1 *= -0.5;
        
        cblas_dsymv(CblasRowMajor, CblasLower, mNumPoints, 1.0, mCinverse, mNumPoints, mY, 1, 0.0, mB, 1);
        cblas_dsymv(CblasRowMajor, CblasLower, mNumPoints, 0.5, mdCdh, mNumPoints, mB, 1, 0.0, mK, 1);
        double term2 = cblas_ddot(mNumPoints, mB, 1, mK, 1);
        
        dLogEdh[n] = term1 + term2;
    
    }

}

/// Calculate the GP predictions for the given test data
/// \param mean Address where the prediction mean will be written
/// \param var Address where the prediction variance will be written
/// \param xtest Pointer to test data input variable array

void gpClassifier::predict(const double* xtest, double& mean, double& var)
{

    const double *xPtr = mX;
    for(double *kPtr = mK; kPtr != mK + mNumPoints; ++kPtr)
    {
        *kPtr = mKernel->kernel(xtest, xPtr, 0);
        xPtr += mDim;
    }
        
    cblas_dsymv(CblasRowMajor, CblasLower, mNumPoints, 1.0, mCinverse, mNumPoints, mK, 1, 0.0, mB, 1);
    
    mean = cblas_ddot(mNumPoints, mY, 1, mB, 1);  
    var = mKernel->kernel(xtest, xtest, 1);
    var -= cblas_ddot(mNumPoints, mK, 1, mB, 1);  
      
    
}

/// Calculate the inverse of C using forward and backward elimination of the Cholesky matrix
void gpClassifier::invert()
{

    // Initialize
    mLogDet = 0;
    
    /////////////////////////////////////////////////////////////////////////
    //                                                                     //
    // This code uses a row major convention for matrix manipulations,     //
    // but LAPACK routines assume a column major convention.  Because      //
    // our matrices are symmetric, however, these conflicting conventions  //
    // have no theoretical affect when referencing matrix elements.        //
    //                                                                     //
    // When attempting to save space by storing only the upper or lower    //
    // triangular parts of the matrix we have to switch our                //
    // definition of "U" and "L" in the LAPACK routines.                   //
    //                                                                     //
    /////////////////////////////////////////////////////////////////////////

    char uplo[] = "U";
    __CLPK_integer clpk_n = mNumPoints;
    __CLPK_integer info = 0;
    
    // Perform Cholesky decomposition
    dpotrf_(uplo, &clpk_n, mCinverse, &clpk_n, &info);
    if(info != 0)
    {
        cout << "gpClassifier::invert() - Cholesky decomposition failed, C isn't positive definite!" << endl;
        exit(0);
    }
    
    // Calculate the log determinant
    
    double *point = mCinverse;    
    for(int i = 0; i < mNumPoints; ++i)
    {
        mLogDet += log(*point);
        point += mNumPoints + 1;
    }
    
    mLogDet *= 2.0;
    
    // Calculate the lower triangular component of the inverse matrix
    dpotri_(uplo, &clpk_n, mCinverse, &clpk_n, &info);
    if(info != 0)
    {
        cout << "gpClassifier::invert() - Cholesky inversion failed!" << endl;
        exit(0);
    }

}

#define _GPAUX_
#endif
