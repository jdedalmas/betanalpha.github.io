#ifndef _BETAMATRIX_

#include "math.h"

/// Compute the Cholesky-Banachiewicz decomposition of C
/// \param C Input symmetric matrix
/// \param L Storage for output Cholesky matrix
/// \param nPoints Size of input matrix

void gaussMixer::fCholesky(const double *C, double *L, const int nPoints)
{
        
    // Zeroes
    double *point = L;
    for(unsigned int i = 0; i < nPoints; ++i)
    {
        for(unsigned int j = 0; j < nPoints; ++j)
        {
            *point = 0;
            ++point;
        }
    }
    
    // Cholesky-Banachiewicz (Row by Row)
    for(unsigned int i = 0; i < nPoints; ++i)
    {
        // Loop over nonzero columns
        for(unsigned int j = 0; j <= i; ++j)
        {
        
            double arg = C[i * nPoints + j];
            // Diagonal elements
            if(j == i)
            {

                for(unsigned int k = 0; k < j; ++k)
                {
                    arg -= L[i * nPoints + k] * L[i * nPoints + k];
                }
                arg = sqrt(arg);
            }
            // Nondiagonal Elements
            else
            {
                for(unsigned int k = 0; k < j; ++k)
                {
                    arg -= L[i * nPoints + k] * L[j * nPoints + k];
                }
                
                arg /= L[j * nPoints + j];
            }
            
            L[i * nPoints + j] = arg;
            
        }
        
    }

}

/// Compute the inverse of C using forward and backward
/// elimination of the Cholesky matrix
/// \param C Input symmetric matrix
/// \param Cinverse Storage for the output inverse matrix
/// \param L Cholesky matrix
/// \param LT Storage for the transpose of the Cholesky matrix
/// \param nPoints Size of input matrix

void gaussMixer::fInvert(const double *C, double *Cinverse, double *L, double *LT, const int nPoints)
{

    // Cholesky decomposition
    fCholesky(C, L, nPoints);
    
    for(unsigned int i = 0; i < nPoints; ++i)
    {
        for(unsigned int j = 0; j < nPoints; ++j)
        {
            LT[i * nPoints + j] = L[j * nPoints + i];
        }
    }

    // Inverse matrix begins as a set of unit vectors
    double *point = Cinverse;
    for(unsigned int i = 0; i < nPoints; ++i)
    {
        
        for(unsigned int j = 0; j < nPoints; ++j)
        {
            
            if(i == j)
            {
                *point = 1;
            }
            else
            {
                *point = 0;
            }
            
            ++point;
            
        }
        
    }
    
    // Calculate the columns of the matrix inverse
    for(unsigned int n = 0; n < nPoints; ++n)
    {
    
        bool isSolution = 1;
        
        // Forward Gaussian elimination
        for(unsigned int i = 0; i < nPoints; ++i)
        {
            
            for(unsigned int j = 0; j < i; ++j) 
            {
                Cinverse[i * nPoints + n] -= L[i * nPoints + j] * Cinverse[j * nPoints + n];
            }
            
            if(L[i * nPoints + i] == 0)
            {
                isSolution = 0;
                break;
            }
            else
            {
                Cinverse[i * nPoints + n] /= L[i * nPoints + i];
            }
            
        }
        
        assert(isSolution);
        
        // Backward Gaussian elimination
        for(unsigned int i = 0; i < nPoints; ++i)
        {
            
            int iBack = nPoints - 1 - i;
            for(unsigned int j = 0; j < i; ++j) 
            {
                int jBack = nPoints - 1 - j;
                Cinverse[iBack * nPoints + n] -= LT[iBack * nPoints + jBack] * Cinverse[jBack * nPoints + n];
            }
            
            if(LT[iBack * nPoints + iBack] == 0)
            {
                isSolution = 0;
                break;
            }
            else
            {
                Cinverse[iBack * nPoints + n] /= LT[iBack * nPoints + iBack];
            }
            
        }
        
        assert(isSolution);
    
    }
    
}

#define _BETAMATRIX_
#endif