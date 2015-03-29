#ifndef _GPCLASSIFIER_

#include <vector>
#include "TString.h"

// Forward declarations
class gpKernel;
class TFile;
class TTree;

using namespace std;

/// \mainpage
///
/// \author Michael Betancourt,
/// Massachusetts Institute of Technology
///
/// Implementation of gaussian process regression using
/// ROOT (http://root.cern.ch/drupal/) libraries for data I/O.
///
/// The CBLAS/LAPACK implementation in Apple's Accelerate framework
/// is used for the heavy linear algebra work in order to
/// fully take advantage of Apple hardware.  Moving to a standard
/// CBLAS/LAPACK implementation requires a few tweaks in
/// gpClassifier::invert().
///
/// For more information on gaussian processes see
/// http://www.guassianprocesses.org

/// \example main.cpp

///
/// \author Michael Betancourt,
/// Massachusetts Institute of Technology
///
/// Base class implementing gaussian process regression.
/// The name classifier is used as the initial motivation for the
/// implementation was regression with the function
/*!
\f[
y = \left\{ \begin{array}{rc} +1 ,& \mathrm{Signal} \\ -1 ,& \mathrm{Background} \\ \end{array} \right.
\f]
*/
///
/// <b> Basic Usage </b>
///
/// The constructor gpClassifier sets the name of the file containing
/// training data upon instantiation.
///
/// setInputTree() sets the specific tree name within the given file.
/// 
/// addInputVariable() sets which input variables will be used for training,
/// setOutputVariable() sets the variable that will be used for the regression output.
///
/// retrieveTrainingData() processes the input file, validating the input
/// and output variables.
///
/// useKernel() sets the external kernel instance used for the training.
///
/// prepareTrainingData() allocates all necessary memory and parses the
/// input/output variables into local arrays.  In the process the input
/// variables are standardized by subtracting the sample means and
/// normalizing by the standard deviations.
///
/// train() implements a conjugate gradient optimization of the kernel
/// hyperparameters in order to maximize the model evidence.
///
/// prepareClassifier() readies the class for prediction by building and inverting
/// the covariance matrix of the training data given the current hyperparameters.
///
/// displayRelevance() prints out the relevance of each variable provided
/// that automatic relevance determination has been implemented by the given kernel.
///
/// processSample() computes predictions for all points in the given file.  Note
/// that the file must have a tree with the branches for each input and output variable.
///
/// <b> Tips </b>
/// 
/// Gaussian processes assume that input variables follow gaussian distributions,
/// or in other words the algorithm uses only mean and variance information.
/// Algorithmic performance can be greatly improved by preprocessing the input
/// variables so that they are approximately gaussian.  Taking the log of
/// positive variables, for example.

class gpClassifier
{

    public:

        gpClassifier(TString trainingFileName);
        ~gpClassifier();
        
        // Mutators
        void useKernel(gpKernel* kernel) { mKernel = kernel; } ///< Set pointer to external kernel implementation

        void setInputTree(TString name) { mTrainingTreeName = name; }                ///< Set name of input tree
        void addInputVariable(TString name) { mInputName.push_back(name); }          ///< Add input variable
        void setOutputVariable(TString name) { mOutputName = name; }                 ///< Set name of output variable
        
        /// Set number of conjugate gradient iterations
        /// in each hyperparameter optimization
        void setMaxSearchEpochs(int n) { mMaxEpochs = n; } 

        /// Set minimum log evidence required for initiating a training iteration
        void setMinInitLogEvidence(double m) { mMinInitLogEvidence = m; }

        // Accessors
        //double logEvidence() { return mLogEvidence; } ///< Return the log model evidence
        int dim() { return mDim; } ///< Return dimensionality of the feature space
        
        // Training/Testing Methods
        void retrieveTrainingData();
        void prepareTrainingData();
        void train(int nTrainingIterations = 1, bool goVerbose = false);
        void train(double* hyperStart, bool goVerbose = false);
        
        void useHyperParameters(double* hyperParameters);
        void storeHyperParameters();
        
        void displayRelevance();
        
        void prepareClassifier();
        void testPoint(double* xtest, double& mean, double& variance);
        void processSample(TString testFileName, TString testTreeName = "");
        
        // Kernel Methods
        void covariance();
        void dCovariance(const int n);
        double logEvidence();
        void gradLogE(double* dLogEdh);
        void predict(const double* xtest, double& mean, double& var);
        void invert();
        
    private:
    
        // Flags
        bool mReady;              ///< Regression readiness flag
        bool mTrained;            ///< Existing training flag
        bool mGoHyperParameters;   ///< Existing hyperparameters flag
    
        // I/O
        TFile* mTrainingFile;       ///< Input training file
        TTree* mTrainingTree;       ///< Tree containing training data
        
        TString mTrainingFileName;  ///< Name of input training file
        TString mTrainingTreeName;  ///< Name of input training tree
        
        vector<TString> mInputName; ///< Vector of input variable names
        TString mOutputName;        ///< Name of output variable
    
        // Array dimensions
        int mDim;                ///< Dimensionality of feature space
        int mNumPoints;          ///< Number of training points
        int mNumHyperParameters; ///< Number of hyperparameters
        
        // Internal data 
        gpKernel* mKernel; ///< Pointer to external kernel        

        double mLogEvidence;        ///< log evidence of the existing model
    
        double mLogDet;             ///< log determinant of inverse covariance matrix
        double* mCinverse;          ///< Pointer to inverse covariance matrix
        double* mdCdh;              ///< Matrix derivative of covariance matrix
        
        double* mHyperParameters;   ///< Pointer to hyperparameter array
        
        double* mX;                 ///< Input variables array
        double* mY;                 ///< Output array
        
        double* mMean;              ///< Mean array, used for standarizing input variables
        double* mSigma;             ///< Standard deviation array, used for standarizing input variables
        
        double* mB;                 ///< Utility array
        double* mK;                 ///< Utility array
        
        // Conjugate gradient vectors and arrays
        double* mH1; ///< Position in hyperparameter space
        double* mH2; ///< Position in hyperparameter space
        double* mS;  ///< Search direction
        double* mD;  ///< Normalized search direction
        double* mG1; ///< log evidence gradient
        double* mG2; ///< log evidence gradient
        double* mGi; ///< log evidence gradient
        
        int mMaxEpochs; ///< Number of conjugate gradient iterations in each hyperparameter optimization
        double mMinInitLogEvidence; ///< Minimum log evidence required for initializing a training iteration
        
        // Conjugate gradient methods
        void fSearch(bool goVerbose);
        double fDot(double* g, double* s);
        double fInterpolate(double d1, double d2, double f1, double f2, double dfds1, double dfds2);
        double fExtrapolate(double d1, double d2, double f1, double f2, double dfds1, double dfds2);
        
        
    friend class gpCovariance;
        
};

#define _GPCLASSIFIER_
#endif