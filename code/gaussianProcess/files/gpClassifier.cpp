#include <iostream>
#include <map>
#include "math.h"
#include <time.h>
#include <cstdlib>

#include "TString.h"
#include "TFile.h"
#include "TTree.h"
#include "TObjArray.h"
#include "TRandom3.h"

#include "gpClassifier.h"
#include "gpAux.h"
#include "gpConjugateGradient.h"

#include "gpKernel.h"

/// Constructor
/// \param trainingFileName Name of file containing labeled training data
/// \see ~gpClassifier()

gpClassifier::gpClassifier(TString trainingFileName)
{

    mReady = false;
    mTrained = false;
    mGoHyperParameters = false;

    mTrainingFileName = trainingFileName;
    mTrainingTreeName = "";
    mOutputName = "";
    
    mDim = 0;
    mNumPoints = 0;
    mNumHyperParameters = 0;
    
    mKernel = 0;
    
    mLogEvidence = 0;
    mLogDet = 0;

    mMaxEpochs = 60;
    mMinInitLogEvidence = -1e6;
    
}

/// Destructor
/// \see gpClassifier(TString trainingFileName, covarianceType type)

gpClassifier::~gpClassifier()
{

    // Clean up if needed
    if(mReady)
    {
    
        mTrainingFile->Delete();
        
        delete[] mX;
        delete[] mY;
        delete[] mCinverse;
        delete[] mMean;
        delete[] mSigma;
        
        delete[] mB;
        delete[] mK;
    
    }   

}

/// Input training data once file paths and variable names have been set
void gpClassifier::retrieveTrainingData()
{

    // Input file containing training data
    mTrainingFile = new TFile(mTrainingFileName, "update");
    if(!mTrainingFile)
    {
        cout << "gpClassifier::retrieveTrainingData() - " << mTrainingFileName << " does not exist!" << endl;
        return;
    }
    
    // Ensure that the training file is properly formatted
    if(!mTrainingFile->Get(mTrainingTreeName))
    {
        cout << "gpClassifier::retrieveTrainingData() - Training file not correctly formatted!" << endl;
        cout << "gpClassifier::retrieveTrainingData() - No tree named \"" << mTrainingTreeName << "\"!" << endl;
        return;
    }
    else
    {
        mTrainingTree = (TTree*)mTrainingFile->Get(mTrainingTreeName);
    }
    
    // Ensure that the training tree is nonempty
    if(!mTrainingTree->GetEntries())
    {
        cout << "gpClassifier::retrieveTrainingData() - " << mTrainingTreeName << " is empty!" << endl;
        return;
    }
    
    // Ensure that the input and output branches exist
    vector<TString>::iterator it = mInputName.begin();
    
    while(it != mInputName.end())
    {
    
        if(!mTrainingTree->GetBranch(*it))
        {
            cout << "gpClassifier::retrieveTrainingData() - No branch for " << *it << "!  Ignoring variable." << endl;
            it = mInputName.erase(it);
        }
        else
        {
            ++it;
        }
        
    }
    
    if(!mInputName.size()) 
    {
        cout << "gpClassifier::retrieveTrainingData() - No input variables!" << endl;
        return;
    }
        
    if(!mTrainingTree->GetBranch(mOutputName))
    {
        cout << "gpClassifier::retrieveTrainingData() - No branch for target output " << mOutputName << "!" << endl;
        return;
    } 
    
    mDim = mInputName.size();
    mNumPoints = mTrainingTree->GetEntries();
   
}

/// Prepare training data: allocate memory for basic data members
/// and transfer data from tree to local memory while standarizing
/// the input variables
void gpClassifier::prepareTrainingData()
{

    //////////////////////////////////////////////////
    //                Check readiness               //
    //////////////////////////////////////////////////
    
    if(!mKernel)
    {
        cout << "gpClassifier::prepareTrainingData() - No kernel!" << endl;
        return;
    }
    
    mNumHyperParameters = mKernel->nHyperParameters();
    mHyperParameters = mKernel->hyperParameters();

    //////////////////////////////////////////////////
    //                 Allocate memory              //
    //////////////////////////////////////////////////
    
    int nMatrix = mNumPoints * mNumPoints;

    // Array of data
    mX = new double[mNumPoints * mDim];
    
    // Array of targets
    mY = new double[mNumPoints];

    // Inverse of the Covariance Matrix
    mCinverse = new double[nMatrix];
    
    // Normalization parameters
    mMean = new double[mDim];
    mSigma = new double[mDim];
    
    // Auxillery arrays
    mB = new double[mNumPoints];
    mK = new double[mNumPoints];
    
    // Temporary array to hold variables for each case
    double *variable;
    variable = new double[mDim];
    
    double target = 0;

    //////////////////////////////////////////////////
    //       Transfer data from tree to arrays      //
    //////////////////////////////////////////////////
    
    // Set addresses for input variables
    for(int m = 0; m < mDim; ++m)
    {
        mTrainingTree->SetBranchAddress(mInputName.at(m), variable + m);   
    }
    // Set address for target
    mTrainingTree->SetBranchAddress(mOutputName, &target); 
    
    // Calculate sample mean and variance of the training data
    for(double* ptr = mMean; ptr != mMean + mDim; ++ptr) *ptr = 0;
    for(double* ptr = mSigma; ptr != mSigma + mDim; ++ptr) *ptr = 0;
    
    double* meanPtr;
    double* sigmaPtr;
    double* varPtr;
    for(int i = 0; i < mNumPoints; ++i)
    {
        
        mTrainingTree->GetEntry(i);
        
        meanPtr = mMean;
        sigmaPtr = mSigma;
        varPtr = variable;
        for(int j = 0; j < mDim; ++j)
        {
        
            *meanPtr += *varPtr;
            *sigmaPtr += (*varPtr) * (*varPtr);
            ++meanPtr;
            ++sigmaPtr;
            ++varPtr;
            
        }
        
    }
    
    meanPtr = mMean;
    sigmaPtr = mSigma;
    for(int j = 0; j < mDim; ++j)
    {
        *meanPtr  /= mNumPoints;
        *sigmaPtr /= mNumPoints;
        *sigmaPtr -= (*meanPtr) * (*meanPtr);
        *sigmaPtr = sqrt(*sigmaPtr);
        ++meanPtr;
        ++sigmaPtr;
    }
    
    // Fill data arrays from the training trees, normalizing so that 
    // the sample mean vanishes and the sample variance is unity
    double *xPtr = mX;
    double *yPtr = mY;
    
    // Loop over entries
    for(int i = 0; i < mNumPoints; ++i)
    {
        
        mTrainingTree->GetEntry(i);
        
        // Loop over variables
        meanPtr = mMean;
        sigmaPtr = mSigma;
        varPtr = variable;
        for(int j = 0; j < mDim; ++j)
        {
            *xPtr = (*varPtr - *meanPtr) / *sigmaPtr;
            ++xPtr;
            ++meanPtr;
            ++sigmaPtr;
            ++varPtr;
        }
        
        *yPtr = target;
        ++yPtr;
        
    }
    
    // Clean up the now extraneous variable array
    delete[] variable;
    
    //////////////////////////////////////////////////
    //       Retrieve stored hyperparameters, if    //
    //       if they exist in the training file     //
    //////////////////////////////////////////////////
    
    TTree* hyperTree;
    double hyperStore;
    
    TString hyperTreeName = "hyper";
    hyperTreeName += mKernel->name();
    
    // Set address for the hyperparameters...
    if(mTrainingFile->Get(hyperTreeName))
    {
    
        hyperTree = (TTree*)mTrainingFile->Get(hyperTreeName);
        hyperTree->SetBranchAddress("h", &hyperStore);
        
        // Fill hyperparameter array
        double* hyperPtr = mHyperParameters;
        for(int n = 0; n < mNumHyperParameters; ++n)
        {
            hyperTree->GetEntry(n);
            *hyperPtr = hyperStore;
            ++hyperPtr;
        }
        
        mGoHyperParameters = 1;
        
    }
    
    //////////////////////////////////////////////////
    //            Display final settings            //
    //////////////////////////////////////////////////
    
    cout << "Instantiating a gpClassifer with a " << flush;
    cout << mKernel->title() << " covariance function." << endl;
    cout << "Training data taken from " << mTrainingFileName << endl;
    cout << endl;
    cout << "Input Variables:" << endl;
    for(int m = 0; m < mDim; ++m)
    {
        cout << "\t" << mInputName.at(m) << endl;
    }
    cout << "Output Variable:" << endl;
    cout << " \t" << mOutputName << endl;
    cout << endl;

    mReady = 1;    

    return;

}

/// Learn the hyperparameters from the data by optimizing
/// the log model evidence with multiple conjugate gradient
/// searches from randomized initial points, 
/// storing the hyperparameters with the largest evidence.
/// \param nTrainingIterations Number of searches
/// \param goVerbose Switch for verbose training output

void gpClassifier::train(int nTrainingIterations, bool goVerbose)
{
    
    cout << "Commence training..." << endl;
    
    // Ensure that the classifier is ready
    if(!mReady)
    {
        cout << "gpClassifier::train() - Training data has not been loaded." << endl;
        cout << "gpClassifier::train() - Have you called gpClassifier::retrieveTrainingData() yet?" << endl;
        return;
    }
    
    // Best performing hyperparameters, plus an added entry for the log model evidence
    double *bestHyperparameters;
    bestHyperparameters = new double[mNumHyperParameters + 1];
    
    // Allocate memory for the arrays needed by the hyperparameter optimization
    mdCdh = new double[mNumPoints * mNumPoints];
    mH1 = new double[mNumHyperParameters];
    mS = new double[mNumHyperParameters];
    mD = new double[mNumHyperParameters];
    mG1 = new double[mNumHyperParameters];
    mG2 = new double[mNumHyperParameters];
    mGi = new double[mNumHyperParameters];
    
    // Fire up the random number generator
    TRandom3 random(0);

    /////////////////////////////////////////////////////
    //    Optimize the hyperparameters from the data   //
    //         with a conjugate gradient search        //
    /////////////////////////////////////////////////////
    
    for(int i = 0; i < nTrainingIterations; ++i)
    {
        
        // Randomly sample a reasonable seed for the hyperparameter optimization
        mLogEvidence = mMinInitLogEvidence > 0 ? 0.5 * mMinInitLogEvidence : 2.0 * mMinInitLogEvidence;
        
        if(goVerbose) 
        {
            cout << "Sampling an initial search point with logEvidence < "
                 << mMinInitLogEvidence << flush;
        }
        
        while(mLogEvidence < mMinInitLogEvidence)
        {
        
            if(goVerbose) cout << "." << flush;
        
            // Randonmly sample the hyperparameters
            for(int n = 0; n < mNumHyperParameters; ++n)
            {
                mHyperParameters[n] = random.Rndm();
            }
            
            // Calculate performance given the sampled hyperparameters
            covariance(); invert();
            mLogEvidence = logEvidence();
            
        }

        if(goVerbose) cout << endl << "Beginning search with the logEvidence " << mLogEvidence << endl;
        
        // Perform the conjugate gradient
        fSearch(goVerbose);
        
        // Calculate performance given the learned hyperparameters
        covariance(); invert();
        mLogEvidence = logEvidence();
        
        // Display results of the conjugate gradient search
        cout << endl;
        cout << "Training Iteration " << i + 1 
             << ", log(Evidence) = " << mLogEvidence << endl;
        mKernel->displayHyperParameters("\t");
        
        // Store results if this is the first search...
        if(i == 0)
        {
            memcpy(bestHyperparameters, mHyperParameters, mNumHyperParameters * sizeof(double));
            bestHyperparameters[mNumHyperParameters] = mLogEvidence;
        
        }
        // or if the results improve upon previous searches
        else
        {
            
            if(mLogEvidence > bestHyperparameters[mNumHyperParameters])
            {
             
                memcpy(bestHyperparameters, mHyperParameters, mNumHyperParameters * sizeof(double));
                bestHyperparameters[mNumHyperParameters] = mLogEvidence;
                
            }
            
        }
        
    }
    
    cout << endl;
    
    // Use the best hyperparameters found
    memcpy(mHyperParameters, bestHyperparameters, mNumHyperParameters * sizeof(double));
    mLogEvidence = bestHyperparameters[mNumHyperParameters];

    mTrained = true;
    mGoHyperParameters = true;
    storeHyperParameters();
    
    /////////////////////////////////////////////////////
    //                     Clean up                    //
    /////////////////////////////////////////////////////
    
    delete[] bestHyperparameters;
    delete[] mdCdh;
    delete[] mH1;
    delete[] mS;
    delete[] mD;
    delete[] mG1;
    delete[] mG2;
    delete[] mGi;
    
    return;

}

/// Learn the hyperparameters from the data by optimizing
/// the log model evidence with a conjugate gradient
/// search initialized with the given hyperparameters.
/// \param hyperStart Hyperparameters for initializing the search
/// \param goVerbose Switch for verbose training output

void gpClassifier::train(double *hyperStart, bool goVerbose)
{
    
    cout << "Commence training..." << endl;
    
    // Ensure that the classifier is ready
    if(!mReady)
    {
        cout << "gpClassifier::train() - Training data has not been loaded." << endl;
        cout << "gpClassifier::train() - Have you called gpClassifier::retrieveTrainingData() yet?" << endl;
        return;
    }
    
    // Allocate memory for the arrays needed by the hyperparameter optimization
    mdCdh = new double[mNumPoints * mNumPoints];
    mH1 = new double[mNumHyperParameters];
    mS = new double[mNumHyperParameters];
    mD = new double[mNumHyperParameters];
    mG1 = new double[mNumHyperParameters];
    mG2 = new double[mNumHyperParameters];
    mGi = new double[mNumHyperParameters];
    
    /////////////////////////////////////////////////////
    //    Optimize the hyperparameters from the data   //
    //         with a conjugate gradient search        //
    /////////////////////////////////////////////////////
    
    // Set the hyperparameters to the given values
    for(int n = 0; n < mNumHyperParameters; ++n)
    {
        mHyperParameters[n] = hyperStart[n];
    }
    
    // Calculate performance given the sampled hyperparameters
    covariance(); invert();
    mLogEvidence = logEvidence();

    cout << "Beginning search with the logEvidence " << mLogEvidence << endl;
        
    // Perform the conjugate gradient
    fSearch(goVerbose);
        
    // Calculate performance given the learned hyperparameters
    covariance(); invert();
    mLogEvidence = logEvidence();
        
    // Display results of the conjugate gradient search
    cout << "Finishing search with the logEvidence " << mLogEvidence << endl;
    mKernel->displayHyperParameters("\t");

    mTrained = true;
    mGoHyperParameters = true;
    storeHyperParameters();
    
    /////////////////////////////////////////////////////
    //                     Clean up                    //
    /////////////////////////////////////////////////////
    
    delete[] mdCdh;
    delete[] mH1;
    delete[] mS;
    delete[] mD;
    delete[] mG1;
    delete[] mG2;
    delete[] mGi;
    
    return;

}

/// Use given hyperparameters in lieu of training
/// \param hyperParameters Pointer to desired hyperparameter array
void gpClassifier::useHyperParameters(double* hyperParameters)
{

    memcpy(mHyperParameters, hyperParameters, mNumHyperParameters * sizeof(double));
    mGoHyperParameters = 1;
    storeHyperParameters();

}

/// Store the current hyperparameters in the training file
void gpClassifier::storeHyperParameters()
{

    if(!mReady)
    {
        cout << "gpClassifier::train() - Training data has not been loaded." << endl;
        cout << "gpClassifier::train() - Have you called gpClassifier::retrieveTrainingData() yet?" << endl;
        return;
    }
    
    if(!mGoHyperParameters)
    {
        cout << "gpClassifier::storeHyperParameters() - Hyperparameters have not been set!" << endl;
        return;
    }

    TTree* hyperTree;
    double hyperStore;
    bool hyperDelete = 0;
    
    TString hyperTreeName = "hyper";
    hyperTreeName += mKernel->name();
    
    // Set address for the hyperparameters if the tree already exists
    if(mTrainingFile->Get(hyperTreeName))
    {
    
        hyperTree = (TTree*)mTrainingFile->Get(hyperTreeName);
        hyperTree->SetBranchAddress("h", &hyperStore);
        
    }
    // Otherwise create a new tree
    {
        hyperTree = new TTree(hyperTreeName, "Hyperparameters");
        hyperTree->Branch("h", &hyperStore, "h/D");
        hyperDelete = 1;
    }
        
    // Empty the tree
    hyperTree->Reset();
        
    // Fill hyperparameter array
    for(double* hyperPtr = mHyperParameters; hyperPtr != mHyperParameters + mNumHyperParameters; ++hyperPtr)
    {
        hyperStore = *hyperPtr;
        hyperTree->Fill();
    }
    
    hyperTree->Write();
    
    if(hyperDelete) hyperTree->Delete();
    
    mGoHyperParameters = true;

}


/// Display relevance of each variable
void gpClassifier::displayRelevance()
{

    // Check if kernel is present
    if(!mKernel) 
    {
        cout << "gpClassifier::displayRelevance() - No kernel!" << endl; 
        return; 
    }
    
    // Check if ARD is applicable
    if(!mKernel->ARD())
    {
        cout << "gpClassifier::displayRelevance() - ARD not implemented for the " 
             << mKernel->title() << " covariance function." << endl;
        return;
    }
    
    // Ensure that the classifier is ready
    if(!mGoHyperParameters)
    {
        cout << "gpClassifier::displayRelevance() - Hyperparameters have not been set!" << endl;
        return;
    }
    
    // Format input variable names, buffering with whitespace to force consistent column spacing
    int bufferLength = 0;
    for(int i = 0; i < mDim; ++i)
    {
        int length = mInputName.at(i).Length();
        bufferLength = length > bufferLength ? length : bufferLength;
    }
    
    for(int i = 0; i < mDim; ++i)
    {
        int whiteSpace = bufferLength - mInputName.at(i).Length();
        mInputName.at(i).Append(' ', whiteSpace);
    }
    

    // Sort input variables by relevance
    map<double, int> relevance;
    
    int start = mNumHyperParameters - mDim;
    for(int m = start; m < mNumHyperParameters; ++m)
    {
        double r = mHyperParameters[m] * mHyperParameters[m];
        relevance[r] = m;
    }

    // Output
    cout << "Initiating automatic relevance determination with " << mKernel->displayRelevance() << endl;
    cout << endl;
    
    TString title = "Variable";
    title.Append(' ', bufferLength - title.Length());

    cout << "\t" << title << "\tr_{i}" << endl;
    map<double, int>::reverse_iterator rit;
    for(rit=relevance.rbegin(); rit != relevance.rend(); ++rit)
    {
        cout << "\t" << mInputName.at(rit->second - start) 
             << "\t" << rit->first << endl;
    }
    cout << endl;
    
    // Remove buffering whitespace
    for(int i = 0; i < mDim; ++i) mInputName.at(i).ReplaceAll(" ", "");

    return;

}

/// Retrieve stored training results
void gpClassifier::prepareClassifier()
{

    cout << "Readying the classifier." << endl << endl;

    // Ensure that a valid file has been loaded
    if(!mReady)
    {
        cout << "gpClassifier::prepareClassifier() - Training data has not been loaded." << endl;
        cout << "gpClassifier::prepareClassifier() - Have you called gpClassifier::retrieveTrainingData() yet?" << endl;
        return;
    }
    if(!mGoHyperParameters)
    {
        cout << "gpClassifier::prepareClassifier() - No hyperparameters have been set!" << endl;
        return;
    }

    time_t start;
    time_t end;
    
    time(&start);
    covariance();
    time(&end);
    cout << "\tTook " << difftime(end, start) << " seconds "
         << "to build the covariance matrix." << endl;
    
    time(&start);
    invert();
    time(&end);
    cout << "\tTook " << difftime(end, start) << " seconds "
         << "to invert the covariance matrix." << endl;
    
    if(!mTrained) 
    {
        mLogEvidence = logEvidence();
        mTrained = true;
    }
    
    // Display model hyperparameters
    cout << "\tUsing the hyperparameters:" << endl;
    mKernel->displayHyperParameters("\t\t");
    cout << "\twith the log evidence = " << mLogEvidence << endl;
    cout << endl;
    
}

/// Compute the mean and variance of the gaussian process
/// output distribution for a single point
/// \param xtest Pointer to test point array
/// \param mean Address where the output mean will be written
/// \param variance Address where the output variance will be written

void gpClassifier::testPoint(double* xtest, double& mean, double& variance)
{

    // Ensure training data has been loaded
    if(!mTrained) 
    {
    
        cout << "gpClassifier::testPoint() - Classifier not ready!" << endl;
        cout << "gpClassifier::testPoint() - \tHave you called gpClassifier::prepareClassifier() yet?" << endl;
        
        mean = 0.0;
        variance = -1.0;
        
        return;
        
    }
    
    // Normalize test point
    double* meanPtr = mMean;
    double* sigmaPtr = mSigma;
    double* xPtr = xtest;
    
    for(int j = 0; j < mDim; ++j)
    {
        *xPtr = (*xPtr - *meanPtr) / *sigmaPtr;
        ++xPtr;
        ++meanPtr;
        ++sigmaPtr;
    }
    
    // Test point
    predict(xtest, mean, variance);
    
}

/// Compute the mean and variance of the gaussian process
/// output distribution for all point in a given file,
/// storing the results in new branches attached of testTreeName.
/// \param testFileName Name of input file
/// \param testTreeName Name of input tree

void gpClassifier::processSample(TString testFileName, TString testTreeName)
{
 
    cout << "Processing the file " << testFileName << endl;
 
    if(testTreeName == "") testTreeName = mTrainingTreeName;
 
    // Ensure that training results have been loaded
    if(!mTrained) 
    {
        cout << "gpClassifier::processSample() - Classifier not ready!" << endl;
        cout << "gpClassifier::processSample() - Have you called gpClassifier::prepareClassifier() yet?" << endl;
        return;
    }
    
    //////////////////////////////////////////////////
    //              Input testing data              //
    //////////////////////////////////////////////////
    
    TFile testFile(testFileName, "update");
    
    // Ensure that the testing tree exists
    TTree* t = (TTree*)testFile.Get(testTreeName);
    if(!t)
    {
        cout << "gpClassifier::processSample() - " << testTreeName 
             << " is not a valid tree in " << testFileName << endl;
        return;
    }
    
    // Ensure the testing data has the same branches as the training data
    for(int m = 0; m < mDim; ++m)
    {
        if(!t->GetBranch(mInputName.at(m)))
        {
            cout << "gpClassifier::processSample() - " << testTreeName << " does not have a branch for "
                 << "the input variable " << mInputName.at(m) << endl;
            return;
        }
    }
    
    // Ensure that branches for gpMean and gpVariance don't already exist
    for(int m = 0; m < t->GetListOfBranches()->GetEntries(); ++m)
    {

        if(t->GetListOfBranches()->At(m)->GetName() == "gpMean")
        {
            cout << "gpClassifier::processSample() - The branch \"gpMean\" "
                 << "must be reserved for the gaussian process output." << endl;
        }

        if(t->GetListOfBranches()->At(m)->GetName() == "gpVariance")
        {
            cout << "gpClassifier::processSample() - The branch \"gpVariance\" "
                 << "must be reserved for the gaussian process output." << endl;
        }
        
    }
    
    // Instantiate array to hold each test point
    double* xtest;
    xtest = new double[mDim];
    
    // Set addresses
    for(int m = 0; m < mDim; ++m) t->SetBranchAddress(mInputName.at(m), xtest + m);
    
    // Create branches for the predictive mean and variance
    double gpMean = 0;
    double gpVariance = 0;

    TBranch* meanBranch = t->Branch("gpMean", &gpMean, "gpMean/D");
    TBranch* varBranch = t->Branch("gpVariance", &gpVariance, "gpVariance/D");
    
    cout << "Processing " << testFileName << " - " << endl;
    cout << "\tStoring the gaussian process predictive mean and variance " << endl;
    cout << "\tin the branches \"gpMean\" and \"gpVariance\", respectively." << endl;
    
    //////////////////////////////////////////////////
    //   Calculate predictions for each test point  //
    //////////////////////////////////////////////////

    for(int n = 0; n < t->GetEntries(); ++n)
    {
        
        t->GetEntry(n);
        
        testPoint(xtest, gpMean, gpVariance);
         
        meanBranch->Fill();
        varBranch->Fill();

    }
    
    t->Write("", TObject::kOverwrite);
    
    //////////////////////////////////////////////////
    //                   Clean up                   //
    //////////////////////////////////////////////////
        
    delete[] xtest;
    
    return;
    
}
