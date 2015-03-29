#ifndef _BETA_BAYES_HIST_COMP_

class TH1;
class TGraph;
class TString;


/// \mainpage Bayesian Histogram Comparison
///
/// Implementation of a Bayesian approach to the comparison of two histograms, 
/// providing the results from both model comparison as well as a mixture
/// model.  Both approaches admit histograms generated from independent events
/// as well as importance samping, allowing for the comparison of histograms
/// generated from data or simulation.
///
/// A ROOT (http://root.cern.ch/drupal/) installation is required.

/// \example main.cpp

///
/// \author Michael Betancourt,
/// Massachusetts Institute of Technology
///
/// Implementation of a Bayesian approach to the comparison of two histograms, 
/// providing the results from both model comparison as well as a mixture
/// model.  Both approaches admit histograms generated from independent events
/// as well as importance samping, allowing for the comparison of histograms
/// generated from data or simulation.
///
/// Note that the algorithms accept one, two, and three dimensional histograms.

class bayesHistComp
{

    public:
    
        bayesHistComp() { bayesHistComp(100); }; ///< Constructor with default mixture discretization
        bayesHistComp(int nPiPoints);
        ~bayesHistComp();
        
        // Accessors
        double credibility() { return mCredibility; } ///< Return most recent credibility calculation
        double MAP() { return mPiMAP; } ///< Return position of the posterior mode
        double modelOdds() { return mModelOdds; } ///< Return the posterior probability of "same" in the non-mixture model
        
        double* mixturePosterior() { return mP; } ///< Return pointer to the mixture posterior array
        int nMixturePoints() { return mNumPiPoints; } ///< Return size of mixture posterior array        
        
        // Mutators
        void setVerbosity(bool flag = true) { mGoVerbose = flag; } ///< Set verbosity setting, defaults to true
        void skipEmptyBins() { mSkipEmptyBins = true; } ///< Skip bins when both histograms are empty
        
        // Comparison methods
        void compareDataData(TH1 *dataOne, TH1 *dataTwo);
        void compareDataData(TH1 *dataOne, TH1 *dataTwo, TGraph *posterior);
        void compareDataData(TH1 *dataOne, TH1 *dataTwo, TString imageName);
        
        void compareDataSimu(TH1 *data, TH1 *simuWeight, TH1 *simuNoWeight);
        void compareDataSimu(TH1 *data, TH1 *simuWeight, TH1 *simuNoWeight, TGraph *posterior);
        void compareDataSimu(TH1 *data, TH1 *simuWeight, TH1 *simuNoWeight, TString imageName);
        
        void compareSimuSimu(TH1 *simuOneW, TH1 *simuOneNoW, TH1 *simuTwoW, TH1 *simuTwoNoW);
        void compareSimuSimu(TH1 *simuOneW, TH1 *simuOneNoW, TH1 *simuTwoW, TH1 *simuTwoNoW, TGraph *posterior);
        void compareSimuSimu(TH1 *simuOneW, TH1 *simuOneNoW, TH1 *simuTwoW, TH1 *simuTwoNoW, TString imageName);
        
    private:
    
        /// Enumeration of possible comparisons
        enum histModel { kDataData = 0, kDataSimu, kSimuSimu };
    
        bool mGoVerbose; ///< Verbosity flag
    
        bool mSkipEmptyBins; ///< Skip empty bins flag
    
        int mNumPiPoints; ///< Number of points in the discretization of pi
        double* mPi; ///< Discretization of pi
        double* mP; ///< Posterior of pi given the histograms
        
        double mPiMAP; ///< Position of the posterior mode
        double mCredibility; ///< Posterior mass above pi = 1/2
        double mModelOdds; ///< Posterior probability of "same" in the non-mixture model
        
        int mNumBins; ///< Number of histogram bins
        
        TH1 *mHistOne; ///< Utility pointer to first histogram (weighted if simulation)
        TH1 *mHistOneN; ///< Utility pointer to first histogram (unweighted if simulation)
        
        TH1 *mHistTwo; ///< Utility pointer to second histogram (weighted if simulation)
        TH1 *mHistTwoN; ///< Utility pointer to second histogram (unweighted if simulation)
        
        void fCompareDataData(TH1 *dataOne, TH1 *dataTwo);
        void fCompareDataSimu(TH1 *data, TH1 *simuWeight, TH1 *simuNoWeight);
        void fCompareSimuSimu(TH1 *simuOneW, TH1 *simuOneNoW, TH1 *simuTwoW, TH1 *simuTwoNoW);
        
        double fComputeMixturePosterior(histModel model);
        
        double fComputeLogEvidenceRatio(double m, double n);
        double fComputeLogEvidenceRatio(double m, double s, double nS);
        double fComputeLogEvidenceRatio(double s, double nS, double r, double nR);
        
        void fDraw(TGraph *externalPosterior, TString imageName);
        void fCalcCredibility();
        
        
    //ClassDef(bayesHistComp, 0);

};

#define  _BETA_BAYES_HIST_COMP_
#endif