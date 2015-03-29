// C++ stdlib
#include <iostream>
#include "math.h"

// ROOT Libraries
#include "TRandom3.h"

// Local Libraries
#include "betaChain.h"
#include "binomialNest.h"

using namespace std;

int main (int argc, char * const argv[]) 
{

    // Fire up the random number generator
    TRandom3 random(0);

    // Truth
    double mu = 0.7;
    int N = 1000;
    int k = 0;
    for(int i = 0; i < N; ++i) if(random.Rndm() < mu) ++k;
    double muHat = (double)k / (double)N;
    
    cout << "Creating toy model with " << endl;
    cout << "\tmu = " << mu << endl;
    cout << "\tk = " << k << endl;
    cout << "\tN = " << N << endl;
    cout << "\tmu_{hat} = " << muHat << endl;

    // Prior
    double alpha = 2;
    double beta = 2;

    // Created objects
    int nChains = 10;

    binomialNest nest(k, N, &random);
    nest.setVerbosity(true);
    nest.setStepSize(0.5);
    nest.setStepSizeJitter(0.001);
    nest.setMaxStepSize(1.0);
    nest.setMinStepSize(1e-8);
    
    for(int i = 0; i < nChains; ++i) nest.addChain(new betaChain(alpha, beta));
    nest.seed(0.1, 0.9);
    nest.setBurnParameters(400, 200, 1.2);
    nest.burn();

    nest.setNumNests(1000);
    nest.setNestThreshold(1e-3);
    nest.setDisplayFrequency(10);
    nest.setLikelihoodConstraint(true);
    nest.nestedSample();
    
    //nest.computeMeanExpectations();
    //nest.computeSampledExpectations(20);
    
    cout << endl;
    
    double logE = 0;
    for(int i = N - k + 1; i < N + 1; ++i) logE += log(i);
    for(int i = 1; i < k + 1; ++i) logE -= log(i);
    for(int i = alpha; i < alpha + k ; ++i) logE += log(i);
    for(int i = beta; i < beta + N - k; ++i) logE += log(i);
    for(int i = alpha + beta; i < alpha + beta + N; ++i) logE -= log(i);
    
    cout << "TrueLogZ = " << logE << endl;
    
    return 0;

    
}
