#include "gpKernel.h"

using namespace std;

/// Constructor
/// \param ARD Flag for automatic relevance detection in the inherited covariance function
/// \param title Title of the covariance function
/// \param name Name of the covariance function

gpKernel::gpKernel(bool ARD, TString title, TString name): mARD(ARD), mTitle(title), mName(name) {}

