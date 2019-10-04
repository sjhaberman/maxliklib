//Value for standard logistic distribution function at beta.
#include<cmath>




double logisticf0(double beta)
{

    
    double results;
    results=1.0/(1.0+exp(-beta));
    
    
    return results;
}
