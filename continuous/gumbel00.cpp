//Value and fderivativs for logarithm of standard gumbel density.

#include<cmath>




double gumbel00(double y)
{
    double results;
    double p;
    p=exp(-y);
    results=-y-p;
    
    
    return results;
}
