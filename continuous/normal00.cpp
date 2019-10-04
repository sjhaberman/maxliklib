//Value for logarithm of standard normal density.

#include<armadillo>
using namespace arma;


double normal00(double y)
{
    
    double results;
        
    results=-0.5*log(2.0*datum::pi)-0.5*y*y;
    
    
    return results;
}
