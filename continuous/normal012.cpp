//Value and first two derivatives for logarithm of standard normal density.

#include<armadillo>
using namespace arma;
struct fd2
{
    double value;
    double der1;
    double der2;
    
};


fd2 normal012(double y)
{
    
    fd2 results;
        
    results.value=-0.5*log(2.0*datum::pi)-0.5*y*y;
    results.der1=-y;
    results.der2=-1.0;
    return results;
}
