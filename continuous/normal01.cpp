//Value and first derivative for logarithm of standard normal density.

#include<armadillo>
using namespace arma;
struct fd1
{
    double value;
    double der1;
   
    
};


fd1 normal01(double y)
{
    
    fd1 results;
        
    results.value=-0.5*log(2.0*datum::pi)-0.5*y*y;
    results.der1=-y;
    
    return results;
}
