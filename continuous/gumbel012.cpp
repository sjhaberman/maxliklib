//Value and first two derivatives for logarithm of standard gumbel density.

#include<armadillo>
using namespace arma;
struct fd2
{
    double value;
    double der1;
    double der2;
    
};


fd2 gumbel012(double y)
{
    
    fd2 results;
    double p;
    p=exp(-y);
    results.value=-y-p;
    results.der1=-1+p;
    results.der2=-p;
    return results;
}
