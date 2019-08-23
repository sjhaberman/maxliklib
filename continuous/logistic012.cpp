//Value and first two derivatives for logarithm of standard normal density.

#include<armadillo>
using namespace arma;
struct fd2
{
    double value;
    double der1;
    double der2;
    
};


fd2 logistic012(double y)
{
    
    fd2 results;
    double p,q,w;
    p=exp(-y);
    p=p/(1.0+p);
    q=1.0-p;
    w=p*q;
    results.value=log(w);
    results.der1=p-q;
    results.der2=-2.0*w;
    return results;
}
