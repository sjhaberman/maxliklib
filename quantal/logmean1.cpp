//Log likelihood component and its first aderivative
//for Poisson log-linear model with response y and parameter beta.

#include<armadillo>
using namespace arma;
struct fd1
{
    double value;
    double der1;
    
};


fd1 logmean1(int y,double beta)
{
    double fy,mu;
    fd1 results;
    fy=double(y);
    mu=exp(beta);
    results.value=fy*beta-mu;
    results.der1=fy-mu;
   
    return results;
}
