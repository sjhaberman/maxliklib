//Log likelihood component and its first and second derivative
//for Poisson log-linear model with response y and parameter beta.

#include<armadillo>
using namespace arma;
struct fd2
{
    double value;
    double der1;
    double der2;
};


fd2 logmean(int y,double beta)
{
    double fy,mu;
    fd2 results;
    fy=double(y);
    mu=exp(beta);
    results.value=fy*beta-mu;
    results.der1=y-mu;
    results.der2=-mu;
    return results;
}
