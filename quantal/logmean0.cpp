//Log likelihood component
//for Poisson log-linear model with response y and parameter beta.

#include<armadillo>
using namespace arma;



double logmean0(int y,double beta)
{
    double fy,mu;
    double results;
    fy=double(y);
    mu=exp(beta);
    results=fy*beta-mu;
    
   
    return results;
}
