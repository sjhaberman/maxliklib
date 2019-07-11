//Log likelihood component, gradient, and hessian matrix
//for multinomial logit model with response y from 0 to r-1 and parameter
//vector beta of dimension r.
#include<armadillo>
using namespace arma;

struct fd2v
{
    double value;
    vec grad;
    mat hess;
};



fd2v multlogit(int y,vec beta)
{
//Probability of response of 1.
    double r;
    
    vec e;
    fd2v results;
    e=exp(beta);
    
    r=sum(e);
    
    
    results.value=beta(y)-log(r);
   
    results.grad=-e/r;
    
   
    results.hess=diagmat(results.grad)+results.grad*trans(results.grad);
   
    
    
    
    results.grad(y)=1.0+results.grad(y);
    
    
    return results;
}
