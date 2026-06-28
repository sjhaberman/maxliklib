//Log likelihood component and its gradient and hessian matrix
//for model with right censoring.  Response is y, and parameter
//vector beta has elements beta(0) and beta(1). Here y(1)
//is 0 for no censoring and 1 for censoring.  Without censoring,
//y(0) is the original variable.  With censoring, y(0) is the
//censoring value.  The model is determined by transform.  Here 'G' is for
//gumbel, 'L' is for logistic, and 'N' is for normal.
//If order is 0, only the function is
//found, if order is 1, then the function and gradient are found.
//If order is 2, then the function, gradient, and Hessian are returned.
//Models have y(0)-beta(0))/exp(beta(1))) with a standard minimum Gumbel,
//maximum Gumbel, logistic, or normal distribution.
#include<armadillo>
using namespace arma;
struct f2v{double value; vec grad; mat hess;};
f2v contresp(const int & , const char &, const vec & , const vec & );
f2v berresp(const int & , const char & , const vec & , const vec & );
f2v truncresp(const int & order,  const char & transform,
    const vec & y, const vec & beta){
    double x;
    vec z(1);
    z(0)=0.0;
    vec gamma(1);
    f2v remainder,result;
    if(y(1)>0.0){
        x=exp(beta(1));
        gamma(0)=beta(0)+y(0)*x;
        remainder=berresp(order, transform, z, gamma);
        result.value=remainder.value;
        if(order==0) return result;
        result.grad.set_size(2);
        result.grad(0)=remainder.grad(0);
        result.grad(1)=x*result.grad(0);
        if(order==1)return result;
        result.hess.set_size(2,2);
        result.hess(0,0)=remainder.hess(0,0);
        result.hess(0,1)=result.hess(0,0)*x;
        result.hess(1,0)=result.hess(0,1);
        result.hess(1,1)=result.grad(1)+x*result.hess(1,0);
        return result;
    }
    else return contresp(order, transform, y, beta);
}

