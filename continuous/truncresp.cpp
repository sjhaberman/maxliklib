//Log likelihood component and its gradient and hessian matrix
//for model with right censoring.  Response is y, and parameter
//vector beta has elements beta(1) and beta(2)>0.  Here y.iresp(0)
//is 0 for no censoring and 1 for censoring.  Without censoring,
//y.dresp(0) is the original variable.  With censoring, y.dresp(0) is the
//censoring value.  The model is determined by transform.  Here 'G' is for
//gumbel, 'L' is for logistic, and 'N' is for normal.
//If order is 0, only the function is
//found, if order is 1, then the function and gradient are found.
//If order is 2,
//then the function, gradient, and Hessian are returned.
#include<armadillo>
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
struct resp
{
    ivec iresp;
    vec dresp;
};
f2v contresp(const int & , const char &, const resp & , const vec & );
f2v berresp(const int & , const char & , const resp & , const vec & );
f2v truncresp(const int & order,  const char & transform, const resp & y, const vec & beta)
{
    resp z;
    z.iresp.set_size(1);
    z.iresp(0)=0;
    vec gamma(1);
    f2v remainder,result;   
    if(order>0) result.grad.set_size(2);
    if(order>1) result.hess.set_size(2,2);
    if(y.iresp(0)>0)
    {
        gamma(0)=beta(0)+y.dresp(0)*beta(1);
        if(order>0) remainder.grad.set_size(1);
        if(order>1) remainder.hess.set_size(1,1);
        remainder=berresp(order, transform, z, gamma);
        result.value=remainder.value;
        if(order>0)
        {
            result.grad(0)=remainder.grad(0);
            result.grad(1)=beta(1)*result.grad(0);
        }
        if(order>1)
        {
            result.hess(0,0)=remainder.hess(0,0);
            result.hess(0,1)=result.hess(0,0)*beta(1);
            result.hess(1,0)=result.hess(0,1);
            result.hess(1,1)=beta(1)*result.hess(1,0);
        }
    }
    else
    {
        result=contresp(order, transform, y, beta);
    }
    return result;	
}

