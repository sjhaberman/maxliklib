//Log likelihood component and its gradient and hessian matrix
//for model with right censoring.  Response is y, and parameter
//vector beta has elements beta(1) and beta(2)>0.  Here y.iresp(0)
//is 0 for no censoring and 1 for censoring.  Without censoring,
//y.dresp(0) is the original variable.  With censoring, y.dresp(0) is the
//censoring value.  The model is determined by transform.  Here 'G' is for
//gumbel, 'L' is for logistic, and 'N' is for normal.
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
f2v contresp(const char &, const vec & y, const vec & beta);
f2v berresp(const char & , const ivec & y, const vec & beta);
f2v truncresp(const char & transform, const resp & y, const vec & beta)
{
    ivec i={0};
    vec gamma(1);
    f2v remainder,result;   
    result.grad.set_size(2);
    result.hess.set_size(2,2);
    if(y.iresp(0)>0)
    {
       gamma(0)=beta(0)+y.dresp(0)*beta(1);
       remainder.grad.set_size(1);
       remainder.hess.set_size(1,1);
       remainder=berresp(transform,i,gamma);
       result.value=remainder.value;
       result.grad(0)=remainder.grad(0);
       result.grad(1)=beta(1)*result.grad(0);
       result.hess(0,0)=remainder.hess(0,0);
       result.hess(0,1)=result.hess(0,0)*beta(1);
       result.hess(1,0)=result.hess(0,1);
       result.hess(1,1)=beta(1)*result.hess(1,0);
    }
    else
    {
        result=contresp(transform,y.dresp,beta);
    }
    return result;	
}

