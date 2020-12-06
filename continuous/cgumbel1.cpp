//Log likelihood component and its gradient
//for gumbel model with censoring.  Response is y, and parameter
//vector beta has elements beta(1) and beta(2)>0.  Here y.iresp(0)
//is 0 for no censoring and 1 for censoring.  Without censoring,
//y.dresp(0) is the original variable.  With censoring, y.dresp(0) is the
//censoring value.
#include<armadillo>
using namespace arma;
struct f1v
{
    double value;
    vec grad;
};
struct resp
{
    ivec iresp;
    vec dresp;
};
f1v gumbel1(vec & y,vec & beta);
f1v loglog1(ivec & y,vec & beta);
f1v cgumbel1(resp & y,vec & beta)
{
    ivec i={0};
    vec gamma(1);
    f1v remainder,result;
    remainder.grad.set_size(1);
    result.grad.set_size(2);
    if(y.iresp(0)>0)
    {
       gamma(0)=beta(0)+y.dresp(0)*beta(1);
       remainder=loglog1(i,gamma);
       result.value=remainder.value;
       result.grad(0)=remainder.grad(0);
       result.grad(1)=beta(1)*result.grad(0);
    }
    else
    {
        result=gumbel1(y.dresp,beta);
    }
    return result;	
}

