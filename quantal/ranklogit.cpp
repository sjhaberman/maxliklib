//Log likelihood component, gradient, and hessian matrix
//for model for ranks based on latent Gumbel distributions.
//There are r objects to be
//ranked and parameter
//vector beta has dimension r-1.  The analysis
//considers the k largest objects for a
//positive integer k no greater than r-1.
//The vector y provides the ranking.
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
f2v ranklogit(const int & order, const resp & y, const vec & beta)
{
    double s;
    int i,k,rr;
    vec f;
    f2v results;
    rr=beta.n_elem;
    f.set_size(rr);
    k=y.iresp.n_elem;
    results.value=0.0;
    if(order>0)
    {
        results.grad.set_size(rr);
        results.grad.zeros();
    }
    if(order>1)
    {
        results.hess.set_size(rr,rr);
        results.hess.zeros();
    }
    for(i=0;i<rr;i++)
    {
        f(i)=exp(beta(i));
    }
    s=1.0+sum(f);
    for(i=0;i<k;i++)
    {
        results.value=results.value-log(s);
        if(order>0) results.grad=results.grad-f/s;
        if(order>1) results.hess=results.hess-diagmat(f)/s+f*trans(f)/(s*s);
        if(y.iresp(i)>0)
        {
            results.value=results.value+beta(y.iresp(i)-1);
            if(order>0) results.grad(y.iresp(i)-1)=results.grad(y.iresp(i)-1)+1.0;
            s=s-f(y.iresp(i)-1);
            f(y.iresp(i)-1)=0.0;
        }
        else
        {
            s=s-1.0;
        }
    }
    return results;
}
