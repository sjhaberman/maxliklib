//Log likelihood component, gradient, and Hessian
//for cumulative log-log model with response y and parameter
//vector beta.
#include<armadillo>
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
f2v cumloglog(ivec & y,vec & beta)
{
    double p,q,r;
    int i,n;
    f2v results;
    n=beta.n_elem;
    results.value=0.0;
    results.grad.set_size(n);
    results.grad.zeros();
    results.hess.set_size(n,n);
    results.hess.zeros();
    for(i=0;i<n;i++)
    {
        r=exp(-beta(i));
        if(i<y(0))
        {
            results.value=results.value-r;
            results.grad(i)=r;
            results.hess(i,i)=-r;
        }
        else
        {
            p=exp(-r);
            q=1.0-p;
            results.value=results.value+log(q);
            results.grad(i)=-p*r/q;
            results.hess(i,i)=(r-1.0-results.grad(i))*results.grad(i);
            break;
        }
    }
    return results;
}
