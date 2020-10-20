//Log likelihood component, gradient, and Hessian
//for cumulative logit model with response y and parameter beta.
#include<armadillo>
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
f2v cumlogit(ivec & y,vec & beta)
{
    double p,q;
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
        p=1.0/(1.0+exp(-beta(i)));
        q=1.0-p;
        results.hess(i,i)=-p*q;
        if(i<y(0))
        {
          results.value=results.value+log(p);
          results.grad(i)=q;
        }
        else
        {
            results.value=results.value+log(q);
            results.grad(i)=-p;
            break;
        }
    }
    return results;
}
