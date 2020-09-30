//Log likelihood component, gradient, and Hessian
//for cumulative complementary log-log model with response y and parameter
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
    int i;
    f2v results;
    results.value=0.0;
    results.grad=zeros(beta.n_elem);
    results.hess=zeros(beta.n_elem,beta.n_elem);
    for(i=0;i<beta.n_elem;i++)
    {
        r=exp(beta(i));
        if(i<y(0))
        {
            results.value=results.value-r;
            results.grad(i)=-r;
            results.hess(i,i)=-r;
        }
        else
        {
            q=exp(-r);
            p=1.0-q;
            results.value=results.value+log(p);
            results.grad(i)=q*r/p;
            results.hess(i,i)=(1.0-r-results.grad(i))*results.grad(i);
            break;
        }
    }
    return results;
}
