//Log likelihood component, gradient, and Hessian
//for cumulative probit model with response y and
//parameter vector beta.
#include<armadillo>
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
f2v cumprobit(ivec & y,vec & beta)
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
        p=normcdf(beta(i));
        r=normpdf(beta(i));
        if(i<y(0))
        {
            results.value=results.value+log(p);
            results.grad(i)=r/p;
            results.hess(i,i)=-(beta(i)+results.grad(i))*results.grad(i);
        }
        else
        {
            q=1.0-p;
            results.value=results.value+log(q);
            results.grad(i)=-r/q;
            results.hess(i,i)=-(beta(i)+results.grad(i))*results.grad(i);
            break;
        }
    }
    return results;
}
