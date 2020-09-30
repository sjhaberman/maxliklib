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
f2v cumprobit(ivec & y,vec& beta)
{
    double p,q,r;
    int i;
    f2v results;
    results.value=0.0;
    results.grad=zeros(beta.n_elem);
    results.hess=zeros(beta.n_elem,beta.n_elem);
    for(i=0;i<beta.n_elem;i++)
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
