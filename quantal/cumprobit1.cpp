//Log likelihood component and gradient
//for cumulative probit model with response y and
//parameter vector beta.
#include<armadillo>
using namespace arma;
struct f1v
{
    double value;
    vec grad;
};
f1v cumprobit1(ivec & y,vec & beta)
{
    double p,q,r;
    int i,n;
    f1v results;
    results.value=0.0;
    results.grad=zeros(beta.n_elem);
    for(i=0;i<beta.n_elem;i++)
    {
        p=normcdf(beta(i));
        r=normpdf(beta(i));
        if(i<y(0))
        {
            results.value=results.value+log(p);
            results.grad(i)=r/p;
        }
        else
        {
            q=1.0-p;
            results.value=results.value+log(q);
            results.grad(i)=-r/q;
            break;
        }
    }
    return results;
}
