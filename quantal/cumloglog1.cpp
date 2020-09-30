//Log likelihood component and gradient
//for cumulative complementary log-log model with response y and parameter
//vector beta.
#include<armadillo>
using namespace arma;
struct f1v
{
    double value;
    vec grad;
};
f1v cumloglog1(ivec & y,vec & beta)
{
    double p,q,r;
    int i;
    f1v results;
    results.value=0.0;
    results.grad=zeros(beta.n_elem);
    for(i=0;i<beta.n_elem;i++)
    {
        r=exp(beta(i));
        if(i<y(0))
        {
            results.value=results.value-r;
            results.grad(i)=-r;
        }
        else
        {
            q=exp(-r);
            p=1.0-q;
            results.value=results.value+log(p);
            results.grad(i)=q*r/p;
            break;
        }
    }
    return results;
}
