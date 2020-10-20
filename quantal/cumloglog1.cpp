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
    int i,n;
    f1v results;
    n=beta.n_elem;
    results.value=0.0;
    results.grad.set_size(n);
    results.grad.zeros();
    for(i=0;i<n;i++)
    {
        r=exp(-beta(i));
        if(i<y(0))
        {
            results.value=results.value-r;
            results.grad(i)=r;
        }
        else
        {
            p=exp(-r);
            q=1.0-p;
            results.value=results.value+log(q);
            results.grad(i)=-p*r/q;
            break;
        }
    }
    return results;
}
