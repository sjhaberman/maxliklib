//Log likelihood component and gradient
//for model for ranks based on latent Gumbel distributions.
//There are r objects to be
//ranked and parameter
//vector beta has dimension r-1.  The analysis
//considers the k largest objects for a
//positive integer k no greater than r-1.
//The vector y provides the ranking.
#include<armadillo>
using namespace arma;
struct f1v
{
    double value;
    vec grad;
};
f1v ranklogit1(ivec & y,vec & beta)
{
    double s;
    int i,k,rr;
    vec f;
    f1v results;
    rr=beta.n_elem;
    k=y.n_elem;
    results.value=0.0;
    results.grad.set_size(rr);
    results.grad.zeros();
    f=exp(beta);
    s=1.0+sum(f);
    for(i=0;i<k;i++)
    {
        results.value=results.value-log(s);
        results.grad=results.grad-f/s;
        if(y(i)>0)
        {
            results.value=results.value+beta(y(i)-1);
            results.grad(y(i)-1)=results.grad(y(i)-1)+1.0;
            s=s-f(y(i)-1);
            f(y(i)-1)=0.0;
        }
        else
        {
            s=s-1.0;
        }
    }
    return results;
}
