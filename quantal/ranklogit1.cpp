//Log likelihood component and gradient
//for model for ranks based on latent extreme value distributions.  There are r objects to be
//ranked and parameter
//vector beta has dimension r.  The vector y provides the ranking.
#include<armadillo>
using namespace arma;

struct fd1v
{
    double value;
    vec grad;
    
};



fd1v ranklogit1(ivec y,vec beta)
{

    double r;
    int i;
    vec e,f;
    fd1v results;
    e=zeros(beta.n_elem);
    e(y(0))=exp(beta(y(0)));
    r=e(y(0));
    
    
    results.value=0.0;
   
    results.grad=zeros(beta.n_elem);
    
   
    
    for(i=1;i<beta.n_elem;i++)
    {
        e(y(i))=exp(beta(y(i)));
        r=r+e(y(i));
        results.value=results.value+(beta(y(i))-log(r));
        f=-e/r;
        results.grad=results.grad+f;
        
        results.grad(y(i))=1.0+results.grad(y(i));
    }
    return results;
}
