//Log likelihood component, gradient, and hessian matrix
//for model for ranks based on latent extreme value distributions.  There are r objects to be
//ranked and parameter
//vector beta has dimension r.  The vector y provides the ranking.
#include<armadillo>
using namespace arma;

struct fd2v
{
    double value;
    vec grad;
    mat hess;
};



fd2v ranklogit(ivec y,vec beta)
{

    double r;
    int i;
    vec e,f;
    fd2v results;
    e=zeros(beta.n_elem);
    e(y(0))=exp(beta(y(0)));
    r=e(y(0));
    
    
    results.value=0.0;
   
    results.grad=zeros(beta.n_elem);
    
   
    results.hess=zeros(beta.n_elem,beta.n_elem);
    for(i=1;i<beta.n_elem;i++)
    {
        e(y(i))=exp(beta(y(i)));
        r=r+e(y(i));
        results.value=results.value+(beta(y(i))-log(r));
        f=-e/r;
        results.grad=results.grad+f;
        results.hess=results.hess+diagmat(f)+f*trans(f);
        results.grad(y(i))=1.0+results.grad(y(i));
    }
    return results;
}
