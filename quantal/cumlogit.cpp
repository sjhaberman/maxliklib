//Log likelihood component, gradient, and Hessian
//for cumulative logit model with response y and parameter beta.
#include<armadillo>
using namespace arma;

struct fd2v
{
    double value;
    vec grad;
    mat hess;
};


fd2v cumlogit(int y,vec beta)
{

    double p,q;
    int i;
    fd2v results;
    
    results.value=0.0;
    results.grad=zeros(beta.n_elem);
    results.hess=zeros(beta.n_elem,beta.n_elem);
    for(i=0;i<beta.n_elem;i++)
    {
        p=1.0/(1.0+exp(-beta(i)));
        q=1.0-p;
        results.hess(i,i)=-p*q;
        if(i<y)
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
