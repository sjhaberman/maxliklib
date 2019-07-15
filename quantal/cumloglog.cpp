//Log likelihood component, derivative, and second derivative
//for cumulative complementary log-log model with response y and parameter
//vector beta.
#include<armadillo>
using namespace arma;

struct fd2v
{
    double value;
    vec grad;
    mat hess;
};



fd2v cumloglog(int y,vec beta)
{
    double p,q,r;
    int i;
    
    fd2v results;
    results.value=0.0;
    results.grad=zeros(beta.n_elem);
    results.hess=zeros(beta.n_elem,beta.n_elem);
    for(i=0;i<beta.n_elem;i++)
    {
        p=1.0-exp(-exp(beta(i)));
        q=1.0-p;
        
        if(i<y)
        {
            r=exp(beta(i))/p;
            results.value=results.value+log(p);
            results.grad(i)=q/p;
            results.hess(i,i)=q*r*(1.0-r);
        }
        else
        {
            r=log(q);
            results.value=results.value+r;
            results.grad(i)=r;
            results.hess(i,i)=r;
            break;
        }
    }
   
    
   
    return results;
}
