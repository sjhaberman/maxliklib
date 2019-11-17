//Log likelihood component and gradient
//for cumulative complementary log-log model with response y and parameter
//vector beta.
#include<armadillo>
using namespace arma;

struct fd1v
{
    double value;
    vec grad;
    
};



fd1v cumloglog1(int y,vec beta)
{
    double p,q,r;
    int i;
    
    fd1v results;
    results.value=0.0;
    results.grad=zeros(beta.n_elem);
    
    for(i=0;i<beta.n_elem;i++)
    {
        q=exp(-exp(beta(i)));
        p=1.0-q;
        
        if(i<y)
        {
            r=exp(beta(i))/p;
            results.value=results.value+log(p);
            results.grad(i)=r*q;
            
        }
        else
        {
            r=log(q);
            results.value=results.value+r;
            results.grad(i)=r;
            
            break;
        }
    }
   
    
   
    return results;
}
