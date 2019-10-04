//Log likelihood component and its first derivative
//for probit model with response y and parameter beta.

#include<armadillo>
using namespace arma;
struct fd1
{
    double value;
    double der1;
   
};


fd1 probit1(int y,double beta)
{
    double p,q,r;
    fd1 results;
    
    p=normcdf(beta);
    
    r=normpdf(beta);
    if(y==1)
    {
        results.value=log(p);
        results.der1=r/p;
        
    }
    else
    {
        
        q=1.0-p;
        results.value=log(q);
        results.der1=-r/q;
        
    }
   
    return results;
}
