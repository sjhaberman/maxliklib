//Log likelihood component and derivative
//for complementary log-log model with response y and parameter beta.
#include<cmath>

struct fd1
{
    double value;
    double der1;
    
};



fd1 cloglog1(int y,double beta)
{
//Probability of response of 1.
    double p,q,r;
    fd1 results;
    q=exp(-exp(beta));
    p=1.0-q;
    
    if(y==1)
    {
        r=exp(beta)/p;
        results.value=log(p);
        results.der1=r*q;
       
    }
    else
    {
        results.value=log(q);
        results.der1=results.value;
        
    }
    return results;
}
