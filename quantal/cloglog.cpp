//Log likelihood component, derivative, and second derivative
//for complementary log-log model with response y and parameter beta.
#include<cmath>

struct fd2
{
    double value;
    double der1;
    double der2;
};



fd2 cloglog(int y,double beta)
{
//Probability of response of 1.
    double p,q,r;
    fd2 results;
    q=exp(-exp(beta));
    p=1.0-q;
    
    if(y==1)
    {
        r=exp(beta)/p;
        results.value=log(p);
        results.der1=r*q;
        results.der2=r*q*(1.0-r);
    }
    else
    {
        results.value=log(q);
        results.der1=results.value;
        results.der2=results.value;
    }
    return results;
}
