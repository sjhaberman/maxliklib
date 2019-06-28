//Log likelihood component, derivative, and second derivative
//for logit model with response y and parameter beta.
#include<cmath>

struct fd2
{
    double value;
    double der1;
    double der2;
};



fd2 logit(int y,double beta)
{
//Probability of response of 1.
    double p,q;
    fd2 results;
    p=1.0/(1.0+exp(-beta));
    q=1.0-p;
    if(y==1)
    {
        results.value=log(p);
    }
    else
    {
        results.value=log(q);
    }
    results.der1=double(y)-p;
    results.der2=-p*q;
    return results;
}
