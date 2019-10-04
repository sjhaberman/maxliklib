//Log likelihood component and derivative
//for logit model with response y and parameter beta.
#include<cmath>

struct fd1
{
    double value;
    double der1;
   
};



fd1 logit1(int y,double beta)
{
//Probability of response of 1.
    double p,q;
    fd1 results;
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
    
    return results;
}
