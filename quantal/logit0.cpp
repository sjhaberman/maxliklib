//Log likelihood component
//for logit model with response y and parameter beta.
#include<cmath>




double logit0(int y,double beta)
{
//Probability of response of 1.
    double p,q;
    double results;
    p=1.0/(1.0+exp(-beta));
    q=1.0-p;
    if(y==1)
    {
        results=log(p);
    }
    else
    {
        results=log(q);
    }
    
    
    return results;
}
