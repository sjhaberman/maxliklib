//Log likelihood component
//for complementary log-log model with response y and parameter beta.
#include<cmath>





double cloglog0(int y,double beta)
{
//Probability of response of 1.
    double p,q,r;
    double results;
    p=1.0-exp(-exp(beta));
 
    
    if(y==1)
    {
        
        results=log(p);
        
       
    }
    else
    {
        q=1.0-p;
        results=log(q);
        
        
    }
    return results;
}
