//Log likelihood component
//for complementary log-log model with response y and parameter beta.
#include<cmath>





double cloglog0(int y,double beta)
{
//Probability of response of 1.
    double q;
    double results;
    q=exp(-exp(beta));
 
    
    if(y==1)
    {
        
        results=log(1.0-q);
        
       
    }
    else
    {
        
        results=log(q);
        
        
    }
    return results;
}
