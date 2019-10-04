//Log likelihood component
//for probit model with response y and parameter beta.

#include<armadillo>
using namespace arma;



double probit0(int y,double beta)
{
    double p,q;
    double results;
    
    p=normcdf(beta);
    
   
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
