//Log likelihood component
//for cumulative complementary log-log model with response y and parameter
//vector beta.
#include<armadillo>
using namespace arma;





double cumloglog0(int y,vec beta)
{
    double p,q,r;
    int i;
    
    double results;
    results=0.0;
    
    
    for(i=0;i<beta.n_elem;i++)
    {
        q=exp(-exp(beta(i)));
      
        
        if(i<y)
        {
            p=1.0-q;
            r=exp(beta(i))/p;
            results=results+log(p);
            
            
        }
        else
        {
            
            r=log(q);
            results=results+r;
            
            
            break;
        }
    }
   
    
   
    return results;
}
