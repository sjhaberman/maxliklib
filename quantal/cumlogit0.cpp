//Log likelihood component and gradient
//for cumulative logit model with response y and parameter beta.
#include<armadillo>
using namespace arma;




double cumlogit0(int y,vec beta)
{

    double p,q;
    int i;
    double results;
    
    results=0.0;
    
    
    for(i=0;i<beta.n_elem;i++)
    {
        p=1.0/(1.0+exp(-beta(i)));
        q=1.0-p;
        
        if(i<y)
        {
            results=results+log(p);
            
            
        }
        else
        {
            results=results+log(q);
         
            
            break;
        }
    }
    
    return results;
}
