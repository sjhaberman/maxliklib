//Log likelihood component
//for model for ranks based on latent extreme value distributions.  There are r objects to be
//ranked and parameter
//vector beta has dimension r.  The vector y provides the ranking.
#include<armadillo>
using namespace arma;




double ranklogit0(ivec y,vec beta)
{

    double r;
    int i;
    vec e,f;
    double results;
    e=zeros(beta.n_elem);
    e(y(0))=exp(beta(y(0)));
    r=e(y(0));
    
    
    results=0.0;
   
    
    
   
    
    for(i=1;i<beta.n_elem;i++)
    {
        e(y(i))=exp(beta(y(i)));
        r=r+e(y(i));
        results=results+(beta(y(i))-log(r));
        
    }
    return results;
}
