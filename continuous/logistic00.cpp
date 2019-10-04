//Value  for logarithm of standard logistic density.

#include<cmath>




double logistic00(double y)
{
    
    double results;
    double p,q,w;
    p=exp(-y);
    p=p/(1.0+p);
    q=1.0-p;
    w=p*q;
    results=log(w);
    
    
    return results;
}
