//Value and first derivative for logarithm of standard logistic density.

#include<cmath>

struct fd1
{
    double value;
    double der1;
    
    
};


fd1 logistic01(double y)
{
    
    fd1 results;
    double p,q,w;
    p=exp(-y);
    p=p/(1.0+p);
    q=1.0-p;
    w=p*q;
    results.value=log(w);
    results.der1=p-q;
    
    return results;
}
