//Value and derivative for logarithm of standard gumbel density.

#include<cmath>

struct fd1
{
    double value;
    double der1;
    
    
};


fd1 gumbel01(double y)
{
    
    fd1 results;
    double p;
    p=exp(-y);
    results.value=-y-p;
    results.der1=-1.0+p;
    
    return results;
}
