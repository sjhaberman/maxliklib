//Value and derivative for standard Gumbel distribution function at beta.
#include<cmath>

struct fd1
{
    double value;
    double der1;
    
};



fd1 gumbelf01(double beta)
{

    double p;
    fd1 results;
    p=exp(-beta);
    results.value=exp(-p);
    results.der1=p*results.value;
   
    
    return results;
}
