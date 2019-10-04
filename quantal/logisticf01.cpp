//Value and derivative for standard logistic distribution function at beta.
#include<cmath>

struct fd1
{
    double value;
    double der1;
    
};



fd1 logisticf01(double beta)
{

    
    fd1 results;
    results.value=1.0/(1.0+exp(-beta));
    results.der1=results.value*(1.0-results.value);
   
    
    return results;
}
