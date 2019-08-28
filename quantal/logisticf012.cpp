//Value, derivative, and second derivative for standard logistic distribution function at beta.
#include<cmath>

struct fd2
{
    double value;
    double der1;
    double der2;
};



fd2 logisticf012(double beta)
{

    
    fd2 results;
    results.value=1.0/(1.0+exp(-beta));
    results.der1=results.value*(1.0-results.value);
    results.der2=results.der1*(1.0-2.0*results.value);
    
    return results;
}
