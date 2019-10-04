//Value for standard Gumbel distribution function at beta.
#include<cmath>





double gumbelf0(double beta)
{

    double p;
    double results;
    p=exp(-beta);
    results=exp(-p);
    
   
    
    return results;
}
