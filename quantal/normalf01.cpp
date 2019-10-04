//Value and derivative for standard normal distribution function at beta.
#include<armadillo>
using namespace arma;
struct fd1
{
    double value;
    double der1;
    
};



fd1 normalf01(double beta)
{

    
    fd1 results;
    results.value=normcdf(beta);
    results.der1=normpdf(beta);
   
    
    return results;
}
