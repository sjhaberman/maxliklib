//Value, derivative, and second derivative for standard normal distribution function at beta.
#include<armadillo>
using namespace arma;
struct fd2
{
    double value;
    double der1;
    double der2;
};



fd2 normalf012(double beta)
{

    
    fd2 results;
    results.value=normcdf(beta);
    results.der1=normpdf(beta);
    results.der2=-beta*results.der1;
    
    return results;
}
