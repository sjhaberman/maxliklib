//Value a for standard normal distribution function at beta.
#include<armadillo>
using namespace arma;




double normalf0(double beta)
{

    
    double results;
    results=normcdf(beta);
    
   
    
    return results;
}
