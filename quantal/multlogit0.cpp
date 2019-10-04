//Log likelihood component and gradien
//for multinomial logit model with response y from 0 to r-1 and parameter
//vector beta of dimension r-1.
#include<armadillo>
using namespace arma;




double multlogit0(int y,vec beta)
{

    double r;
    int z;
    vec e;
    double results;
    e=exp(beta);
    
    r=1.0+sum(e);
    results=-log(r);
    
   
    if(y>0)
    {
        z=y-1;
        results=beta(z)+results;
        
    }
    
    return results;
}
