//Log likelihood component and gradient
//for multinomial logit model with response y from 0 to r-1 and parameter
//vector beta of dimension r-1.
#include<armadillo>
using namespace arma;

struct fd1v
{
    double value;
    vec grad;
    
};



fd1v multlogit1(int y,vec beta)
{

    double r;
    int z;
    vec e;
    fd1v results;
    e=exp(beta);
    
    r=1.0+sum(e);
    results.value=-log(r);
    results.grad=-e/r;
   
    if(y>0)
    {
        z=y-1;
        results.value=beta(z)+results.value;
        results.grad(z)=1.0+results.grad(z);
    }
    
    return results;
}
