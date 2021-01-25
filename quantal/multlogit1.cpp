//Log likelihood component and gradient
//for multinomial logit model with response vector y such that y(0)
//has integer values from 0 to r-1 and parameter
//vector beta of dimension r-1.
#include<armadillo>
using namespace arma;
struct f1v
{
    double value;
    vec grad;
};
f1v multlogit1(const ivec & y, const vec & beta)
{
    double s;
    int r,z;
    vec e;
    f1v results;
    r=beta.n_elem;
    results.grad.set_size(r);
    e=exp(beta);
    s=1.0+sum(e);
    results.value=-log(s);
    results.grad=-e/s;
    if(y(0)>0)
    {
        z=y(0)-1;
        results.value=beta(z)+results.value;
        results.grad(z)=1.0+results.grad(z);
    }    
    return results;
}
