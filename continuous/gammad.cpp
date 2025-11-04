//Log likelihood component and its gradient and hessian matrix
//for gamma model with response y and parameter vector beta
//with elements beta(0) and beta(1).  The first parameter
//is the log of the scale parameter and the second parameter is the
//log of theargument of the gamma function
//associated with the distribution.
//If order is 0, only the function is
//found, if order is 1, then the function and gradient are found.
//If order is 2,
//then the function, gradient, and Hessian are returned.
#include<armadillo>
#include<boost/math/special_functions/digamma.hpp>
#include<boost/math/special_functions/trigamma.hpp>
using namespace arma;
using namespace boost::math;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
struct resp
{
    ivec iresp;
    vec dresp;
};
f2v gammad(const int & order, const resp & y, const vec & beta)
{
    double z,zz;
    f2v results;
    vec g(2);
    g(0)=exp(beta(0));
    g(1)=exp(beta(1));
    if(order>0) results.grad.set_size(2);
    if(order>1) results.hess.set_size(2,2);
    z=log(y.dresp(0));
    zz=y.dresp(0);
    results.value=g(1)*(z+beta(0))-g(0)*zz-lgamma(g(1));
    if(order>0)
    {
        results.grad(0)=g(1)-g(0)*zz;
        results.grad(1)=g(1)*(z+beta(0))-g(1)*digamma(g(1));
    }
    if(order>1)
    {
        results.hess(0,0)=-g(0)*zz;
        results.hess(0,1)=g(1);
        results.hess(1,0)=g(1);
        results.hess(1,1)=results.grad(1)-g(1)*g(1)*trigamma(g(1));
    }
    return results;
}
