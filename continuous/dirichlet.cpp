//Log likelihood component and its gradient and hessian matrix
//for Dirichlet model with response vector y and parameter vector with elements
//exp(beta(j)) for j from 0 to r, where r is the dimension
//of the observed vector y.dresp.
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
f2v dirichlet(const int & order, const resp & y, const vec & beta)
{
    double dz,tz,zz,zzz;
    int i,j,q;
    f2v results;
    q=beta.n_elem;
    vec yy(q),z(q);
    z=exp(beta);
    yy=log(y.dresp);
    if(order>0) results.grad.set_size(q);
    if(order>1) results.hess.set_size(q,q);
    zz=sum(z);
    results.value=dot(z,yy)-sum(yy)+lgamma(zz);
    for(i=0;i<q;i++) results.value=results.value-lgamma(z(i));
    if(order>0)
    {
        dz=digamma(zz);
        for(i=0;i<q;i++)results.grad(i)=z(i)*(yy(i)+dz-digamma(z(i)));
    }
    if(order>1)
    {
        tz=trigamma(zz);
        for(i=0;i<q;i++)
        {
            for(j=0;j<q;j++)
            {
                results.hess(i,j)=z(i)*z(j)*tz;
                if(i==j)results.hess(i,j)=
                    results.hess(i,j)-z(i)*z(i)*trigamma(z(i))
                    +results.grad(i);
            }
        }
    }
    return results;
}
