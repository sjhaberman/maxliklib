//Log likelihood component and its gradient and hessian matrix
//for logit-dirichlet model with response y and parameter vector beta
//of dimension q = r+1, where r is the dimension of the observed vector
//y.dresp.  All elements of beta are positive.  For a vector u of dimension
//q with a Dirichlet distribution, y(i)=log(u(i)/u(q)).  If order is 0,
//only the function is
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
f2v logitdirichlet(const int & order, const resp & y, const vec & beta)
{
    double dz,tz,zz,zzz;
    int i,j,q;
    f2v results;
    q=beta.n_elem;
    vec z(q);
    if(order>0) results.grad.set_size(q);
    if(order>1) results.hess.set_size(q,q);
    if(min(beta)<=0.0)
    {
        results.value=datum::nan;
        if(order>0) results.grad.fill(datum::nan);
        if(order>1) results.hess.fill(datum::nan);
        return results;
    }
    zz=sum(beta);
    zzz=log(1.0+sum(exp(y.dresp)));
    results.value=dot(beta.subvec(0,q-2),y.dresp)-zz*zzz+lgamma(zz);
    for(i=0;i<q;i++) results.value=results.value-lgamma(beta(i));
    if(order>0)
    {
        dz=digamma(zz);
        for(i=0;i<q;i++)
        {
            results.grad(i)=-zzz+dz-digamma(beta(i));
            if(i<q-1)results.grad(i)=results.grad(i)+y.dresp(i);
        }
    }
    if(order>1)
    {
        tz=trigamma(zz);
        for(i=0;i<q;i++)
        {
            for(j=0;j<q;j++)
            {
                results.hess(i,j)=tz;
                if(i==j)results.hess(i,j)=results.hess(i,j)-trigamma(beta(i));
            }
        }
    }
    return results;
}
