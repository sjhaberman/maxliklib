//Log likelihood component and its gradient and Hessian
//for continuous model with response y and parameter beta.
//transform determines the model with 'G' for minimum Gumbel,
//'H' for maximum Gumbel,
//'L' for logistic, and 'N' for normal.
//If order is 0, only the function is
//found, if order is 1, then the function and gradient are found.
//If order is 2,
//then the function, gradient, and Hessian are returned.
#include<armadillo>
using namespace arma;
struct f2
{
    double value;
    double der;
    double der2;
};
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
f2 gumbell(const int & , const double &);
f2 gumbelu(const int & , const double &);
f2 logistic(const int & , const double  &);
f2 normal(const int & , const double & );
f2v contresp(const int & order, const char & transform, const resp & y,
    const vec &  beta)
{
    double z,zz;
    f2 result;
    f2v results;
    zz=exp(beta(1));
    z=beta(0)+zz*y.dresp(0);
    if(order>0) results.grad.set_size(2);
    if(order>1) results.hess.set_size(2,2);
    switch (transform)
    {
        case 'G':
            result = gumbell(order, z);
            break;
        case 'H':
            result = gumbelu(order, z);
            break;
        case 'L':
            result = logistic(order, z);
            break;
        default: result = normal(order, z);
    }
    results.value=result.value+beta(1);
    if(order>0)
    {
        results.grad(0)=result.der;
        results.grad(1)=zz*y.dresp(0)*result.der+1.0;
    }
    if(order>1)
    {
        results.hess(0,0)=result.der2;
        results.hess(1,0)=zz*y.dresp(0)*result.der2;
        results.hess(0,1)=results.hess(1,0);
        results.hess(1,1)=zz*y.dresp(0)*(result.der+results.hess(0,1));
    }
    return results;
}
