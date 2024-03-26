//Log likelihood component, gradient, and Hessian matrix
//for Bernoulli response model with response y and one-dimensional parameter beta.
//Choice of tranformation is determined by transform, with 'G' for
//log-log, 'L' for logit, and 'N' for probit.  If order is 0, only the function is
//found, if order is 1, then the function and gradient are found.  If order is 2,
//then the function, gradient, and Hessian are returned.
#include<armadillo>
using namespace arma;
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
f2v logit(const int & , const resp & , const vec & );
f2v loglogl(const int & , const resp & , const vec & );
f2v loglogu(const int & , const resp & , const vec & );
f2v probit(const int &, const resp & , const vec & );
f2v berresp(const int & order, const char & transform, const resp & y,
    const vec & beta)
{
    f2v result;
    if(order>0) result.grad.set_size(1);
    if(order>1) result.hess.set_size(1,1);
    switch(transform)
    {
        case 'G': return loglogl(order, y, beta);
        case 'H': return loglogu(order, y, beta);
        case 'L': return logit(order, y, beta);
        default:  return probit(order, y, beta);
    }
}
