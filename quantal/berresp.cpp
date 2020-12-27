//Log likelihood component, gradient, and Hessian matrix
//for Bernoulli response model with response y and one-dimensional parameter beta.
//Choice of tranformation is determined by transform, with 'G' for
//log-log, 'L' for logit, and 'P' for probit.
#include<armadillo>
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
f2v logit(const ivec & y, const vec & beta);
f2v loglog(const ivec & y, const vec & beta);
f2v probit(const ivec & y, const vec & beta);
f2v berresp(const char & transform, const ivec & y, const vec & beta)
{
    f2v result;
    result.grad.set_size(1);
    result.hess.set_size(1,1);
    switch(transform)
    {
        case 'G': result=loglog(y,beta); return result;
        case 'L': result=logit(y,beta); return result;
        default:  result=probit(y,beta); return result;
    }
}
