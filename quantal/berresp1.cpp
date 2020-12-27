//Log likelihood component and gradient
//for Bernoulli response model with response y and one-dimensional parameter beta.
//Choice of tranformation is determined by transform, with 'G' for
//log-log, 'L' for logit, and 'P' for probit.
#include<armadillo>
using namespace arma;
struct f1v
{
    double value;
    vec grad;
};
f1v logit1(const ivec & y, const vec & beta);
f1v loglog1(const ivec & y, const vec & beta);
f1v probit1(const ivec & y, const vec & beta);
f1v berresp1(const char & transform, const ivec & y, const vec & beta)
{
    f1v result;
    result.grad.set_size(1);
    switch(transform)
    {
        case 'G': result=loglog1(y,beta); return result;
        case 'L': result=logit1(y,beta); return result;
        default:  result=probit1(y,beta); return result;
    }
}
