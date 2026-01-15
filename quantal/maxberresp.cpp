//Log likelihood component, gradient, and hessian matrix
//for Bernoulli response that is the maximum of two independent Bernoulli
//random variables. The response is y and the two-dimesional parameter
//vector is beta. The struct variable choice governs the model selection.
//transform='G' applies to the complementary log-log case,
//transform='H' applies to the log-log case,
//transform='L' applies to the logit case, and 
//transform='N' applies to the probit case.
//If order is 0, only the function is
//found, if order is 1, then the function and gradient are found.
//If order is 2,
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
f2v berresp(const int & , const char & , const resp & , const vec & );
f2v maxberresp(const int & order, const char  & transform,
    const resp & y, const vec & beta)
{
    double p, p0, p1, r, r0, r1;
    f2v result,result0, result1;
    vec gamma0={beta(0)},gamma1={beta(1)};
    if(order>0) result.grad.set_size(2);
    if(order>1) result.hess.set_size(2,2);
    result0=berresp(order, transform, y, gamma0);
    result1=berresp(order, transform, y, gamma1);
    if(y.iresp(0)>0)
    {
        p0=exp(result0.value);
        p1=exp(result1.value);
        p=p0+p1-p0*p1;
        result.value=log(p);
        if(order>0)
        {
            r0=p0*(1.0-p1)/p;
            r1=p1*(1.0-p0)/p;
            r=-p0*p1/p;
            result.grad(0)=result0.grad(0)*r0;
            result.grad(1)=result1.grad(0)*r1;
        }
        if(order>1)
        {
            result.hess(0,0)=(result0.hess(0,0)
                +result0.grad(0)*result0.grad(0))*r0
                -result.grad(0)*result.grad(0);
            result.hess(1,1)=(result1.hess(0,0)
                +result1.grad(0)*result1.grad(0))*r1
                -result.grad(1)*result.grad(1);
            result.hess(0,1)=result0.grad(0)*result1.grad(0)*r
                -result.grad(0)*result.grad(1);
            result.hess(1,0)=result.hess(0,1);
        }
    }
    else
    {
       result.value=result0.value+result1.value;
       if(order>0)
       {
           result.grad(0)=result0.grad(0);
           result.grad(1)=result1.grad(0);
       }
       if(order>1)
       {
           result.hess(0,0)=result0.hess(0,0);
           result.hess(0,1)=0.0;
           result.hess(1,0)=0.0;
           result.hess(1,1)=result1.hess(0,0);
       }
    }
    return result;
}
