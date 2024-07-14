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
    double p, p1, p2, r, r1, r2;
    f2v result,result1, result2;
    vec gamma1(1), gamma2(1);
    if(order>0)
    {
        result1.grad.set_size(1);
        result2.grad.set_size(1);
        result.grad.set_size(2);
    }
    if(order>1)
    {
        result1.hess.set_size(1,1);
        result2.hess.set_size(1,1);
        result.hess.set_size(2,2);
    }
    gamma1(0)=beta(0);
    gamma2(0)=beta(1);
    result1=berresp(order, transform, y, gamma1);
    result2=berresp(order, transform, y, gamma2);
    if(y.iresp(0)>0)
    {
        p1=exp(result1.value);
        p2=exp(result2.value);
        p=p1+p2-p1*p2;
        result.value=log(p);
        if(order>0)
        {
            r1=p1*(1.0-p2)/p;
            r2=p2*(1.0-p1)/p;
            r=-p1*p2/p;
            result.grad(0)=result1.grad(0)*r1;
            result.grad(1)=result2.grad(0)*r2;
        }
        if(order>1)
        {
            result.hess(0,0)=(result1.hess(0,0)
                +result1.grad(0)*result1.grad(0))*r1
                -result.grad(0)*result.grad(0);
            result.hess(1,1)=(result2.hess(0,0)
                +result2.grad(0)*result2.grad(0))*r2
                -result.grad(1)*result.grad(1);
            result.hess(0,1)=result1.grad(0)*result2.grad(0)*r
                -result.grad(0)*result.grad(1);
            result.hess(1,0)=result.hess(0,1);
        }
    }
    else
    {
       result.value=result1.value+result2.value;
       if(order>0)
       {
           result.grad(0)=result1.grad(0);
           result.grad(1)=result2.grad(0);
       }
       if(order>1)
       {
           result.hess(0,0)=result1.hess(0,0);
           result.hess(0,1)=0.0;
           result.hess(1,0)=0.0;
           result.hess(1,1)=result2.hess(0,0);
       }
    }
    return result;
}
