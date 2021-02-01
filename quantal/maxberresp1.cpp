//Log likelihood component and  gradient
//for Bernoulli response that is the maximum of two independent Bernoulli
//random variables. The response is y and the two-dimesional parameter
//vector is beta. The struct variable choice governs the model selection.
//transform='G' applies to the log-log case,
//transform='L' applies to the logit case, and 
//transform='P' applies to the probit case.
#include<armadillo>
using namespace arma;
struct f1v
{
    double value;
    vec grad;
};
f1v berresp1(const char & , const ivec &, const vec &);
f1v maxberresp1(const char & transform, const ivec & y, const vec & beta)
{
    double p, p1, p2, r1, r2;
    f1v result, result1, result2;
    vec gamma1(1), gamma2(1);
    result1.grad.set_size(1);
    result2.grad.set_size(1);
    result.grad.set_size(2);
    gamma1(0)=beta(0);
    gamma2(0)=beta(1);	
    result1=berresp1(transform, y, gamma1);
    result2=berresp1(transform, y, gamma2);
    if(y(0)>0)
    {
        p1=exp(result1.value);
        p2=exp(result2.value);
        p=p1+p2-p1*p2;
        result.value=log(p);
        r1=p1*(1.0-p2)/p;
        r2=p2*(1.0-p1)/p;
        result.grad(0)=result1.grad(0)*r1;
        result.grad(1)=result2.grad(0)*r2;
    }
    else
    {
       result.value=result1.value+result2.value;
       result.grad(0)=result1.grad(0);
       result.grad(1)=result2.grad(0);
    }
    return result;
}