//Log likelihood and its gradient and Hessian
//for probit model with response vector y and
//predictor matrix x.
//Weight weight is used.
#include<armadillo>
using namespace arma;
struct fd2v
{
    double value;
    vec grad;
    mat hess;
};
struct regvars
{
    mat x;
    vec y;
    vec weight;
};
mat wcrossprod(mat,vec);
extern regvars xyw;

//Probabilities corresponding to logit.
fd2v probitfunct(vec beta)
{
    fd2v results;
    vec lambda,logp,p,q,r,w;
    lambda=xyw.x*beta;
    q=normcdf(lambda);
    p=xyw.y%q+(1.0-xyw.y)%(1.0-q);
    r=normpdf(lambda);
    w=xyw.weight%(xyw.y%square(r/q)+(1.0-xyw.y)%square(r/(1.0-q))+r%lambda%(xyw.y-q)/(q%(1.0-q)));
    logp=log(p);
    results.value=cdot(xyw.weight,logp);
    results.grad=trans(xyw.x)*(xyw.weight%r%(xyw.y-q)/(q%(1-q)));
    results.hess=-wcrossprod(xyw.x,w);
    return results;
}
