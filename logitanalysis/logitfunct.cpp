//Log likelihood and its gradient and Hessian
//for logit model with response vector y and
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
fd2v logitfunct(vec beta)
{
    fd2v results;
    vec lambda,logp,p,q,r,w;
    lambda=xyw.x*beta;
    r=exp(lambda);
    q=r+1.0;
    p=r/q;
    w=xyw.weight%p%(1.0-p);
    logp=xyw.y%lambda-log(q);
    results.value=cdot(xyw.weight,logp);
    results.grad=trans(xyw.x)*(xyw.weight%(xyw.y-p));
    results.hess=-wcrossprod(xyw.x,w);
    return results;
}
