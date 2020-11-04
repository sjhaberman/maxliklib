//Log likelihood and its gradient and approximate
//Hessian matrix
//for quantal response model with response vector y and
//predictor matrix x.
//Weight w is used.
//Model codes are found in quantal1.cpp.
//Sparseness is not exploited.
//The Louis approximation is used.
#include<armadillo>
using namespace arma;
struct f1v
{
    double value;
    vec grad;
};
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
struct model
{
  char type;
  char transform;
};
extern model choices[];
extern vec w;
extern ivec y[];
extern mat x[];
extern mat offset[];
f1v quantal1(model &,ivec &,vec &);
f2v quantallikl(vec & beta)
{
    vec lambda,u;
    f1v obsresults;
    f2v results;
    int i,p,r,n;
    results.value=0.0;
    p=beta.n_elem;
    n=w.n_elem;
    results.grad.set_size(p);
    results.grad.zeros();
    results.hess.set_size(p,p);
    results.hess.zeros();
    u.set_size(p);
    for (i=0;i<n;i++)
    {
        r=offset[i].n_elem;
        lambda.set_size(r);
        obsresults.grad.set_size(r);
        lambda=offset[i]+x[i]*beta;
        obsresults=quantal1(choices[i],y[i],lambda);
        u=trans(x[i])*obsresults.grad;
        results.value=results.value+w(i)*obsresults.value;
        results.grad=results.grad+w(i)*u;
        results.hess=results.hess-w(i)*u*trans(u);
    }
    return results;
}
