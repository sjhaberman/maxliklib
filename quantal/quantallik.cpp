//Log likelihood and its gradient and Hessian
//for quantal response model with response vector y and
//predictor matrix x.
//Weight w is used.
//Model codes are found in quantal.cpp.
//Sparseness is not exploited.
#include<armadillo>
using namespace arma;
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
extern model choice;
extern vec w;
extern ivec y[];
extern mat x[];
extern mat offset[];
f2v quantal(model &,ivec &,vec &);
f2v quantallik(vec & beta)
{
    vec lambda;
    f2v obsresults;
    f2v results;
    int i,p,n;
    results.value=0.0;
    p=beta.n_elem;
    n=w.n_elem;
    results.grad.set_size(p);
    results.grad.zeros();
    results.hess.set_size(p,p);
    results.hess.zeros();
    for (i=0;i<n;i++)
    {
        lambda=offset[i]+x[i]*beta;
        choice=choices[i];
        obsresults=quantal(choice,y[i],lambda);
        results.value=results.value+w(i)*obsresults.value;
        results.grad=results.grad+w(i)*trans(x[i])*obsresults.grad;
        results.hess=results.hess+
            w(i)*trans(x[i])*obsresults.hess*x[i];
    }
    return results;
}
