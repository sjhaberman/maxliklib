//Find maximum likelihood estimates for response model
//Responses are y.
//Predictors are x.
//Weights are w.  The Louis approximation is used.
#include<armadillo>
using namespace arma;
using namespace std;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
struct f1v
{
    double value;
    vec grad;
};
struct maxf2v
{
    vec locmax;
    double max;
    vec grad;
    mat hess;
};
struct params
{
    int maxit;
    int maxits;
    double eta;
    double gamma1;
    double gamma2;
    double kappa;
    double tol;
};
maxf2v nrv(const params&,const vec &,function<f2v(vec &)>);
f2v genresplikl(vec &);
maxf2v genrespmlel(const params & mparams,const vec & start)
{
    maxf2v results;
    int p;
    p=start.n_elem;
    results.locmax.set_size(p);
    results.grad.set_size(p);
    results.hess.set_size(p,p);
    results=nrv(mparams,start,genresplikl);
    return results;
}
