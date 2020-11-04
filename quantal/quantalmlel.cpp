//Find maximum likelihood estimates for quantal response model
//based on one parameter for a response.
//Maximum number of iterations is maxit.
//Tolerance is tol.
//Responses are y.
//Predictors are x.
//Weights are w.  The Louis method is used.
#include<armadillo>
using namespace arma;
using namespace std;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
struct maxf2v
{
    vec locmax;
    double max;
    vec grad;
    mat hess;
};
struct paramnr
{
    int maxit;
    int maxits;
    double eta;
    double gamma1;
    double gamma2;
    double kappa;
    double tol;
};
maxf2v nrv(const paramnr&,const vec &,const function<f2v(vec &)>);
f2v quantallikl(vec &);
maxf2v quantalmlel(const paramnr & nrparams,const vec & start)
{
    maxf2v results;
    int p;
    p=start.n_elem;
    results.locmax.set_size(p);
    results.grad.set_size(p);
    results.hess.set_size(p,p);
    results=nrv(nrparams,start,quantallikl);
    return results;
}
