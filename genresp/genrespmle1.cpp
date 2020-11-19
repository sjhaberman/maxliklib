//Find maximum likelihood estimates for response model
//Responses are y.
//Predictors are x.
//Weights are w.
//Conjugate gradient method is used.
#include<armadillo>
using namespace arma;
using namespace std;
struct f1v
{
    double value;
    vec grad;
};
struct maxf1v
{
    vec locmax;
    double max;
    vec grad;
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
maxf1v conjgrad(const params &,const vec &, function<f1v(vec &)>);
f1v genresplik1(vec &);
maxf1v genrespmle1(const params & mparams,const vec & start)
{
    maxf1v results;
    int p;
    p=start.n_elem;
    results.locmax.set_size(p);
    results.grad.set_size(p);
    results=conjgrad(mparams,start,genresplik1);
    return results;
}
