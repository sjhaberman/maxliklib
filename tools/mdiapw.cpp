//Make MDIA adjustment to quadrature weights.
//Variables are defined as in kl.cpp and nrv.cpp.
#include<armadillo>
using namespace arma;
using namespace std;
//Function, gradient, and Hessian.
struct pw
{
    vec points;
    vec weights;
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
    bool print;
    int maxit;
    int maxits;
    double eta;
    double gamma1;
    double gamma2;
    double kappa;
    double tol;
};
maxf2v mdia(const vec & p, const mat & T, const vec & u, const params & mparams);
pw mdiapw(const pw & pws, const mat & T, const vec & u, const params & mparams)
{
    int d,n,order=2;
    d=u.n_elem;
    double s;
    maxf2v result;
    result.grad.set_size(d);
    result.hess.set_size(d,d);
    result.locmax.set_size(d);
    vec start(d,fill::zeros),q(d);
    result=mdia(pws.weights,T,u,mparams);
    pw results;
    n=pws.weights.n_elem;
    results.weights.set_size(n);
    results.points.set_size(n);
    results.points=pws.points;
    q=T*result.locmax;
    results.weights=pws.weights%exp(q);
    s=sum(results.weights);
    results.weights=results.weights/s;                                    
    return results;
}
