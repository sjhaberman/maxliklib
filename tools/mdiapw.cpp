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
maxf2v mdia(const vec & , const mat & , const vec & , const params & );
pw mdiapw(const pw & pws, const mat & T, const vec & u, const params & mparams)
{
    int order=2;
    double s;
    maxf2v result;
    result=mdia(pws.weights,T,u,mparams);
    pw results;
    results.points=pws.points;
    vec q=T*result.locmax;
    results.weights=pws.weights%exp(q);
    s=sum(results.weights);
    results.weights=results.weights/s;                                    
    return results;
}
