//Make MDIA adjustment to quadrature weights.
//Variables are defined as in kl.cpp and maxselect.cpp.
#include<armadillo>
using namespace arma;
using namespace std;
//Function, gradient, and Hessian.
struct pw{vec points; vec weights;};
struct maxf2v{vec locmax; double max; vec grad; mat hess;};
struct params{bool print; uword maxit; uword maxits; double eta;
    double gamma1; double gamma2; double kappa; double tol;};
maxf2v mdia(const int & , const params & , const char & , const vec & ,
    const mat & , const vec & );
pw mdiapw(const int & order, const params & mparams,
    const char & algorithm, const pw & pws, const mat & T, const vec & u){
    double s;
    maxf2v result;
    result=mdia(order, mparams, algorithm, pws.weights, T, u);
    pw results;
    results.points=pws.points;
    vec q=T*result.locmax;
    results.weights=pws.weights%exp(q);
    s=sum(results.weights);
    results.weights=results.weights/s;                                    
    return results;
}
