//Make MDIA adjustment.  Variables are defined as in kl.cpp and maxselect.cpp.
#include<armadillo>
using namespace arma;
using namespace std;
//Function, gradient, and Hessian.
struct f2v{double value; vec grad; mat hess;};
struct maxf2v{vec locmax; double max; vec grad; mat hess;};
struct params{bool print; uword maxit; uword maxits; double eta; double gamma1;
    double gamma2; double kappa; double tol;};
f2v kl(const int & , const vec & , const mat & , const vec & , const vec & );
maxf2v maxselect(const int & , const params & , const char & ,
    const vec & , const function<f2v(const int &, const vec &)> );
maxf2v mdia(const int & order, const params & mparams,
    const char & algorithm, const vec & p, const mat & T, const vec & u) {
    vec start(u.n_elem);
    const function<f2v(const int & order,const vec & gamma)> f=
        [ &p, &T, &u](const int & order, const vec & gamma) mutable
        {return kl(order,p,T,u,gamma);};
    return maxselect(order, mparams, algorithm, start, f);
}
