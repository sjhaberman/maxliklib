//Make MDIA adjustment.  Variables are defined as in kl.cpp and nrv.cpp.
#include<armadillo>
using namespace arma;
using namespace std;
//Function, gradient, and Hessian.
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
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
f2v kl(const int & , const vec & , const mat & , const vec & , const vec & );
maxf2v nrv(const int & , const params & , const vec & ,
    const function<f2v(const int &, const vec &)> );
maxf2v mdia(const vec & p, const mat & T,
    const vec & u, const params & mparams)
{
    maxf2v results;
    maxf2v result;
    int order=2;
    vec start(u.n_elem);
    const function<f2v(const int & order,const vec & gamma)> f=
        [ &p, &T, &u](const int & order,const vec & gamma) mutable
        {return kl(order,p,T,u,gamma);};    
    return result=nrv(order, mparams, start, f);
}
