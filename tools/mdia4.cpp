//Adjust symmetric distribution to fit variance of 1 and fourth moment of 3.
#include<armadillo>
using namespace arma;
using namespace std;
//Function, gradient, and Hessian.
struct pw
{
    vec points;
    vec weights;
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
pw mdiapw(const pw & , const mat & , const vec & , const params & );
pw mdia4(const pw & pws)
{
    pw results;
    params mparams;
    mparams.print=true;
    mparams.maxit=100;
    mparams.maxits=10;
    mparams.eta=0.5;
    mparams.gamma1=0.1;
    mparams.gamma2=0.1;
    mparams.kappa=3.0;
    mparams.tol=0.000001;
    int n,order=2;
    n=pws.points.n_elem;  
    results.points.set_size(n);
    results.weights.set_size(n);
    vec start(2,fill::zeros);
    mat T(n,2);
    T.col(0)=square(pws.points);
    T.col(1)=square(T.col(0));
    vec u(2);
    u(0)=1.0;
    u(1)=3.0;
    results=mdiapw(pws,T,u,mparams);
    return results;
}
