//Conjugate gradient algorithm for function maximization of a continuously
//differentiable real function f.value of real vectors of dimension p on a
//nonempty open convex set O of p-dimensional vectors.  Numerical differentiation is used.
//The step size is step.  
//The strict pseudoconcavity condition described in the document
//convergence.pdf is assumed to apply for the real number a.  For the starting
//vector start, it is assumed that the value of f.value at start exceeds a.
//Parameters used are defined in mparams.
//The maximum number of main iterations is mparams.maxit.
//The maximum number of secondary iterations per main iteration
//is mparams.maxits.
//The maximum fraction of a step toward a boundary is
//mparams.eta.
//For secondary iterations, the improvement check
//is mparams.gamma1<1.
//The cosine check is mparams.gamma2<1.
//The largest permitted step length is mparams.kappa>0.
//If a main iteration leads to a change of the function f less
//than mparams.tol, then iterations cease.
#include<armadillo>
using namespace std;
using namespace arma;
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

maxf2v conjgrad(const int & order, const params & mparams, const vec & start, const function<f2v(const int & , const vec &)> f);
f2v ngh(const int & , const double & , const vec & , 
    const function <f2v(const int & , const vec & )>);
maxf2v conjgradn(const int & order, const double & step, const params & mparams, const vec & start, const function<f2v(const int & , const vec &)> f)
{
     auto f2=[step,f](const int & order, const vec & x){return ngh(order,step,x,f);};    
     return conjgrad(order, mparams, start, f2);
}
