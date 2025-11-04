//Select maximization algorithm.  Choices are modified Newton-Raphson
//(algorithm='N'), modified Louis (algorithm='L'),
//conjugate gradient (algorithm='C'), and gradient ascent (algorithm='G').
//The function f.value maximized is always a
//continuously differentiable real function of real vectors of dimension p on a
//nonempty open convex set O of p-dimensional vectors.  The gradient of f.value
//is f.grad.  For the Newton-Raphson case,
//the function must have a continuous Hessian
//matrix f.hess.  The integer order must be 1, 2, or 3.
//If order is 2, then f.hess, the
//Hessian (order=2) or approximate Hessian (order=3) of f.value is found
//even if it is not
//used for the gradient ascent or conjugate gradient cases.
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
//is mparams.gamma2>1.
//The cosine check is mparams.gamma1<1.
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
maxf2v conjgrad(const int &, const params & , const vec & ,
    const function<f2v(const int & , const vec & )> f);
maxf2v gradascent(const int &, const params & , const vec & ,
    const function<f2v(const int & , const vec & )> f);
maxf2v nrv(const int &, const params & , const vec & ,
    const function<f2v(const int & , const vec & )> f);
maxf2v maxselect(const int & order, const params & mparams,
    const char & algorithm,
    const vec & start, const function<f2v(const int & , const vec & )> f)
{
    
    maxf2v results;
    int p;
    p=start.n_elem;
    results.locmax.set_size(p);
    results.grad.set_size(p);
    if(order>1)results.hess.set_size(p,p);
    if(algorithm=='N'||algorithm=='L')results=nrv(order, mparams, start, f);
    if(algorithm=='C')results=conjgrad(order, mparams, start, f);
    if(algorithm=='G')results=gradascent(order, mparams, start, f);
    return results;
}
