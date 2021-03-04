//Newton-Raphson algorithm for function maximization of a continuously
//differentiable real function of real vectors of dimension p on a
//nonempty open convex set O of p-dimensional vectors.  The gradient of f.value
//is f.grad.  The Hessian matrix is f.hess.
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
//The cosine check  is mparams.gamma2.
//The largest permitted step length is mparams.kappa>0.
//If a main iteration leads to a change of the function f less
//than mparams.tol, then iterations cease.
#include<armadillo>
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
    int maxit;
    int maxits;
    double eta;
    double gamma1;
    double gamma2;
    double kappa;
    double tol;
};
maxf2v maxf2vvar(const int & , const vec & , const f2v & );
maxf2v maxlinq2(const int & , const params &, const vec & , const maxf2v & ,
    const std::function <f2v(const int &, const vec & )> f);
maxf2v nrv(const int & order, const params & mparams, const vec & start, const std::function<f2v(const int &, const vec &)> f)
{
    f2v fy0;
    int i, p;
    p=start.n_elem;
    fy0.grad.set_size(p);
    fy0.hess.set_size(p,p);
    maxf2v vary0,vary1;
    vary0.grad.set_size(p);
    vary0.locmax.set_size(p);
    vary0.hess.set_size(p,p);
    vary1.grad.set_size(p);
    vary1.locmax.set_size(p);
    vary1.hess.set_size(p,p);
    vec v;
    v.set_size(p);
// Function settings at start.
    v=start;
    fy0=f(order, v);
    vary0=maxf2vvar(order, v, fy0);
// Return if starting impossible.
    if(isnan(vary0.max)||mparams.maxit<=0) return vary0;
// Iterations.
    for(i=0;i<mparams.maxit;i++)
    {
// Stop if gradient of zero.
        if(!any(vary0.grad)) return vary0;
// Find Newton-Raphson step if possible.
        if ((-vary0.hess).is_sympd())
        {
            v=solve(-vary0.hess,vary0.grad);
            if(dot(v,vary0.grad)<mparams.gamma1*norm(v,2)*norm(vary0.grad,2))v=vary0.grad;
        }
        else
        {
            v=vary0.grad;
        }
// Line search.
        if(norm(v,2)>mparams.kappa)v=(mparams.kappa/norm(v,2))*v;
        vary1 = maxlinq2(order, mparams,v,vary0,f);
//  Convergence check
        if(vary1.max<vary0.max+mparams.tol) return vary1;
        vary0=vary1;
    }
    return vary1;
}
