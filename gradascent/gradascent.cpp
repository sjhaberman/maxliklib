//Gradient ascent algorithm for function maximization of a continuously
//differentiable real function f.value of real vectors of dimension p on a
//nonempty open convex set O of p-dimensional vectors.  The gradient of f.value
//is f.grad.
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
//The cosine check  mparams.gamma2 is not used.
//The largest permitted step length is mparams.kappa>0.
//If a main iteration leads to a change of the function f less
//than mparams.tol, then iterations cease.
#include<armadillo>
using namespace std;
using namespace arma;
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
maxf1v maxf1vvar(const vec &y,const f1v &fy);
maxf1v maxlinq(const params &mparams,const vec & v,maxf1v & vary0, function <f1v(vec &)> f);
maxf1v gradascent(const params & mparams,const vec & start, function<f1v(vec &)> f)
{
    f1v fy0;
    int i,p;
    maxf1v vary0,vary1;
    vec v;
    p=start.n_elem;
    fy0.grad.set_size(p);
    vary0.grad.set_size(p);
    vary0.locmax.set_size(p);
    vary1.grad.set_size(p);
    vary1.locmax.set_size(p);
    v.set_size(p);
    v=start;
// Function settings at start.
    fy0=f(v);
    vary0=maxf1vvar(start,fy0);
// Return if starting impossible.
    if(isnan(vary0.max)) return vary0;
// Iterations.
    for(i=0;i<mparams.maxit;i++)
    {
// Stop if gradient of zero.
        v=vary0.grad;
        if(!any(v)) return vary0;
        if(norm(v,2)>mparams.kappa)v=(mparams.kappa/norm(v,2))*v;
// Line search.
        vary1 = maxlinq(mparams,v,vary0,f);
//  Convergence check
        if(vary1.max<vary0.max+mparams.tol) return vary1;
        vary0=vary1;
    }
    return vary1;
}
