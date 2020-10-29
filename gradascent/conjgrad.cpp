//Conjugate gradient algorithm for function maximization of a continuously
//differentiable real function f.value of real vectors of dimension p on a
//nonempty open convex set O of p-dimensional vectors.  The gradient of f.value
//is f.grad.
//The strict pseudoconcavity condition described in the document
//convergence.pdf is assumed to apply for the real number a.  For the starting
//vector start, it is assumed that the value of f.value at start exceeds a.
//Parameters used are defined in gaparams.
//The maximum number of main iterations is gaparams.maxit.
//The maximum number of secondary iterations per main iteration
//is gaparams.maxits. The double function on O used for step sizes for
//numerical differentiation is gaparams.c.
//The maximum fraction of a step toward a boundary is
//gaparams.eta.
//For secondary iterations, the improvement check
//is gaparams.gamma1<1.
//The cosine check is gaparams.gamma2<1.
//The largest permitted step length is gaparams.kappa>0.
//If a main iteration leads to a change of the function f less
//than gaparams.tol, then iterations cease.
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
struct paramga
{
    int maxit;
    int maxits;
    function<double(vec)> c;
    double eta;
    double gamma1;
    double gamma2;
    double kappa;
    double tol;
};
maxf1v maxf1vvar(const vec &y,const f1v &fy);
maxf1v maxlin(const paramga &gaparams,const vec & v,maxf1v & vary0,const function <f1v(vec &)> f);
maxf1v conjgrad(const paramga &gaparams,const vec &start,const function<f1v(vec &)> f)
{
    double tau;
    f1v fy0;
    int i,p;
    maxf1v vary0,vary1;
    vec v,v1,v2;
    p=start.n_elem;
    fy0.grad.set_size(p);
    vary0.grad.set_size(p);
    vary0.locmax.set_size(p);
    vary1.grad.set_size(p);
    vary1.locmax.set_size(p);
    v.set_size(p);
    v1.set_size(p);
    v2.set_size(p);
    vary0.locmax=start;
    fy0=f(vary0.locmax);
// Function settings at start.
    vary0=maxf1vvar(start,fy0);
// Return if starting impossible.
    if(isnan(vary0.max)) return vary0;
// Iterations.
    for(i=0;i<gaparams.maxit;i++)
    {
// Stop if gradient of zero.
        v1=vary0.grad;
        if(!any(v1)) return vary0;
        if(i>0)
        {
            tau=dot(v1-v2,v1)/dot(v1,v1);
            v=v1+tau*v;
// Check for acceptable direction.
            if(dot(v,v1)<gaparams.gamma2*norm(v,2)*norm(v1,2))v=v1;
        }
        else
        {
            v=v1;
        }
// Line search.
        vary1 = maxlin(gaparams,v,vary0,f);
//  Convergence check
        if(vary1.max<vary0.max+gaparams.tol) return vary1;
        vary0=vary1;
        v2=v1;
    }
    return vary1;
}