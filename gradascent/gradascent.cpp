//Gradient ascent algorithm for function maximization of a twice continuously
//differentiable real function f.value of real vectors of dimension p on a
//nonempty open convex set O of p-dimensional vectors.  The gradient of f.value
//is f.grad.  The integer order must be 1 or 2.  If order is 2, then the
//Hessian f.hess of f.value is found even though it is not used.
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
maxf2v maxf2vvar(const int & , const vec & , const f2v & );
maxf2v maxlinq2(const int & , const params & , const vec & , const maxf2v & ,
    const std::function <f2v(const int &, const vec &)> );
maxf2v gradascent(const int & order, const params & mparams, const vec & start,
    const function<f2v(const int &, const vec &)> f)
{
    f2v fy0;
    int i, p;
    maxf2v vary0, vary1;
    vec v;
    p=start.n_elem;
    fy0.grad.set_size(p);
    vary0.grad.set_size(p);
    vary0.locmax.set_size(p);
    vary1.grad.set_size(p);
    vary1.locmax.set_size(p);
    if(order>1)
    {
        fy0.hess.set_size(p,p);
        vary0.hess.set_size(p,p);
        vary1.hess.set_size(p,p);
    }
    v.set_size(p);
    v=start;
// Function settings at start.
    fy0=f(order, v);
    vary0=maxf2vvar(order, start, fy0);
// Return if starting impossible.
    if(isfinite(vary0.max)||mparams.maxit<=0) return vary0;
// Iterations.
    for(i=0;i<mparams.maxit;i++)
    {
// Stop if gradient of zero.
        v=vary0.grad;
        if(!any(v)) return vary0;
        if(norm(v,2)>mparams.kappa)v=(mparams.kappa/norm(v,2))*v;
// Line search.
        vary1 = maxlinq2(order, mparams, v, vary0, f);
        if(mparams.print)cout<<"Iteration="<<i<<", Function="<<vary1.max<<endl;
//  Convergence check
        if(vary1.max<vary0.max+mparams.tol) return vary1;
        vary0=vary1;
    }
    return vary1;
}
