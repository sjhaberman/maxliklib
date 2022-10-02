//Quadratic fit procedure for a line search.  Second derivatives are
//not considered but are found if order>1.
//Argument definitions found in gradascent.cpp are used.
//The function f.value is a continuously differentiable function on a
//nonempty open convex set O of p-dimensional vectors.  The gradient of f.value
//is f.grad.  The Hessian matrix is f.hess.
//The strict pseudoconcavity condition described in the document
//convergence.pdf is assumed to apply for the real number a less than the
//value of f.value at the starting vector.
//Parameters used are defined in mparams.  The following members of gaparams
//are relevant for the line search.
//The maximum number of secondary iterations per main iteration
//is mparams.maxits.
//The maximum fraction of a step toward a boundary is
//mparams.eta.
//For secondary iterations, the improvement check
//is mparams.gamma1<1.
//The largest permitted step length is mparams.kappa>0.
#include<armadillo>
using namespace arma;
struct bounds
{
     double lower;
     double upper;
};
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
maxf2v maxf2vvar(const int & , const vec &,const f2v &);
double modit(const double &, const double &, const double &,
             const double &, const bounds & );
bounds rebound(const double &, const double &, const bounds & );
double maxquad(const double & , const double & , const double & , const double & ,
     const double & , const double & );
maxf2v maxlinq2(const int & order, const params & mparams, const vec & v,
    const maxf2v & vary0, const std::function<f2v(const int &, const vec &)>f)
{
// Values alpha1 and alpha2 correspond respectively to vectors y1 and y2.
// Bounds on optimal line position are lower and upper.
// The derivative in the direction v at y1 is der1, and
// deltaf is the change in f.value.
// Line searching is based on a quadratic approximation based on
// f.value(y1), f.value(y2), and der1.
// By reordering if needed, f.value(y1)is at least f.value(y2).
    bounds b;
    double alpha1, alpha2, alpha3, deltaf, der1, der2, der3, normv,
    stepmax, width;
// vy1 gives value, gradient, and Hessian of f at y1,
// vy2 is correponding information at y2,
// and vy3 gives correponding information at y3.
    f2v fy2;
    vec y0, y1, y2, y3;
    maxf2v  vary1, vary2, vary3;
//  i counts secondary iterations.
    int i,p;
    p=v.n_elem;
    y0.set_size(p);
    y1.set_size(p);
    y2.set_size(p);
    y3.set_size(p);
    vary1.locmax.set_size(p);
    vary1.grad.set_size(p);
    if(order>1) vary1.hess.set_size(p,p);
    vary1.max=vary0.max;
    vary1.locmax=vary0.locmax;
    vary1.grad=vary0.grad;
    if(order>1) vary1.hess=vary0.hess;
    vary2.locmax.set_size(p);
    vary2.grad.set_size(p);
    if(order>1) vary2.hess.set_size(p,p);
    fy2.grad.set_size(p);
    if(order>1) fy2.hess.set_size(p,p);
// Stuck if starting value out of domain.
    if(isnan(vary0.max)) return vary1;
//  Start at 0 and 1.
    alpha1=0.0;
// Find maximum step size stepmax for line.
    stepmax=mparams.kappa/norm(v,2);
    alpha2=1.0;
    y0=vary0.locmax;
    y1=y0;
    der1=dot(v,vary1.grad);
    y2=y0+alpha2*v;
    fy2=f(order, y2);
// b.lower is lower bound for location of maximum.
    b.lower=0;
// b.upper is upper bound for location of maximum.
    b.upper=INFINITY;
// modify bounds if needed so that alpha2 in range.
    while(isnan(fy2.value))
    {
        b.upper=alpha2;
        alpha2=(1.0-mparams.eta)*alpha1+mparams.eta*alpha2;
        y2=y0+alpha2*v;
        fy2=f(order, y2);
    }
    vary2=maxf2vvar(order, y2,fy2);
// Switch if needed.
    if(vary2.max>vary1.max)
    {
        alpha3=alpha1;
        alpha1=alpha2;
        alpha2=alpha3;
        y3=y1;
        y1=y2;
        y2=y3;
        vary3.max=vary1.max;
        vary3.locmax=vary1.locmax;
        vary3.grad=vary1.grad;
        if(order>1) vary3.hess=vary1.hess;
        vary1.max=vary2.max;
        vary1.locmax=vary2.locmax;
        vary1.grad=vary2.grad;
        if(order>1) vary1.hess=vary2.hess;
        vary2.max=vary3.max;
        vary2.locmax=vary3.locmax;
        vary2.grad=vary3.grad;
        if(order>1) vary2.hess=vary3.hess;
        der1=dot(v,vary1.grad);
        deltaf=vary1.max-vary0.max;
        if(deltaf>=mparams.gamma1*fabs(der1*alpha1))return vary1;
    }
    else
    {
        b.upper=alpha2;
    }
// Up to maxit iterations.
    for(i=0;i<mparams.maxits;i++)
    {
// See if maximum found exactly at this point.
        if(der1==0.0)return vary1;
// Revise bounds.
        if(alpha1>0.0) b = rebound(alpha1, der1, b);
        alpha2=maxquad(alpha1, alpha2, vary1.max, vary2.max, der1, stepmax);
        alpha2=modit(mparams.eta, alpha1, alpha2, stepmax, b);
        y2=y0+alpha2*v;
        fy2=f(order, y2);
// modify bounds if needed so that alpha2 in range.
        while(isnan(fy2.value))
        {
            b.upper=alpha2;
            alpha2=(1.0-mparams.eta)*alpha1+mparams.eta*alpha2;
            y2=y0+alpha2*v;
            fy2=f(order, y2);
        }
        vary2=maxf2vvar(order, y2, fy2);
// Switch if needed.
        if(vary2.max>vary1.max)
        {
            alpha3=alpha1;
            alpha1=alpha2;
            alpha2=alpha3;
            y3=y1;
            y1=y2;
            y2=y3;
            vary3.max=vary1.max;
            vary3.locmax=vary1.locmax;
            vary3.grad=vary1.grad;
            if(order>1) vary3.hess=vary1.hess;
            vary1.max=vary2.max;
            vary1.locmax=vary2.locmax;
            vary1.grad=vary2.grad;
            if(order>1) vary1.hess=vary2.hess;
            vary2.max=vary3.max;
            vary2.locmax=vary3.locmax;
            vary2.grad=vary3.grad;
            if(order>1) vary2.hess=vary3.hess;
            der1=dot(v,vary1.grad);
// Convergence check.
            deltaf=vary1.max-vary0.max;
            if(deltaf>=mparams.gamma1*fabs(der1*alpha1))return vary1;
            b = rebound(alpha1,der1,b);
        }
    }
    return vary1;
}
