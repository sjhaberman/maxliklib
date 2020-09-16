//Approximate Newton-Raphson procedure for a line search.
//The algorithm is based on numerical
//differentiation.  Argument definitions found in gradascent.cpp are used.
//The function f.value is a continuously differentiable function on a
//nonempty open convex set O of p-dimensional vectors.  The gradient of f.value
//is f.grad.
//The strict pseudoconcavity condition described in the document
//convergence.pdf is assumed to apply for the real number a less than the
//value of f.value at the starting vector.
//Parameters used are defined in gaparams.  The following members of gaparams
//are relevant for the line search.
//The maximum number of secondary iterations per main iteration
//is gaparams.maxits. The double function on O used for step sizes for
//numerical differentiation is gaparams.c.
//The maximum fraction of a step toward a boundary is
//gaparams.eta.
//For secondary iterations, the improvement check
//is gaparams.gamma1<1.
//The largest permitted step length is gaparams.kappa>0.
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
double modit(const double &eta,const double &alpha0,const double &alpha1,
             const double &stepmax,const double &lower,const double &upper);
void rebound(const double &y,const double &der,double &lower,double &upper);
maxf1v maxlin(const paramga & gaparams,const vec &v,maxf1v & vary0,function<f1v(vec)>f)
{
// Old value alpha1 corresponds to vector y1.  New value alpha1 corresponds
// to vector y2. alphaw is used for numerical differentiation and corresponds
// to yw.  Bounds on optimal line position are lower and upper.
// alphaw is alpha1+width.  The Euclidean norm of the direction vector v is
// normv.  The maximum allowed change from alpha1 to alpha2 is stepmax.
// The derivative along the line at alpha1 is der1, and the derivative at
// alpha2 is der2.  The approximate second derivative at alpha1 is sder1.  deltaf
// is the change in f.value.
    double alpha1,alpha2,alphaw,deltaf,der1,der2,lower,normv,
    sder1,stepmax,upper,width;
// vy1 gives value and gradient of f at y1, vy2 is correponding information at y2,
// and vyw gives correponding information at yw.
    f1v fy1,fy2,fyw;
    vec y0,y1,y2,yw;
    maxf1v  result,vary1;
//  i counts secondary iterations.
    int i;
    result=vary0;
// Stuck if starting value out of domain.
    if(isnan(result.max)) return result;
//  Start at 0.
    alpha1=0.0;
    y0=vary0.locmax;
// lower is lower bound for location of maximum.
    lower=0;
// upper is upper bound for location of maximum.
    upper=INFINITY;
// Find maximum step size stepmax for line.
    normv=norm(v,2);
    stepmax=gaparams.kappa/normv;
// Up to maxit iterations.
    for(i=0;i<gaparams.maxits;i++)
    {
// Initialize vary1 if needed.
        if(i==0)
        {
            vary1=vary0;
            y1=y0;
            der1=dot(v,vary1.grad);
        }
// width is interval width for differentiation.
        width=gaparams.c(y1+width*v)/normv;
// Set upper bound for numerical differentiation.
        alphaw=alpha1+width;
        yw=y0+alphaw*v;
        fyw=f(yw);
// See if maximum found exactly at this point.
        if(der1==0.0)return result;
// Revise bounds.
        rebound(alpha1,der1,lower,upper);
// Approximate second derivative at y1.
        sder1=(dot(v,fyw.grad)-der1)/width;
// The proposed step.
        if(sder1<0.0)
        {
// Use approximation of second derivative if suitable.
            alpha2=alpha1-der1/sder1;
        }
        else
        {
// Just take a large step.
            alpha2=alpha1+stepmax;
        }
// Truncate if needed.
        alpha2=modit(gaparams.eta,alpha1,alpha2,stepmax,lower,upper);
// New tentative vector value and function values.
        y2=y0+alpha2*v;
        fy2=f(y2);
        result=maxf1vvar(y2,fy2);
        if(isnan(result.max))
        {
            der2=dot(v,result.grad);
            if(der2==0.0)return result;
            rebound(alpha2,der2,lower,upper);
        }
// Act if alpha2 is out of range.
        while(isnan(result.max))
        {
// Shrink and try again.
            alpha2=(1.0-gaparams.eta)*alpha1+gaparams.eta*alpha2;
            y2=y0+alpha2*v;
            fy2=f(y2);
            result=maxf1vvar(y2,fy2);
            if(isnan(result.max))
            {
                der2=dot(v,result.grad);
                if(der2==0.0)return result;
                rebound(alpha2,der2,lower,upper);
            }
            else
            {
                if(alpha2>alpha1)
                {
                    upper=alpha2;
                }
                else
                {
                    lower=alpha2;
                }
            }
        }
// Convergence check.
        deltaf=result.max-vary0.max;
        if(deltaf>=gaparams.gamma1*fabs(der2*alpha2))return result;
// Prepare for next step.
        y1=y2;
        alpha1=alpha2;
        vary1=result;
        der1=der2;
    }
    return result;
}

