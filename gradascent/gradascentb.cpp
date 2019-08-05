//Gradient ascent algorithm for function maximization of a continuously
//differentiable real function of real vectors of dimension p on an open convex set.
//The function, gradient, and Hessian are
//f.value, and f.grad.
//For any vecot x in the convex set and any nonzero vector v, f.value(x+av) approaches minus
//infinity whenever for the real number a, x+av approaches the upper of lower bound
//of the set A(a) of x+av is the convex set.
//Maximum number of main iterations is maxit.
//Maximum number of secondary iterations per main iteration is maxits.
//If change in approximation to maximum is less
//than tol, then iterations cease.
//The largest permitted step is stepmax>0.
//The improvement check is b>1.



#include<armadillo>
using namespace std;
using namespace arma;

struct fd1bv
{
    double value;
    vec grad;
    bool fin;
};

struct twopointgvarb
{
    vec locmax;
    double max;
    vec grad;
    bool fin;
};
twopointgvarb twopointgvarbf(vec,function<fd1bv(vec)>);

twopointgvarb twopointbg(const int,
            vec,twopointgvarb,const double,const double,
            function<fd1bv(vec)>);
twopointgvarb gradascentb(const int maxit,const int maxits,const double tol,const vec start,
        const double stepmax,const double b,
        function<fd1bv(vec)> f)
{
    
    
   
    double c,d;
    int i;
    
    twopointgvarb varx,vary;
    vec v,x,y;

// Function settings at start.
    varx=twopointgvarbf(start,f);
    if(!varx.fin) return varx;

// Iterations.
    for(i=0;i<maxit;i++)
    {

// Stop if gradient of zero.
        if(!any(varx.grad)) return varx;
// Direction.
        v=varx.grad;
        

// Step truncation.
        
        d=norm(v,"inf");
        if(d<=stepmax)
        {
            c=1.0;
        }
        else
        {
            c=stepmax/d;
        }
        v=c*v;
        
        vary = twopointbg(maxits,v,varx,stepmax,b,f);
        
        if(vary.max<varx.max+tol) return vary;

        varx=vary;
        x=y;
    }
    return vary;
}

