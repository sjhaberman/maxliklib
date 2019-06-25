//Gradient ascent algorithm for function maximization of a continuously
//differentiable real function of real vectors of dimension p.
//The function, gradient, and Hessian are
//f.value, and f.grad.
//The function should approach negative infinity whenever the vector norm of the function argument
//approaches infinity.
//Maximum number of main iterations is maxit.
//Maximum number of secondary iterations per main iteration is maxits.
//If change in approximation to maximum is less
//than tol, then iterations cease.
//The largest permitted step is stepmax>0.
//The improvement check is b>1.



#include<armadillo>
using namespace std;
using namespace arma;

struct fd1v
{
    double value;
    vec grad;
};

struct twopointgvar
{
    vec locmax;
    double max;
    vec grad;
};
twopointgvar twopointgvarf(vec x,function<fd1v(vec)> f)
{
    fd1v resultf;
    twopointgvar result;
    resultf=f(x);
    result.locmax=x;
    result.max=resultf.value;
    result.grad=resultf.grad;
    return result;
};

twopointgvar twopointg(const int,
            vec,twopointgvar,const double,const double,
            function<fd1v(vec)>);
twopointgvar gradascent(const int maxit,const int maxits,const double tol,const vec start,
        const double stepmax,const double b,
        function<fd1v(vec)> f)
{
    
    
   
    double c,d;
    int i;
    
    twopointgvar varx,vary;
    vec v,x,y;

// Function settings at start.
    varx=twopointgvarf(start,f);

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
        
        vary = twopointg(maxits,v,varx,stepmax,b,f);
        
        if(vary.max<varx.max+tol) return vary;

        varx=vary;
        x=y;
    }
    return vary;
}

