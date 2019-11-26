//Cyclic ascent algorithm for function maximization for a strictly
//concave function f.value on the space of p-dimensional vectors for a positive
//integer p.
//For element j of the p-dimensional vector x, the function f(j,x)
//is a struct object with
//elements value, der1, and der2, where value is the value of the strictly
//concave function f.value at x,der1 is the partial derivative of f.value
//at x with
//respect the element j of x, and der2 is the second partial derivative of
//f.value at x for element j of x.
//The Hessian is assumed continuous and negative-definite.
//The function must approach negative infinitely whenever
//the vector norm of the function argument
//approaches infinity.
//This condition holds if the function achieves its supremum.
//Maximum number of iteration cycles is maxit.
//Maximum number of secondary iterations per individual ascent step is maxits.
//If change in approximation to maximum for a full cycle is less
//than tol, then iterations cease.
//The largest permitted step is stepmax>0.
//The improvement check is b>1.
//See Chapter 3 of Analysis of Frequency Data by S. J. Haberman
//for a very closely-related
//algorithm.



#include<armadillo>
using namespace std;
using namespace arma;
struct fd2
{
    double value;
    double der1;
    double der2;
};

struct cyclevar
{
    vec locmax;
    double max;
    vec grad;
    vec diaghess;
};
struct cyclevarp
{
    vec locmax;
    double max;
    double der1;
    double der2;
};
cyclevarp cyclevarf(int j, vec x,function<fd2(int,vec)> f)
{
    fd2 resultf;
    cyclevarp result;
    resultf=f(j,x);
    result.locmax=x;
    result.max=resultf.value;
    result.der1=resultf.der1;
    result.der2=resultf.der2;
    return result;
};
cyclevarp nrcv(const int,const double,const double,int,cyclevarp,
              function<fd2(int,vec)>);
cyclevar cyclic(const int maxit,const int maxits,const double tol,const vec start,
        const double stepmax,const double b,
        function<fd2(int,vec)> f)
{
    
    
   
    double fx;
    int i,j;
    cyclevarp varxp;
    vec x;
    cyclevar varx;
// Old value of location is x.
    x=start;
    
    varx.grad=zeros(x.n_elem);
    varx.diaghess=varx.grad;
// Cycle
    for(i=0;i<maxit;i++)
    {
        
// Iteration within cycle
        for(j=0;j<start.n_elem;j++)
        {
// Function settings.
            varxp=cyclevarf(j,x,f);
            if(j==0)fx=varxp.max;
            varx.locmax=varxp.locmax;
            varx.max=varxp.max;
            varx.grad(j)=varxp.der1;
            varx.diaghess(j)=varxp.der2;
// Stop cycle if gradient of zero.
            if(varxp.der1==0.0) continue;
            
// Iterations for stage j of cycle.
            varxp=nrcv(maxits,stepmax,b,j,varxp,f);
            x=varxp.locmax;
            varx.locmax=varxp.locmax;
            varx.max=varxp.max;
            varx.grad(j)=varxp.der1;
            varx.diaghess(j)=varxp.der2;
        }
        
// Check for convergence.
        if(varx.max-fx<tol) return varx;

    }
    return varx;
}

