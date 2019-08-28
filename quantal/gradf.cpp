//Log likelihood component, gradient, and Hessian
//for graded logit model with response y and parameter beta.
#include<armadillo>
using namespace arma;
using namespace std;

struct fd2bv
{
    double value;
    vec grad;
    mat hess;
    bool fin;
};
struct fd2
{
    double value;
    double der1;
    double der2;
};


fd2bv gradf(int y,vec beta,function <fd2(double)> f)
{

    double d;
    fd2bv results;
    fd2 result1;
    fd2 result2;
    if(min(diff(beta))<=0.0)
    {
        results.fin=false;
        return results;
    }
    results.fin=true;
    results.value=0.0;
    results.grad=zeros(beta.n_elem);
    results.hess=zeros(beta.n_elem,beta.n_elem);
    if(y==beta.n_elem)
    {
        result1.value=1.0;
        result1.der1=0.0;
        result1.der2=0.0;
    }
    else
    {
        result1=f(beta(y));
    }
    if(y==0)
    {
        result2.value=0.0;
        result2.der1=0.0;
        result2.der2=0.0;
    }
    else
    {
        result2=f(beta(y-1));
    }
    d=result1.value-result2.value;
    results.value=log(d);
    if(y<beta.n_elem)
    {
        results.grad(y)=result1.der1/d;
        results.hess(y,y)=result1.der2/d-results.grad(y)*results.grad(y);
        
    }
    if(y>0)
    {
        results.grad(y-1)=-result2.der1/d;
        results.hess(y-1,y-1)=-result2.der2/d-results.grad(y-1)*results.grad(y-1);
        if(y<beta.n_elem)
        {
            results.hess(y-1,y)=-results.grad(y)*results.grad(y-1);
            results.hess(y,y-1)=results.hess(y-1,y);
        }
        
    }
    
    
    return results;
}
