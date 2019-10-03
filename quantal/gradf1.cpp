//Log likelihood component and gradient
//for graded  model with response y and parameter beta.
//The distribution function used and its first  derivative
//are provided by f.
#include<armadillo>
using namespace arma;
using namespace std;

struct fd1bv
{
    double value;
    vec grad;
    
    bool fin;
};
struct fd1
{
    double value;
    double der1;
    
};


fd1bv gradf1(int y,vec beta,function <fd1(double)> f)
{

    double d;
    fd1bv results;
    fd1 result1;
    fd1 result2;
    if(min(diff(beta))<=0.0)
    {
        results.fin=false;
        return results;
    }
    results.fin=true;
    results.value=0.0;
    results.grad=zeros(beta.n_elem);
    
    if(y==beta.n_elem)
    {
        result1.value=1.0;
        result1.der1=0.0;
        
    }
    else
    {
        result1=f(beta(y));
    }
    if(y==0)
    {
        result2.value=0.0;
        result2.der1=0.0;
       
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
        
        
    }
    if(y>0)
    {
        results.grad(y-1)=-result2.der1/d;
        
        
       
        
    }
    
    
    return results;
}
