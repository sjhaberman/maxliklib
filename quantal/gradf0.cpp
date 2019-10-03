//Log likelihood component
//for graded  model with response y and parameter beta.
//The distribution function used and its first  derivative
//are provided by f.
#include<armadillo>
using namespace arma;
using namespace std;

struct fd0bv
{
    double value;
   
    
    bool fin;
};



fd0bv gradf0(int y,vec beta,function <double(double)> f)
{

    double d;
    fd0bv results;
    double result1;
    double result2;
    if(min(diff(beta))<=0.0)
    {
        results.fin=false;
        return results;
    }
    results.fin=true;
    results.value=0.0;
    
    
    if(y==beta.n_elem)
    {
        result1=1.0;
        
        
    }
    else
    {
        result1=f(beta(y));
    }
    if(y==0)
    {
        result2=0.0;
        
       
    }
    else
    {
        result2=f(beta(y-1));
    }
    d=result1-result2;
    results.value=log(d);
    
    
    
    return results;
}
