//Regression of sum of squares of y(j,k)-beta(j)*beta(k) over selected pairs j and k such that
//j is unequal to k.
//Algorithm uses minus the sum of squares and may 
//be gradient ascent (G), conjugate gradient (C), Newton-Raphson (N), or Lous (L). 
#include<armadillo>
using namespace std;
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
struct resp
{
  ivec iresp;
  vec dresp;
};
f2v regprodf(const int & order, const field<resp> & y, const vec & beta)
{
    int d, i, j, k, n, p;
    double diff, a, b;
    n=y.n_elem;
    p=beta.n_elem;
    f2v result;
    result.value=0.0;
    if(order>0)
    {
         result.grad.set_size(p);
         result.grad.zeros();
    }
    if(order>1)
    {
         result.hess.set_size(p,p);
         result.hess.zeros();
    }
    for(i=0;i<n;i++)
    {
         j=y(i).iresp(0);
         k=y(i).iresp(1);
         diff=y(i).dresp(0)-beta(j)*beta(k);
         result.value=result.value-diff*diff;
         if(order>0)
         {
              result.grad(j)=result.grad(j)+2.0*beta(k)*diff;
              result.grad(k)=result.grad(k)+2.0*beta(j)*diff;
         }
        if(order>1)
        {
             result.hess(j,k)=-2.0*beta(j)*beta(k);
             if(order==2)result.hess(j,k)=result.hess(j,k)+2.0*diff;
             result.hess(k,j)=result.hess(j,k);
             result.hess(j,j)=result.hess(j,j)-2.0*beta(k)*beta(k);
             result.hess(k,k)=result.hess(k,k)-2.0*beta(j)*beta(j);
        }
    }
    return result;    
}
