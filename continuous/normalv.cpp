//Log likelihood component and its gradient and hessian matrix
//for multivariate normal model with vector response y of dimension
//d and parameter vector beta with d(d+3)/2 elements.
//The first d elements of beta provide a location shift v and the next elements
//are unpacked with unpack.cpp to yield a d by d positive-definite symmetric matrix m.
//The density f(y) is det(m)g(v+my), where g is the density of the standard
//multivariate normal distribution of dimension d.
#include<armadillo>
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
struct vecmat
{
    vec v;
    mat m;
};
struct resp
{
  ivec iresp;
  vec dresp;
};
vec pack(const vecmat & u);
vecmat unpack(const int &, const vec &);
f2v normalv(const int & order, const resp & y, const vec & beta)
{
    double x;
    int i,i1,j,j1,k,k1,m,n,n2;
    vec z;
    vecmat b,dec;
    f2v results;
    m=beta.n_elem;
    if(order>0) results.grad.set_size(m);
    if(order>1) results.hess.set_size(m,m);
    n=y.dresp.n_elem;
    z.set_size(n);
    
    mat f,ff;
    vec d;
    dec.v.set_size(n);
    dec.m.set_size(n,n);
    dec=unpack(n,beta);
    z=dec.v+dec.m*y.dresp;
    if(!dec.m.is_sympd())
    {
        results.value=datum::nan;
        if(order>0) results.grad.fill(datum::nan);
        if(order>1) results.hess.fill(datum::nan);
        return results;
    }
    x=double(n);
    results.value=log_det_sympd(dec.m)-0.5*(dot(z,z)+x*log(datum::tau));
    if(order==0)return results;  
    b.v.set_size(n);
    b.m.set_size(n,n);
    f.set_size(n,n);
    
    f=0.5*eye(n,n)+0.5*ones(n,n);
    b.v=-z;
    mat a(n,n),aa(n,n);
    a=inv_sympd(dec.m);
    b.m=f%(a-0.5*(z*trans(y.dresp)+y.dresp*trans(z)));
    results.grad=pack(b);
    if(order==1)return results;
    
   
    results.hess.submat(0,0,n-1,n-1)=-eye(n,n);
    uvec ind(m-n);
    k=n;
    for(i=0;i<n;i++)
    {
         for(i1=0;i1<=i;i1++)
         {
             for(j=0;j<n;j++)
             {
                 if(i==i1)
                 {
                      if(j==i)
                      {
                          results.hess(j,k)=-y.dresp(j);
                      }
                      else
                      {
                          results.hess(j,k)=0.0;
                      }
                 }
                 else
                 {
                      if(j==i)
                      {
                          results.hess(i1,k)=-0.5*y.dresp(j);
                      }
                      else
                      {
                          if(j==i1)
                          {
                               results.hess(i,k)=-0.5*y.dresp(j);
                          }
                          else
                          {
                               results.hess(j,k)=0.0;
                          }
                      }
                 }

                 
             }
             k=k+1;
        }
    } 
    results.hess.submat(n,0,m-1,n-1)=trans(results.hess.submat(0,n,n-1,m-1));            
    k=0;
    for(i=0;i<n;i++)
    {
        for(j=0;j<=i;j++)
        {
             k1=i*n+j;  
             
             ind(k)=k1;
             
             k=k+1;
        }
    }
    ff.set_size(n,n);
    ff=eye(n,n);
    
    for(i=0;i<n;i++)
    {
        for(j=0;j<=i;j++)
        {
           k=n+((i*(i+1))/2)+j;
           for(i1=0;i1<n;i1++)
           {
               for(j1=0;j1<=i1;j1++)
               {
                   
                   k1=n+((i1*(i1+1))/2)+j1;  
                   results.hess(k,k1)=-0.5*(a(i,j1)*a(i1,j)+a(i,i1)*a(j,j1));
                   results.hess(k,k1)=results.hess(k,k1)
                       -0.25*(ff(i,i1)*y.dresp(j)*y.dresp(j1)
                           +ff(i,j1)*y.dresp(j)*y.dresp(i1)
                           +ff(j,j1)*y.dresp(i1)*y.dresp(i)
                           +ff(j,i1)*y.dresp(i)*y.dresp(j1));
               }
           }
       }
   }   

    return results;
}
