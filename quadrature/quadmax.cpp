//Fit quadratic function and locate its maximum.
//Data are in functdat.
//Output is a maxf2v struct.
#include<armadillo>
using namespace std;
using namespace arma;
struct maxf2v
{
    vec locmax;
    double max;
    vec grad;
    mat hess;
};
struct vecmat
{
    vec v;
    mat m;
};
struct fvecmat
{
    double value;
    vec v;
    mat m;
};
fvecmat trisym(const int & , const vec & );
maxf2v quadmax(const vecmat & pv)
{
    bool flag;
    int h,i,j,k,m,n,p;
    maxf2v results;
    n=pv.v.n_elem;
    m=pv.m.n_rows;    
    results.locmax.set_size(m);
    results.locmax.zeros();
    results.max=datum::nan;
    results.grad=zeros(m);
    results.hess.set_size(m,m);
    results.hess.eye(m,m);
    p=(m+1)*(m+2)/2;
    vec b(p);    
    mat x(n,p);
    fvecmat z;
    z.v.set_size(m);
    z.m.set_size(m,m);
    for(i=0;i<n;i++)
    {
        x(i,0)=1.0;
        h=m+1;
        for(j=0;j<m;j++)
        {
             x(i,j+1)=pv.m(j,i);
	     for(k=0;k<=j;k++)
             {
		 x(i,h)=x(i,k+1)*x(i,j+1);
                 h=h+1;
             }
        }
        
    }
    flag=solve(b,x,pv.v);
    if(!flag) return results;
    z=trisym(m,b);
    z.m=-z.m;
    if(!z.m.is_sympd()) return results;
    results.hess=-2.0*z.m;
    b=0.5*solve(z.m,z.v);
    results.max=z.value+0.5*dot(z.v,b);
    results.locmax=b;  
    return results;
}
            
            
            
