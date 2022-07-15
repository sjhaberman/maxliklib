//Fit quadratic function and locate its maximum.
//Data are in functdat.
//Output is a maxf2v struct.
#include<armadillo>
using namespace arma;
struct functdt
{
    double value;
    vec loc;
};
struct maxf2v
{
    vec locmax;
    double max;
    vec grad;
    mat hess;
};
struct fvecmat
{
    double value;
    vec v;
    mat m;
};
fvecmat trisym(const int & , const vec & );
maxf2v quadmax(const std::vector<functdt> & functdat)
{
    bool flag;
    int h,i,j,k,m,n,p;
    maxf2v results;
    n=functdat.size();
    m=functdat[0].loc.n_elem;
    p=(m+1)*(m+2)/2;
    vec y(n);
    vec v(p);
    mat x(n,p);
    vec b(p);
    fvecmat z;
    z.v.set_size(m);
    z.m.set_size(m,m);
    results.locmax.set_size(m);
    results.grad=zeros(m);
    results.hess.set_size(m,m);
    for(i=0;i<n;i++)
    {
        y(i)=functdat[i].value;
        x(i,0)=1.0;
        h=m+1;
        for(j=0;j<m;j++)
        {
             x(i,j+1)=functdat[i].loc(j);
	     for(k=0;k<=j;k++)
             {
		x(i,h)=x(i,k+1)*x(i,j+1);
                 h=h+1;
             }
        }
        
    }
    flag=solve(b,x,y);
    if(!flag)
    {
       results.max=datum::nan;
       return results;
    }
    else
    {
       z=trisym(m,b);
       z.m=-z.m;
       if(z.m.is_sympd())
       {
             results.locmax=0.5*solve(z.m,z.v);
             results.max=z.value+0.5*dot(z.v,results.locmax);
             results.hess=-2.0*z.m;
       }
       else
       {
            results.max=datum::nan;
            return results;
       }
    }
    return results;
}
            
            
            
