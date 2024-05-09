//Starting values for generalized partial credit model with a standard normal
//latent variable.  The elementary case is considered in which all individuals
//have the same items and responses are nonnegative integers.
//responses is the data matrix.  


#include<armadillo>
using namespace std;
using namespace arma;


vec startpcm(const imat & responses)
{
    int d, i, j, k, n, p, r;
    double x, y;
    n=responses.n_rows;
    p=responses.n_cols;
    r=p*(p-1)/2;
    irowvec nmax(p);
    nmax=max(responses,0);
    d=1+sum(nmax);
    field<ivec>counts(p);
    vec results(d), yy(r);
    
    k=0;
    for(j=0;j<p;j++)
    {
         counts(j).set_size(nmax(j)+1);
         counts(j).zeros();
         for(i=0;i<n;i++)counts(j)(responses(i,j))=counts(j)(responses(i,j))+1;
         if(min(counts(j))==0)
         {
              results.fill(datum::nan); 
              return results;
         }
         x=(double)counts(j)(0);
         for(i=0;i<nmax(j);i++)
         {
              y=(double)counts(j)(i+1);
              results(k+i)=log(y/x);
         }
         k=k+nmax(j);  
    }
    mat m(1,p),rsp(n,p),c(p,p);
    rsp=conv_to<mat>::from(responses);
    m=mean(rsp);
    c=cov(rsp,1);
    i=0;
    for(j=1;j<p;j++)
    {
         for(k=0;k<j;k++)
         {
              yy(i)=c(j,k)/(c(j,j)*c(k,k));
              i=i+1;
         }
    }
    results(d-1)=mean(yy);
    return results;
}
