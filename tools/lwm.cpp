//Thissen et al. algorithm for sum of independent
//multinomials.




#include<armadillo>

using namespace arma;
vec lwm(int n,vec p[])
{
    double small=1.0e-300;
    double p1,p2,p3;
    int bottom,i,j,it,maxsum,top;
    ivec c(n);
    vec dist;
   
    vec old;
    for(i=0;i<n;i++)
    {
        c(i)=p[i].n_elem;
        
    }
   
    maxsum=sum(c)-n;
    bottom=0;
    top=0;
    dist.set_size(maxsum+1);
    dist=zeros(maxsum+1);
    dist(0)=1.0;
    
    for(it=0;it<n;it++)
    {
        while(dist(top)==0.0)
        {
            top=top-1;
        }
        top=top+c(it)-1;

            
        old.set_size(c(it));
        
        for(i=bottom;i<=top;i++)
        {
            
            if(i==bottom)
            {
                old=zeros(c(it));
                for(j=0;j<c(it);j++)
                {
                    if(i-j>=bottom)
                    {
                        old(j)=dist(i-j);
                    }
                }
            }
            else
            {
                old=shift(old,+1);
                old(0)=dist(i);
                
            }
            p3=dot(p[it],old);
            
            if(p3<small)p3=0.0;
            dist(i)=p3;
            
        }
        while(dist(bottom)==0.0)
        {
            bottom=bottom+1;
        }
    }
    return dist;
}
            
            
            
