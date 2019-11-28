//Lord-Wingersky algorithm for compound binomial distribution.




#include<armadillo>

using namespace arma;
vec lw(vec p)
{
    double small=1.0e-300;
    double p1,p2,p3;
    int bottom,i,it,n,top;
   
    
   
    vec dist(p.n_elem+1);
    dist.zeros();
    dist(0)=1.0;
    n=p.n_elem;
    bottom=0;
    top=0;
    for(it=0;it<n;it++)
    {
        if(dist(top)>0.0)
        {
            top=it+1;
        }
        for(i=bottom;i<=top;i++)
        {
            if(i>0)
            {
                p1=p2;
            }
            else
            {
                p1=0.0;
            }
            if(i<top)
            {
                p2=dist(i);
            }
            else
            {
                p2=0.0;
            }
            
            p3=p2*(1-p(it))+p1*p(it);
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
            
            
            
