//Thissen et al. algorithm for sum of independent
//multinomials.  The tolerance is c>0.
//Here S is the sum of X(i) for i from 0 to n-1.
//X(i) is a multinomial trial with values 0 to cc(i)>0.  The vector p[i]
//consists of nonnegative real numbers with sum 1.  The probability that X(i) = j
//is element j of p[i] for j from 0 to cc(i).  The vector lwm that is returned has
//1+maxsum elements, where maxsum is the sum of the cc(i).
#include<armadillo>
using namespace arma;
vec lwm(double & c,int & n,vec p[])
{
    double d,sumd,xn;
    int bottom,bottom1,i,it,maxsum,top,top1;
    ivec cc(n);
    vec dist;
    for(i=0;i<n;i++)
    {
        cc(i)=p[i].n_elem;
    }
    maxsum=sum(cc)-n;
    bottom=0;
    top=cc(0)-1;
    dist.set_size(maxsum+1);
    dist=zeros(maxsum+1);
    dist.subvec(0,cc(0)-1)=p[0];
    for(it=1;it<n;it++)
    {
        xn=(double) it;
        d=xn*c/(2.0*xn+1.0);
        bottom1=bottom;
        top1=top+cc(it)-1;
        dist.subvec(bottom1,top1)=conv(dist.subvec(bottom,top),p[it]);
        sumd=0.0;
        for(i=bottom1;i<top;i++)
        {
            sumd=sumd+dist(i);
            if(i==bottom1&&sumd>d)
            {
                bottom=bottom1;
                break;
            }
            else
            {
                if(sumd>d)
                {
                    dist.subvec(bottom,i-1)=zeros(i-bottom);
                    bottom=i;
                    break;
                }
            }
        }
        sumd=0.0;
        for(i=top1;i>bottom;i--)
        {
            sumd=sumd+dist(i);
            if(i==top1&&sumd>d)
            {
                top=top1;
                break;
            }
            else
            {
                if(sumd>d)
                {
                    dist.subvec(i+1,top1)=zeros(top1-i);
                    top=i;
                    break;
                }
            }
        }
    }
    if(bottom>0||top<maxsum)
    {
        sumd=sum(dist.subvec(bottom,top));
        dist.subvec(bottom,top)=dist.subvec(bottom,top)/sumd;
    }
    return dist;
}
            
            
            
