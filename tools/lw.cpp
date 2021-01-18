//Lord-Wingersky algorithm for compound binomial distribution.
//c>0.0 is the tolerance.
//p is the vector of probabilities of length n between 0 and 1.
//The distribution of the sum S of X(i), i from 0 to n-1, is sought, where
//the X(i) are mutuall independent Bernoulli random variables such that X(i)
//is 1 with probability p(i).  The returned vector lw has elements from 0 to n,
//and lw(i) is the probability that S=i for i from 0 to n.
#include<armadillo>
using namespace arma;
vec lw(const double & c, const vec & p)
{
    double d,sumd,xn;
    int bottom,bottom1,i,it,n,top,top1;
    vec dist(p.n_elem+1),pp(2);
//Distribution of S(0)=X(0).
    dist(0)=1.0-p(0);
    dist(1)=p(0);
//Find n.
    n=p.n_elem;
//bottom is lower bound for nonzero entries of S(k), the sum of X(j) for j from 0
//to k<n.  top is upper bound for nonzero entries of S(k)
    bottom=0;
    top=1;
//Cycle through X(it) for it from 1 to n-1.
    for(it=1;it<n;it++)
    {
        xn=(double) it;
//Bound for when S(it)<=i or S(it)>=i has negligible probability.
        d=xn*c/(2.0*xn+1.0);
//Tentative new values of bottom and top.
        bottom1=bottom;
        top1=top+1;
//Distribution of X(it).
        pp(0)=1.0-p(it);
        pp(1)=p(it);
//Convolution of distribution of S(it-1) and X(it).
        dist.subvec(bottom1,top1)=conv(dist.subvec(bottom,top),pp);
//Negligibility check.
        sumd=0.0;
        for(i=bottom1;i<top;i++)
        {
            sumd=sumd+dist(i);
//Update bottom.
            if(i==bottom1&&sumd>d)
            {
                bottom=bottom1;
                break;
            }
            else
            {
//Insert 0 when needed.
                if(sumd>d)
                {
                    dist.subvec(bottom,i-1)=zeros(i-bottom);
                    bottom=i;
                    break;
                }
            }
        }
        sumd=0.0;
//Update top.
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
//Insert 0 when needed.
                if(sumd>d)
                {
                    dist.subvec(i+1,top1)=zeros(top1-i);
                    top=i;
                    break;
                }
            }
        }
    }
//Adjust if needed for insertion of 0.
    if(bottom>0||top<n)
    {
        sumd=sum(dist.subvec(bottom,top));
        dist.subvec(bottom,top)=dist.subvec(bottom,top)/sumd;
    }
    return dist;
}
