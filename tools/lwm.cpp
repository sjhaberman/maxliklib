//Thissen et al. algorithm for sum of independent
//multinomials.  The tolerance is c>0.
//Here S is the sum of X(i) for i from 0 to n-1.
//X(i) is a multinomial trial with values 0 to cc(i)-1>0.  The vector p(i)
//consists of nonnegative real numbers with sum 1.  The probability that X(i) = j
//is element j of p(i) for j from 0 to cc(i)-1.
//The vector lwm that is returned has
//1+maxsum elements, where maxsum is the sum of the (cc(i)-1) for 0<=i<n.
#include<armadillo>
using namespace std;
using namespace arma;
vec lwm(const double & c, const vector<vec> & p ){
    double d,sumd,xn;
    uword bottom=0,bottom1,i,it,maxsum,n,top,top1;
    n=p.size();
    vector<uword>cc(n);
    for(it=0;it<n;it++) cc[it]=p[it].n_elem-1;
    maxsum=accumulate(cc.begin(),cc.end(),bottom);
//bottom is lower bound for nonzero entries of S(k), the sum of X(j) for j from 0
//to k<n.  top is upper bound for nonzero entries of S(k)
    top=cc[0];
    vec dist(maxsum+1);
//S(0) has distribution of X(0).
    dist.subvec(0,cc[0])=p[0];
    if(n==1)return dist;
    xn=0.0;
//Cycle through X(it) for it from 1 to n-1.
    for(it=1;it<n;it++){
        xn+=1.0;
//Bound for when S(it)<=i or S(it)>=i has negligible probability.
        d=xn*c/(2.0*xn+1.0);
//Tentative new values of bottom and top.
        top1=top+cc[it];
//Convolution of distribution of S(it-1) and X(it).
        dist.subvec(bottom,top1)=conv(dist.subvec(bottom,top),p[it]);
//Negligibility check.
        sumd=0.0;
//Update bottom.
        for(i=bottom;i<top1;i++){
            sumd=sumd+dist(i);
            if(i==bottom&&sumd>d) break;
            else{
//Insert 0 when needed.
                if(sumd>d){
                    dist.subvec(bottom,i-1)=zeros(i-bottom);
                    bottom=i;
                    break;
                }
            }
        }
//Update top.
        sumd=0.0;
        for(i=top1;i>bottom;i--){
            sumd=sumd+dist(i);
            if(i==top1&&sumd>d){
                top=top1;
                break;
            }
            else{
//Insert 0 when needed.
                if(sumd>d){
                    dist.subvec(i+1,top1)=zeros(top1-i);
                    top=i;
                    break;
                }
            }
        }
    }
//Adjust if needed for insertion of 0.
    if(bottom>0||top<maxsum){
        sumd=sum(dist.subvec(bottom,top));
        dist.subvec(bottom,top)=dist.subvec(bottom,top)/sumd;
    }
    return dist;
}

