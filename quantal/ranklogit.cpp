//Log likelihood component, gradient, and hessian matrix
//for model for ranks based on latent Gumbel distributions.
//There are r objects to be
//ranked and parameter
//vector beta has dimension r-1.  The analysis
//considers the k largest objects for a
//positive integer k no greater than r-1.
//The vector y provides the ranking.
//If order is 0, only the function is
//found, if order is 1, then the function and gradient are found.
//If order is 2,
//then the function, gradient, and Hessian are returned.
#include<armadillo>
using namespace arma;
struct f2v{double value; vec grad; mat hess;};
f2v ranklogit(const int & order, const vec & y, const vec & beta){
    double s;
    uword rr,z;
    vec f=exp(beta);
    f2v results;
    rr=beta.n_elem;
    results.value=0.0;
    if(order>0) results.grad=zeros(rr);
    if(order>1) results.hess=zeros(rr,rr);
    s=1.0+sum(f);
    for(auto yit=y.begin();yit!=y.end();++yit){
        z=*yit;
        results.value=results.value-log(s);
        if(order>0) results.grad=results.grad-f/s;
        if(order>1) results.hess=results.hess-diagmat(f)/s+f*trans(f)/(s*s);
        if(z>0){
            --z;
            results.value=results.value+beta(z);
            if(order>0) results.grad(z)+=1.0;
            s=s-f(z);
            f(z)=0.0;
        }
        else s=s-1.0;
    }
    return results;
}
