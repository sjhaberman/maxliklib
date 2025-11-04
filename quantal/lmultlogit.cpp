//Log likelihood component, gradient, and hessian matrix
//for multinomial logit model with response vector y with integer values
//from 0 to r-2 and linear score.  The vector parameter
//vector beta has dimension r.  The first r-1 elements are intercept and the
//last is the slope.
//If order is 0, only the function is
//found, if order is 1, then the function and gradient are found.
//If order is 2,
//then the function, gradient, and Hessian are returned.
#include<armadillo>
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
struct resp
{
    ivec iresp;
    vec dresp;
};
f2v multlogit(const int & , const resp & , const vec & );
mat lscore(const int & );
f2v lmultlogit(const int & order, const resp & y, const vec & beta)
{
    int r;
    r=beta.n_elem;
    vec gamma(r-1);
    mat ls(r-1,r);
    ls=lscore(r);
    gamma=ls*beta;
    f2v result,results;
    if(order>0){
        result.grad.set_size(r-1);
        results.grad.set_size(r);
    }
    if(order>1){
        result.hess.set_size(r-1,r-1);
        results.hess.set_size(r,r);
    }
    result=multlogit(order,y,gamma);
    results.value=result.value;
    if(order>0)results.grad=ls.t()*result.grad;
    if(order>1)results.hess=ls.t()*result.hess*ls;
    return results;
}
