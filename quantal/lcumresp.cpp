//Log likelihood component, gradient, and Hessian
//for cumulative model with response y and common slope parameter.
//Choice of tranformation is determined by transform, with 'G' for
//complementary log-log, 'H' for log-log,
//'L' for logit, and 'N' for probit.  If order is 0, only the function is
//found, if order is 1, then the function and gradient are found.  If order is 2,
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
mat ldscore(const int & );
f2v cumresp(const int & , const char & , const resp & , const vec & );
f2v lcumresp(const int & order, const char & transform, const resp & y,
            const vec & beta)
{
    int r;
    f2v results, result;
    r=beta.n_elem;
    vec gamma(r-1);
    mat ls(r-1,r);
    ls=ldscore(r);
    gamma=ls*beta;
    if(order>0){
        result.grad.set_size(r-1);
        results.grad.set_size(r);
    }
    if(order>1){
        result.hess.set_size(r-1,r-1);
        results.hess.set_size(r,r);
    }
    result=cumresp(order,transform, y, gamma);
    results.value=result.value;
    if(order>0)results.grad=ls.t()*result.grad;
    if(order>1)results.hess=ls.t()*result.hess*ls;
    return results;
}
