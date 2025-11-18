//Log likelihood component, gradient, and Hessian
//for graded response model with response y and linear slope parameter.p
//If order is 0, only the function is
//found, if order is 1, then the function and gradient are found.
//If order is 2,
//then the function, gradient, and Hessian are returned.
//transform is defined as in genresp.cpp.
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
f2v linsel(const int & , const f2v & , const mat & );
f2v nanf2v(const int & , const f2v & );
f2v gradresp(const int & , const char & t, const resp & , const vec & );
f2v lgradresp(const int & order, const char & transform ,
             const resp & y, const vec & beta)
{
    f2v results, result;
    int r;
    r=beta.n_elem;
    vec gamma(r-1);
    mat ls(r-1,r);
    ls=ldscore(r);
    gamma=ls*beta;
    if(order>0)
    {
        results.grad.set_size(r);
        result.grad.set_size(r-1);
    }
    if(order>1)
    {
        results.hess.set_size(r,r);
        result.hess.set_size(r-1,r-1);
    }
//Check for unacceptable gamma.
    if(r>2)
    {
        if(max(diff(gamma))>=0.0){
            results=nanf2v(order,results);
            return results;
        }
    }
    result=gradresp(order,transform,y,gamma);
    results=linsel(order,result,ls);
    return results;
}
