//Log likelihood and its gradient and Hessian
//for quantal response model with response vector y and
//predictor matrix x.
//Weight w is used.
#include<armadillo>
using namespace arma;
struct fd2
{
    double value;
    double der1;
    double der2;
};
struct fd2v
{
    double value;
    vec grad;
    mat hess;
};



extern vec global_w;
extern ivec global_y;
extern mat global_x;

fd2 quantal(int,double);
fd2v quantallik(vec beta)
{
    double lambda;
    
    fd2 obsresults;
    fd2v results;
    int i;
    results.value=0.0;
    results.grad=zeros(beta.n_elem);
    
    results.hess=zeros(beta.n_elem,beta.n_elem);
    for (i=0;i<global_x.n_rows;i++)
    {
        lambda=dot(beta,global_x.row(i));
        obsresults=quantal(global_y(i),lambda);
        results.value=results.value+global_w(i)*obsresults.value;
        results.grad=results.grad+global_w(i)*obsresults.der1*trans(global_x.row(i));
        results.hess=results.hess+
            global_w(i)*obsresults.der2*trans(global_x.row(i))*global_x.row(i);
        
    }
    
    return results;
}
