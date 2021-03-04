//Log likelihood component and its gradient and Hessian
//for continuous model with response y and parameter beta.
//transform determines the model with 'G' for Gumbel,
//'L' for logistic, and 'N' for normal.
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
f2v gumbel(const int & , const resp &, const vec &);
f2v logistic(const int & , const resp &, const vec &);
f2v normal(const int & , const resp & , const vec &);
f2v contresp(const int & order, const char & transform, const resp & y,
    const vec &  beta)
{
    switch (transform)
    {
        case 'G': return gumbel(order, y, beta);
        case 'L': return logistic(order, y, beta);
        default:  return normal(order, y, beta);
    }
}
