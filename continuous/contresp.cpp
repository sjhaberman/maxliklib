//Log likelihood component and its gradient and Hessian
//for continuous model with response y and parameter beta.
//transform determines the model with 'G' for Gumbel,
//'L' for logistic, and 'N' for normal.
#include<armadillo>
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
f2v gumbel(const vec &, const vec &);
f2v logistic(const vec &, const vec &);
f2v normal(const vec &, const vec &);
f2v normalv(const vec &, const vec &);
f2v contresp(const char & transform, const vec & y, const vec &  beta)
{
    switch (transform)
    {
        case 'G': return gumbel(y,beta);
        case 'L': return logistic(y,beta);

        default:
        {
             if(y.n_elem>1)
             {
                  return normalv(y,beta);
             }
             else
             {
                  return normal(y,beta);
             }
       }
    }
}
