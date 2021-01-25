//Log likelihood component and its gradient and Hessian
//for continuous model with response y and parameter beta.
//transform determines the model with 'G' for Gumbel,
//'L' for logistic, and 'N' for normal.
#include<armadillo>
using namespace arma;
struct f1v
{
    double value;
    vec grad;
};
f1v gumbel1(const vec &, const vec &);
f1v logistic1(const vec &, const vec &);
f1v normal1(const vec &, const vec &);
f1v normalv1(const vec &, const vec &);
f1v contresp1(const char & transform, const vec & y, const vec &  beta)
{
    switch (transform)
    {
        case 'G': return gumbel1(y,beta);
        case 'L': return logistic1(y,beta);

        default:
        {
             if(y.n_elem>1)
             {
                  return normalv1(y,beta);
             }
             else
             {
                  return normal1(y,beta);
             }
       }
    }
}
