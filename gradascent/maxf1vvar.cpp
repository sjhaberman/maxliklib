//Combine argument y and corresponding function, gradient, and definition
//function in vary into a single entity maxf1vvar.
#include<armadillo>
using namespace arma;
struct f1v
{
    double value;
    vec grad;
};
struct maxf1v
{
    vec locmax;
    double max;
    vec grad;
};
maxf1v maxf1vvar(const vec &y,const f1v & fy)
{
    maxf1v result;
    result.locmax=y;
    result.max=fy.value;
    result.grad=fy.grad;
    return result;
}

