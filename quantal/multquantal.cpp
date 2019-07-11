//Log likelihood component, gradient, and hessian matrix
//for multinomial quantal model with response y from 0 to r and parameter
//vector beta of dimension r.  If choice is 'L', a multinomial logit model is used.
#include<armadillo>
using namespace arma;
struct fd2v
{
    double value;
    vec grad;
    mat hess;
};
extern char choice;



fd2v multlogit(int,vec);
fd2v multquantal(int y,vec beta)
{
    switch (choice)
    {
        case 'L': return multlogit(y,beta);
        default: return multlogit(y,beta);
    }
}
