//Log likelihood component and its gradient and Hessian
//for continuous model with response y and parameter beta.
#include<armadillo>
using namespace arma;
extern char choice;
struct fd2bv
{
    double value;
    vec grad;
    mat hess;
    bool fin;
};

fd2bv logistic(double,vec);

fd2bv normal(double,vec);

fd2bv cont(double y,vec beta)
{
    switch (choice)
    {
        case 'L': return logistic(y,beta);
        
        default: return normal(y,beta);
    }
}
