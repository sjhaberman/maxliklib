//Log likelihood component, gradient, and hessian matrix
//for multinomial quantal model with response y from 0 to r and parameter
//vector beta of dimension r. The variable choice governs the model selection.
//'L' yields the cumulative logit model, 'N' yields the cumulative probit model, 'C' yields
//the cumulative log-log model, and 'M' yields the multinomial response model.
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
fd2v cumloglog(int,vec);
fd2v cumlogit(int,vec);
fd2v cumprobit(int,vec);
fd2v multquantal(int y,vec beta)
{
    switch(choice)
    {
        case 'L':return cumlogit(y,beta);
        case 'N':return cumprobit(y,beta);
        case 'C': return cumloglog(y,beta);
        default: return multlogit(y,beta);
        
    }
}
