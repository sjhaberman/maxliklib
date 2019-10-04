//Log likelihood component and gradient
//for multinomial quantal model with response y from 0 to r and parameter
//vector beta of dimension r-1. The variable choice governs the model selection.
//'L' yields the cumulative logit model, 'N' yields the cumulative probit model, 'C' yields
//the cumulative log-log model, and 'M' yields the multinomial response model.
#include<armadillo>
using namespace arma;
struct fd1v
{
    double value;
    vec grad;
    
};
extern char choice;



fd1v multlogit1(int,vec);
fd1v cumloglog1(int,vec);
fd1v cumlogit1(int,vec);
fd1v cumprobit1(int,vec);
fd1v multquantal1(int y,vec beta)
{
    switch(choice)
    {
        case 'L':return cumlogit1(y,beta);
        case 'N':return cumprobit1(y,beta);
        case 'C': return cumloglog1(y,beta);
        default: return multlogit1(y,beta);
        
    }
}
