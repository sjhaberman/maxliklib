//Log likelihood component
//for multinomial quantal model with response y from 0 to r and parameter
//vector beta of dimension r-1. The variable choice governs the model selection.
//'L' yields the cumulative logit model, 'N' yields the cumulative probit model, 'C' yields
//the cumulative log-log model, and 'M' yields the multinomial response model.
#include<armadillo>
using namespace arma;

extern char choice;



double multlogit0(int,vec);
double cumloglog0(int,vec);
double cumlogit0(int,vec);
double cumprobit0(int,vec);
double multquantal0(int y,vec beta)
{
    switch(choice)
    {
        case 'L':return cumlogit0(y,beta);
        case 'N':return cumprobit0(y,beta);
        case 'C': return cumloglog0(y,beta);
        default: return multlogit0(y,beta);
        
    }
}
