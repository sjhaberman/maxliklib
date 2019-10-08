//Log likelihood component
//for continuous model with response y and parameter beta.
#include<armadillo>
using namespace arma;
extern char choice;
struct fd0bv
{
    double value;
    
    
    bool fin;
};
fd0bv gumbel0(double,vec);
fd0bv logistic0(double,vec);

fd0bv normal0(double,vec);

fd0bv cont0(double y,vec beta)
{
    switch (choice)
    {
        case 'G': return gumbel0(y,beta);
        case 'L': return logistic0(y,beta);
        
        default: return normal0(y,beta);
    }
}
