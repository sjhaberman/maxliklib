//Log likelihood component and its gradient
//for continuous model with response y and parameter beta.
#include<armadillo>
using namespace arma;
extern char choice;
struct fd1bv
{
    double value;
    vec grad;
    
    bool fin;
};
fd1bv gumbel1(double,vec);
fd1bv logistic1(double,vec);

fd1bv normal1(double,vec);

fd1bv cont1(double y,vec beta)
{
    switch (choice)
    {
        case 'G': return gumbel1(y,beta);
        case 'L': return logistic1(y,beta);
        
        default: return normal1(y,beta);
    }
}
