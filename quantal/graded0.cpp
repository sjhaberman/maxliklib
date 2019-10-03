//Log likelihood component
//for graded model with response y and parameter beta.
#include<armadillo>
using namespace arma;
using namespace std;
extern char choice;
struct fd0bv
{
    double value;
    
    
    bool fin;
};

double gumbelf0(double);
double logisticf0(double);
double normalf0(double);

fd0bv gradf0(int,vec,function <double(double)>);
fd0bv graded0(int y,vec beta)
{
    switch (choice)
    {
        case 'G': return gradf0(y,beta,gumbelf0);
        case 'L': return gradf0(y,beta,logisticf0);
        default: return gradf0(y,beta,normalf0);
    }
    
}
