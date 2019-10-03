//Log likelihood component and gradient
//for graded model with response y and parameter beta.
#include<armadillo>
using namespace arma;
using namespace std;
extern char choice;
struct fd1bv
{
    double value;
    vec grad;
    
    bool fin;
};
struct fd1
{
    double value;
    double der1;
    
};
fd1 gumbelf01(double);
fd1 logisticf01(double);
fd1 normalf01(double);

fd1bv gradf1(int,vec,function <fd1(double)>);
fd1bv graded1(int y,vec beta)
{
    switch (choice)
    {
        case 'G': return gradf1(y,beta,gumbelf01);
        case 'L': return gradf1(y,beta,logisticf01);
        default: return gradf1(y,beta,normalf01);
    }
    
}
