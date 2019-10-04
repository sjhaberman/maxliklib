//Log likelihood component and its gradient
//for gumbel model with response y and parameter vector beta
//with elements beta(1) and beta(2)>0.

#include<armadillo>
using namespace arma;
using namespace std;
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
fd1bv locationscale1(double,vec,function <fd1(double)>);
fd1 gumbel01(double);
fd1bv gumbel1(double y,vec beta)
{
    
    fd1bv results;
    results=locationscale1(y,beta,gumbel01);
    return results;
}
