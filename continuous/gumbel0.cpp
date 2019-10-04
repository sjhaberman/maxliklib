//Log likelihood component
//for gumbel model with response y and parameter vector beta
//with elements beta(1) and beta(2)>0.

#include<armadillo>
using namespace arma;
using namespace std;
struct fd0bv
{
    double value;
    
    
    bool fin;
};

fd0bv locationscale0(double,vec,function <double(double)>);
double gumbel00(double);
fd0bv gumbel0(double y,vec beta)
{
    
    fd0bv results;
    results=locationscale0(y,beta,gumbel00);
    return results;
}
