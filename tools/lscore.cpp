//Matrix for intercepts and slope.  k is input dimension.
#include<armadillo>
using namespace arma;
mat lscore(const int & k)
{
    mat result(k-1,k);
    result.cols(0,k-2)=eye(k-1,k-1);
    result.col(k-1)=regspace(1,k-1);
    return result;
}
