//Matrix for intercepts and slope.  Key is input dimension.
#include<armadillo>
using namespace arma;
mat ldscore(const int & k)
{
    mat result(k-1,k);
    result.cols(0,k-2)=eye(k-1,k-1);
    result.col(k-1)=ones(k-1);
    return result;
}
