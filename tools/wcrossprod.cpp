//Weighted matrix of cross-product.
//Matrix is x and weight is w.
#include<armadillo>
using namespace arma;
mat wcrossprod(const mat & x,const vec & w)
{
    int i,j;
    mat result(x.n_cols,x.n_cols);
    for(i=0;i<x.n_cols;i++)
    {
        for(j=0;j<=i;j++)
        {
            result(i,j)=sum(x.col(i)%x.col(j)%w);
            if(j<i) result(j,i)=result(i,j);
        }
    }
    return result;
}

