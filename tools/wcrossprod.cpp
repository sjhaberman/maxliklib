//Weighted matrix of cross-product.
//Matrix is wx.m and weight is wx.v.
#include<armadillo>
using namespace arma;
// Combination of vector and matrix.
struct vecmat
{
    vec v;
    mat m;
};
mat wcrossprod(const vecmat & wx)
{
    int i,j,k,m,n;
    m=wx.v.n_elem;
    n=wx.m.n_rows;
    mat result(n,n);
    
    for(i=0;i<n;i++)
    {
        for(j=0;j<=i;j++)
        {
            result(i,j)=0.0;
            for (k=0;k<m;k++)result(i,j)+=wx.m(i,k)*wx.m(j,k)*wx.v(k);
            if(j<i) result(j,i)=result(i,j);
        }
    }
    return result;
}

