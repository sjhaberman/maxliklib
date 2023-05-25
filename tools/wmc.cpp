//Weighted mean and covariance matrix.
//Data matrix is wx.m and data weight is wx.v.
#include<armadillo>
using namespace arma;
// Combination of vector and matrix.
struct vecmat
{
    vec v;
    mat m;
};
vecmat wmc(const vecmat & wx)
{
    int i,j,k,m,n;
    m=wx.v.n_elem;
    n=wx.m.n_rows;
    vecmat result;
    result.v.set_size(n);
    result.m.set_size(n,n);
    result.v=wx.m*wx.v;
    for(i=0;i<n;i++)
    {
        for(j=0;j<=i;j++)
        {
            result.m(i,j)=0.0;
            for (k=0;k<m;k++)result.m(i,j)+=(wx.m(i,k)-result.v(i))*(wx.m(j,k)-result.v(j))*wx.v(k);
            if(j<i) result.m(j,i)=result.m(i,j);
        }
    }
    return result;
}

