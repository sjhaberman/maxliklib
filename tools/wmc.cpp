//Weighted mean (order=1)and covariance matrix (order>1).
//Data matrix is wx.m and data weight is wx.v.
#include<armadillo>
using namespace arma;
// Combination of vector and matrix.
struct vecmat
{
    vec v;
    mat m;
};
vecmat wmc(const int & order, const vecmat & wx)
{
    int i,j,k,m,n;
//m observations of dimension n.
    m=wx.v.n_elem;
    n=wx.m.n_cols;
    vecmat result;
    result.v.set_size(n);   
    result.v=trans(wx.m)*wx.v;
    if(order>1)
    {
      result.m.set_size(n,n);
      for(i=0;i<n;i++)
      {
        for(j=0;j<=i;j++)
        {
            result.m(i,j)=0.0;
            for (k=0;k<m;k++)result.m(i,j)+=(wx.m(k,i)-result.v(i))*(wx.m(k,j)-result.v(j))*wx.v(k);
            if(j<i) result.m(j,i)=result.m(i,j);
        }
     }
    }
    return result;
}

