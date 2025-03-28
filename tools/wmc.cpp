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
    int j,m,n;
//m observations of dimension n.
    m=wx.v.n_elem;
    n=wx.m.n_cols;
    vecmat result;
//Column vector of weighted means;
    result.v.set_size(n);  
    result.v=trans(wx.m)*wx.v;
    if(order>1)
    {
        result.m.set_size(n,n);
//t is matrix of residuals.
//u is matrix of residuals by weights.
        mat t(m,n),u(m,n);
        t=wx.m.each_row()-trans(result.v);
        u=wx.v%t.each_col();
//Weighted covariance matrix.
        result.m=trans(u)*t;
    }
    return result;
}

