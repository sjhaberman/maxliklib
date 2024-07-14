//Sort control table by keys.
#include<armadillo>
using namespace arma;
using namespace std;

vector<vector<string>> keysort(const field<string> & source)
{
    int i, i1, j, n, r;
    r=source.n_cols;
    n=source.n_rows;
    vector<vector<string>>result(n);
    for(i=0;i<n;i++)
    {
         result[i].resize(r);
         for(j=0;j<r;j++)result[i][j]=source(i,j);
    }
    sort(result.begin(),result.end());
    return result;
}