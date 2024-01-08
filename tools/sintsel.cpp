//Count the number of selected integers from 0 to n-1. 
//xselect gives elements to select.
#include<armadillo>
using namespace arma;
struct xsel
{
    bool all;
    uvec list;
};
int sintsel(const xsel & xselect, const int & n)
{
    int d;
    if(xselect.all)
    {
         d = n;
    }
    else
    {
         d = xselect.list.n_elem;
    }
    return d;
}
