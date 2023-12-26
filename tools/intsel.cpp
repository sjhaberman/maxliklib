//Select member i from 
//selected integers from 0 to n-1. 
//xselect gives elements to select.
#include<armadillo>
using namespace arma;
struct xsel
{
    bool all;
    uvec list;
};
int intsel(const xsel & xselect, const int & i)
{
    int d;
    if(xselect.all)
    {
         d = i;
    }
    else
    {
         d = xselect.list(i);
    }
    return d;
}
