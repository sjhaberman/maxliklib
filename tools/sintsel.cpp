//Count the number of selected integers from 0 to n-1. 
//xselect gives elements to select.
#include<armadillo>
using namespace arma;
struct xsel{bool all; uvec list;};
uword sintsel(const xsel & xselect, const uword & n){
    if(xselect.all) return n;
    return xselect.list.n_elem;
}
