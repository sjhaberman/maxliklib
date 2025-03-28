//Add f2v components together when selection of variables may be involved.
//xselect gives elements to select, x is the first
//f2v struct, and y is the other f2v struct.  a is a multiplier, and order
//indicates what f2v components are added.
#include<armadillo>
using namespace arma;
struct xsel
{
    bool all;
    uvec list;
};
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
void addsel(const int & order, const xsel & xselect, const f2v & x, f2v & y, const double & a)
{
    y.value=y.value+a*x.value;
    if(order<1) return;  
    if(xselect.all)
    {
        y.grad=y.grad+a*x.grad;
        if(order>1)
        {
            y.hess=y.hess+a*x.hess;
        }
    }
    else
    {
        if(xselect.list.n_elem==0)return;
        y.grad.elem(xselect.list)=y.grad.elem(xselect.list)+a*x.grad;
        if(order>1)
        {
            y.hess.submat(xselect.list,xselect.list)
                =y.hess.submat(xselect.list,xselect.list)+a*x.hess;
        }
    }
    return;
}
