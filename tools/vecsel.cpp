//Select vector elements and form new vector.
//xselect gives elements to select, and y is the original vector.
#include<armadillo>
using namespace arma;
struct xsel
{
    bool all;
    ivec list;
};
vec vecsel(const xsel & xselect, const vec & y)
{
    int d,i,j;
    vec result;
    if(xselect.all)
    {
        d=y.size();
    }
    else
    {
        d=xselect.list.size();
    }
    result.set_size(d);
    if(xselect.all)
    {
        result=y;
    }
    else
    {
        if(d>0)
        {
            for(i=0;i<d;i++)
            {
                j=xselect.list(i);
                if(j>=0&&j<d)
                {
                    result(i)=y(j);
                }
                else
                {
                    result.fill(datum::nan);
                    return result;
                }
            }
        }
    }
    return result;
}
