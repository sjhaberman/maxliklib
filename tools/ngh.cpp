//Numerical approximation of gradient and Hessian.  Original function is f, evaluation is at x, and step size is delta.
#include<armadillo>
using namespace std;
using namespace arma;
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
f2v ngh(const int & order, const double & delta, const vec & x,
     const function <f2v(const int & , const vec & )>f)
{
    int order0=0,d,i,j,k;
    double delta2,deltasq;
    f2v result,fx,fu,fv,fyu,fzu,fyv, fzv;
    fx=f(order0,x);
    result.value=fx.value;
    if(order<1||isnan(result.value))return result;
    d=x.n_elem;
    result.grad.set_size(d);
    if(order>1)result.hess.set_size(d,d);
    vec u,v,yu,zu,yv,zv;
    u.set_size(d);
    v.set_size(d);
    if(order>1)
    {
         yu.set_size(d);
         zu.set_size(d);
         yv.set_size(d);
         zv.set_size(d);
    }
    delta2=delta+delta;
    if(order>1)deltasq=delta*delta;
    for(i=0;i<d;i++)
    {
        u=x;
        v=x;
        u(i)=u(i)+delta;
        v(i)=v(i)-delta;
        fu=f(order0,u);
        fv=f(order0,v);
        if(isnan(fu.value)||isnan(fv.value))
        {
            result.value=datum::nan;
            result.grad.fill(datum::nan);
            return result;
        }
        result.grad(i)=(fu.value-fv.value)/delta2;
        if(order>1)
        {
            result.hess(i,i)=(fu.value-2.0*fx.value+fv.value)/deltasq;
            if(d>1)
            {
                for(j=0;j<i;j++)
                {
                    yu=u;
                    zu=u;
                    yv=v;
                    zv=v;
                    yu(j)=yu(j)+delta;
                    zu(j)=zu(j)-delta;
                    yv(j)=yv(j)+delta;
                    zv(j)=zv(j)-delta;
                    fyu=f(order0,yu);
                    fzu=f(order0,zu);
                    fyv=f(order0,yv);
                    fzv=f(order0,zv);
                    if(isnan(fyu.value)||isnan(fzu.value)||isnan(fyv.value)||isnan(fzv.value))
                    {
                         result.value=datum::nan;
                         result.grad.fill(datum::nan);
                         result.hess.fill(datum::nan);
                         return result;
                    }
                    result.hess(i,j)=(fyu.value-fzu.value-fyv.value+fzv.value)/(4.0*deltasq);
                    result.hess(j,i)=result.hess(i,j);
               }
          }
       }
       
    }
    return result;
}
