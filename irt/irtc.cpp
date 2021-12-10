//Find log likelihood component given latent response theta
//and corresponding gradient(if order>0) and Hessian (if order>1) for a
//generalized IRT model.  The component involves
//the response variables specified in irtdata,
//the correspondence specified by thetamaps
//of the latent response and the predictors and of the latent response
//and the latent distribution.  The model parameter is beta.
#include<armadillo>
using namespace arma;
//Except for thetamap, the structs are also used in genresplik.cpp.
struct f2v
{
    double value;
    vec grad;
    mat hess;
};
struct model
{
    char type;
    char transform;
};
struct resp
{
    ivec iresp;
    vec dresp;
};
struct xsel
{
    bool all;
    uvec list;
};
struct dat
{
    model choice;
    double weight;
    resp dep;
    vec offset;
    mat indep;
    xsel xselect;
};
//dep indicates if the dependent response is obtained
//from theta. drespcols gives the members of
//theta.dresp used for the response, and
//irespcols gives the members of
//theta.dresp used for the response.
//respno is response number.  If dep is false,
//offsets are for effect of theta.dresp on model parameter
//without consideration of beta and
//indeps is the cube for interaction of beta and theta.dresp.
struct thetamap
{
    bool dep;
    int respno;
    xsel drespcols;
    xsel irespcols;
    mat offsets;
    cube indeps;
};
ivec ivecsel(const xsel & xselect, const ivec & y);
vec vecsel(const xsel & , const vec & y);
f2v genresplik(const int & , const std::vector<dat> & , const vec & );
f2v irtc (const int & order, const std::vector<dat> & data,
    const std::vector<thetamap> & thetamaps, const resp & theta,
    const vec &  beta)
{
    f2v results;
    vec thetasel;
    int g,gg,h,i,j,k,m,n,p,q,r,s,t;
    n=data.size();
    m=beta.n_elem;
    k=thetamaps.size();
    if(m>0)
    {
        if(order>0) results.grad.set_size(m);
        if(order>1) results.hess.set_size(m,m);
    }
    std::vector<dat> irtdata(n);
    for(i=0;i<n;i++)
    {
        irtdata[i].weight=data[i].weight;
        irtdata[i].choice.type=data[i].choice.type;
        irtdata[i].choice.transform=data[i].choice.transform;
        p=data[i].offset.n_elem;
        irtdata[i].offset.set_size(p);
        irtdata[i].offset=data[i].offset;
        q=data[i].dep.iresp.size();
        r=data[i].dep.dresp.size();
        irtdata[i].dep.iresp.set_size(q);
        irtdata[i].dep.dresp.set_size(r);
        irtdata[i].dep.iresp=data[i].dep.iresp;
        irtdata[i].dep.dresp=data[i].dep.dresp;
        s=data[i].indep.n_cols;
        irtdata[i].indep.set_size(p,s);
        irtdata[i].indep=data[i].indep;
        irtdata[i].xselect.all=data[i].xselect.all;
        t=data[i].xselect.list.n_elem;
        irtdata[i].xselect.list.set_size(t);
        irtdata[i].xselect.list=data[i].xselect.list;
    }
    if(k>0)
    {
        for(j=0;j<k;j++)
        {
            g=thetamaps[j].respno;
            if(thetamaps[j].drespcols.all)
            {
                h=theta.dresp.size();
            }
            else
            {
                h=thetamaps[j].drespcols.list.size();
            }
            if(h>0)
            {
                thetasel.set_size(h);
                thetasel=vecsel(thetamaps[j].drespcols,theta.dresp);
                if(thetamaps[j].dep)
                {
                    irtdata[g].dep.dresp=thetasel;
                }
                else
                {
                    irtdata[g].offset=irtdata[g].offset+thetamaps[j].offsets*thetasel;
                    for(gg=0;gg<h;gg++)irtdata[g].indep=irtdata[g].indep+thetamaps[j].indeps.slice(gg)*thetasel(gg);
                }
            }
            if(thetamaps[j].irespcols.all)
            {
                h=theta.iresp.size();
            }
            else
            {
                h=thetamaps[j].irespcols.list.size();
            }
            if(h>0&&thetamaps[j].dep)
            {
                irtdata[g].dep.iresp=ivecsel(thetamaps[j].irespcols,theta.iresp);
                
            }
        }
    }
    results=genresplik(order,irtdata,beta);
    return results;
}
