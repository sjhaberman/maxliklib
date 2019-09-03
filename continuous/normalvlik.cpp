//Log likelihood and its gradient and Hessian
//for multivariate normal linear model with response vectors global_y and
//predictor array of matrices global_x.
//Weight vector global_w is used.

#include<armadillo>
using namespace arma;
struct fd2bv
{
    double value;
    vec grad;
    mat hess;
    bool fin;
};


extern mat global_x [ ];
extern vec global_y [ ];
extern vec global_w;
extern vec global_offset [ ];

fd2bv normalv(vec,vec);
fd2bv normalvlik(vec beta)
{
    fd2bv results;
    fd2bv obsresults;
    int i;
    vec lambda;
    
    results.value=0.0;
    results.grad=zeros(beta.n_elem);
    results.hess=zeros(beta.n_elem,beta.n_elem);
    results.fin=true;
    
    
    
    
    for (i=0;i<global_w.n_elem;i++)
    {
        lambda=global_offset[i]+global_x[i]*beta;
        
        obsresults=normalv(global_y[i],lambda);
        if(!obsresults.fin)
        {
            results.fin=false;
            return results;
        }
        results.value=results.value+global_w(i)*obsresults.value;
        results.grad=results.grad+global_w(i)*trans(global_x[i])*obsresults.grad;
        results.hess=results.hess+global_w(i)*trans(global_x[i])*obsresults.hess*global_x[i];
        
       
    }
    
    
    return results;
}
