//Log likelihood and its gradient
//for multivariate normal linear model with response vectors global_y and
//predictor array of matrices global_x.
//Weight vector global_w is used.

#include<armadillo>
using namespace arma;
struct fd1bv
{
    double value;
    vec grad;
    
    bool fin;
};


extern mat global_x [ ];
extern vec global_y [ ];
extern vec global_w;
extern vec global_offset [ ];

fd1bv normalv1(vec,vec);
fd1bv normalvlik1(vec beta)
{
    fd1bv results;
    fd1bv obsresults;
    int i;
    vec lambda;
    
    results.value=0.0;
    results.grad=zeros(beta.n_elem);
    
    results.fin=true;
    
    
    
    
    for (i=0;i<global_w.n_elem;i++)
    {
        lambda=global_offset[i]+global_x[i]*beta;
        
        obsresults=normalv1(global_y[i],lambda);
        if(!obsresults.fin)
        {
            results.fin=false;
            return results;
        }
        results.value=results.value+global_w(i)*obsresults.value;
        results.grad=results.grad+global_w(i)*trans(global_x[i])*obsresults.grad;
        
        
       
    }
    
    
    return results;
}
