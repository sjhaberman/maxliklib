//Log likelihood
//for multivariate normal linear model with response vectors global_y and
//predictor array of matrices global_x.
//Weight vector global_w is used.

#include<armadillo>
using namespace arma;
struct fd0bv
{
    double value;
   
    
    bool fin;
};


extern mat global_x [ ];
extern vec global_y [ ];
extern vec global_w;
extern vec global_offset [ ];

fd0bv normalv0(vec,vec);
fd0bv normalvlik0(vec beta)
{
    fd0bv results;
    fd0bv obsresults;
    int i;
    vec lambda;
    
    results.value=0.0;
    
    
    results.fin=true;
    
    
    
    
    for (i=0;i<global_w.n_elem;i++)
    {
        lambda=global_offset[i]+global_x[i]*beta;
        
        obsresults=normalv0(global_y[i],lambda);
        if(!obsresults.fin)
        {
            results.fin=false;
            return results;
        }
        results.value=results.value+global_w(i)*obsresults.value;
        
        
       
    }
    
    
    return results;
}
