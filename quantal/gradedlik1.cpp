//Log likelihood and its gradient
//for graded response model with integer response vector global_y and
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
extern char choices[];
extern char choice;

extern mat global_x [ ];
extern ivec global_y;
extern vec global_w;
extern vec global_offset [];

fd1bv graded1(int,vec);
fd1bv gradedlik1(vec beta)
{
    fd1bv results;
    fd1bv obsresults;
    int i;
    vec lambda;
    
    results.value=0.0;
    results.grad=zeros(beta.n_elem);
    
    
    
    
    
    for (i=0;i<global_w.n_elem;i++)
    {
        lambda=global_offset[i]+global_x[i]*beta;
        choice=choices[i];
        obsresults=graded1(global_y(i),lambda);
        if(!obsresults.fin)
        {
            results.fin=false;
            return results;
            
        }
        results.value=results.value+global_w(i)*obsresults.value;
        results.grad=results.grad+global_w(i)*trans(global_x[i])*obsresults.grad;
        
        
       
    }
    
    
    results.fin=true;
    return results;
}
