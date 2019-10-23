//Log likelihood
//for graded response model with integer response vector global_y and
//predictor array of matrices global_x.
//Weight vector global_w is used.

#include<armadillo>
using namespace arma;
struct fd0bv
{
    double value;
    
   
    bool fin;
};
extern char choices[];
extern char choice;


extern mat global_x [ ];
extern ivec global_y;
extern vec global_w;
extern vec global_offset [];

fd0bv graded0(int,vec);
fd0bv gradedlik0(vec beta)
{
    fd0bv results;
    fd0bv obsresults;
    int i;
    vec lambda;
    
    results.value=0.0;
   
    
    
    
    
    
    for (i=0;i<global_w.n_elem;i++)
    {
        lambda=global_offset[i]+global_x[i]*beta;
        choice=choices[i];
        obsresults=graded0(global_y(i),lambda);
        if(!obsresults.fin)
        {
            results.fin=false;
            return results;
            
        }
        results.value=results.value+global_w(i)*obsresults.value;
        
        
        
       
    }
    
    
    results.fin=true;
    return results;
}
