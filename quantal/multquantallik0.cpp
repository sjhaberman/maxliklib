//Log likelihood
//for multinomial quantal model with integer response vector global_y and
//predictor array of matrices global_x.
//Weight vector global_w is used.

#include<armadillo>
using namespace arma;

extern char choices[];
extern char choice;

extern mat global_x [ ];
extern ivec global_y;
extern vec global_w;
extern vec global_offset [];

double multquantal0(int,vec);
double multquantallik0(vec beta)
{
    double results;
    double obsresults;
    int i;
    vec lambda;
    
    results=0.0;
    

    
    
    
    
    
    for (i=0;i<global_w.n_elem;i++)
    {
        lambda=global_offset[i]+global_x[i]*beta;
        choice=choices[i];
        obsresults=multquantal0(global_y(i),lambda);
        results=results+global_w(i)*obsresults;
        
        
       
    }
    
    
    return results;
}
