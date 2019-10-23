//Log likelihood and its gradient
//for multinomial quantal model with integer response vector global_y and
//predictor array of matrices global_x.
//Weight vector global_w is used.

#include<armadillo>
using namespace arma;
struct fd1v
{
    double value;
    vec grad;
  
};
extern char choices[];
extern char choice;

extern mat global_x [ ];
extern ivec global_y;
extern vec global_w;
extern vec global_offset [];

fd1v multquantal1(int,vec);
fd1v multquantallik1(vec beta)
{
    fd1v results;
    fd1v obsresults;
    int i;
    vec lambda;
    
    results.value=0.0;
    results.grad=zeros(beta.n_elem);

    
    
    
    
    
    for (i=0;i<global_w.n_elem;i++)
    {
        lambda=global_offset[i]+global_x[i]*beta;
        choice=choices[i];
        obsresults=multquantal1(global_y(i),lambda);
        results.value=results.value+global_w(i)*obsresults.value;
        results.grad=results.grad+global_w(i)*trans(global_x[i])*obsresults.grad;
        
        
       
    }
    
    
    return results;
}
