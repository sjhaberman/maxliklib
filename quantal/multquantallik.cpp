//Log likelihood and its gradient and Hessian
//for multinomial quantal model with integerresponse vector global_y and
//predictor array of matrices global_x.
//Weight vector global_w is used.

#include<armadillo>
using namespace arma;
struct fd2v
{
    double value;
    vec grad;
    mat hess;
};
extern char choices[];
extern char choice;

extern mat global_x [ ];
extern ivec global_y;
extern vec global_w;
extern vec global_offset [];

fd2v multquantal(int,vec);
fd2v multquantallik(vec beta)
{
    fd2v results;
    fd2v obsresults;
    int i;
    vec lambda;
    
    results.value=0.0;
    results.grad=zeros(beta.n_elem);
    results.hess=zeros(beta.n_elem,beta.n_elem);
    
    
    
    
    
    for (i=0;i<global_w.n_elem;i++)
    {
        lambda=global_offset[i]+global_x[i]*beta;
        choice=choices[i];
        obsresults=multquantal(global_y(i),lambda);
        results.value=results.value+global_w(i)*obsresults.value;
        results.grad=results.grad+global_w(i)*trans(global_x[i])*obsresults.grad;
        results.hess=results.hess+global_w(i)*trans(global_x[i])*obsresults.hess*global_x[i];
        
       
    }
    
    
    return results;
}
