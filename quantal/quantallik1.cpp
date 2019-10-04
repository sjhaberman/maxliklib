//Log likelihood and its gradient
//for quantal response model with response vector global_y and
//predictor matrix global_x.
//Weight global_w is used.
//Model codes are found in quantal.cpp.
#include<armadillo>
using namespace arma;
struct fd1
{
    double value;
    double der1;
    
};
struct fd1v
{
    double value;
    vec grad;
    
};


extern char choices[];
extern char choice;
extern vec global_w;
extern ivec global_y;
extern mat global_x;
extern vec global_offset;
fd1 quantal1(int,double);
fd1v quantallik1(vec beta)
{
    double lambda;
    
    fd1 obsresults;
    fd1v results;
    int i;
    results.value=0.0;
    results.grad=zeros(beta.n_elem);
    
   
    for (i=0;i<global_x.n_cols;i++)
    {
        lambda=global_offset(i)+dot(beta,global_x.col(i));
        choice=choices[i];
        obsresults=quantal1(global_y(i),lambda);
        results.value=results.value+global_w(i)*obsresults.value;
        results.grad=results.grad+global_w(i)*obsresults.der1*global_x.col(i);
        
        
    }
    
    return results;
}
