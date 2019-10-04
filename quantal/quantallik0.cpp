//Log likelihood
//for quantal response model with response vector global_y and
//predictor matrix global_x.
//Weight global_w is used.
//Model codes are found in quantal.cpp.
#include<armadillo>
using namespace arma;




extern char choices[];
extern char choice;
extern vec global_w;
extern ivec global_y;
extern mat global_x;
extern vec global_offset;
double quantal0(int,double);
double quantallik0(vec beta)
{
    double lambda;
    
    double obsresults;
    double results;
    int i;
    results=0.0;
    
    
   
    for (i=0;i<global_x.n_cols;i++)
    {
        lambda=global_offset(i)+dot(beta,global_x.col(i));
        choice=choices[i];
        obsresults=quantal0(global_y(i),lambda);
        results=results+global_w(i)*obsresults;
        
        
    }
    
    return results;
}
