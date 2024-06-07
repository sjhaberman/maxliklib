//Get basic control data for maximum likelihood.
//Controlfile is the source of the information.
//The file includes multiple lines of text.  Each line has two strings that are separated by
//a blank character. Each of these strings contains no blank character.  The first string
//is a key word defining the nature of the entry.  The second string is the value associated
//with the first entry.  Results are a two-dimensional field of strings.
#include<armadillo>
#include<string>
using namespace std;
using namespace arma;
field<string> getcontrolfile(string & controlfile)
{
    field<string>result;
    try{result.load(controlfile);}
    catch(...){cout<<"Control file not read."<<endl; return result;}
    return result;
}
 