%{
#include "StrumpackSparseSolver.hpp"
%}

%include "StrumpackSparseSolver.hpp"

// Allow native fortran arrays to be passed to pointer/arrays
%include <typemaps.i>
%apply SWIGTYPE ARRAY[] {
    int*,
    float*,
    double*
};

%template(StrumpackSparseSolverReal4) strumpack::StrumpackSparseSolver<float>;
%template(StrumpackSparseSolverReal8) strumpack::StrumpackSparseSolver<double>;
