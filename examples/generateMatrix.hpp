//                                MFEM Covariance Generator
//
//
// Sample runs:  ./covariance 2
//               ./covariance 3
//
// Description:  This example code generate a covariance
// matrix in a unitary square (or cube) domain with exponential kernel:
// k(r) = exp( -r / lambda )
// where r = || x_i - x_j ||_2 is the Euclidian distance between any two points
// x_i and x_j in R^n, and lambda is the correlation length.
// In this test case we use a structured quadrilateral (2D) or hexaheadral (3D) grid.
// We use lexicographic ordering of the unknowns.
//
// OUTPUTS:
// (1) G.dat the symmetric dense covariance matrix. In this example, we use linear finite elements
//     to discretize the covariance matrix G, that is:
//     - size(G) = number of vertices in the mesh.
//     - G_ij = exp( - || x_i - x_j ||_2 / lambda ) for any two points i and j in the mesh.
// (2) X.dat: dense matrix that stores the vertices coordinates:
//     - size(X) = nDimensions x Number of Vertices
//     - X(:,i) = cartesian coordinates of vertex i.
// (3) A.mtx a symmetric sparse matrix (in triplet format) of the same size as G,
//     that can be used to compute the appropriate reordering of the entries in G.
//     More specifically the matrix A is the finite element discretization of the diffusion
//     reaction operator
//     a(u,v) = \int_\Omega 1/lambda^2 u*v + \int_\Omega grad(u) * grad(v).
//     NOTE: The rationale of using A to reorder G is that it can be proven
//     (see Lindgren, Finn, Håvard Rue, and Johan Lindström.
//     "An explicit link between Gaussian fields and Gaussian Markov random fields:
//      the stochastic partial differential equation approach."
//      Journal of the Royal Statistical Society: Series B 73, no. 4 (2011): 423-498.)
//
// INPUTS:
// ./covariance <ndim> [<nref>] [<corrlen>]
// --  <ndim>      INTEGER   REQUIRED whenever we are in 2D or in 3D (2 or 3).
// --  <nelem>     INTEGER   OPTIONAL: number of elements in each direction (DEFAULT 16).
// --  <corrlen>   REAL      OPTIONAL: correlation length is the scaling parameter in the covariance functions.
//                           The smaller corrlen the faster the entries of G will go to 0.
//                           (DEFAULT 0.2)

#include <fstream>
#include "mfem.hpp"
#include "dense/BLASLAPACKWrapper.hpp"

using namespace std;
using namespace mfem;

class ExponentialKernel
{
public:
	ExponentialKernel(double corlen_, int ndim_):corlen(corlen_), ndim(ndim_), d(ndim){};
	double Eval(const Vector & xi, const Vector & xj);
	void Eval(DenseMatrix & X, DenseMatrix & G);
private:
	double corlen;
	int ndim;
	Vector d;
	Vector xi;
	Vector xj;
};

double ExponentialKernel::Eval(const Vector & xi, const Vector & xj)
{
	if(xi.Size() != ndim)
		mfem_error("ExponentialKernel::Eval #1");

	if(xj.Size() != ndim)
		mfem_error("ExponentialKernel::Eval #2");

	subtract(xi, xj, d);

	double r = d.Norml2()/corlen;

	return exp(-r);

}

void ExponentialKernel::Eval(DenseMatrix & X, DenseMatrix & G)
{
	if(X.Height() != ndim)
		mfem_error("ExponentialKernel::Eval #3");

	if(X.Width() != G.Height())
		mfem_error("ExponentialKernel::Eval #4");

	if(G.Width() != G.Height())
		mfem_error("ExponentialKernel::Eval #5");

	int n = X.Width();

	for(int i(0); i < n; ++i)
	{
		G(i,i) = 1.;
		X.GetColumnReference(i,xi);
		for(int j(0); j < i; ++j)
		{
			X.GetColumnReference(j,xj);
			G(i,j) = G(j,i) = Eval(xi,xj);
		}
	}
}

void idfun(const Vector & x, Vector & y)
{
	y = x;
}

double* generateMatrix(int argc, char *argv[], int *size)
{
   std::stringstream errmsg;
   errmsg << "./covariance <ndim> [<nref>] [<corrlen>]\n";
   errmsg << " -- <ndim>      INTEGER   REQUIRED whenever we are in 2D or in 3D (2 or 3).\n";
   errmsg << " -- <nelem>     INTEGER   OPTIONAL: number of elements in each direction (DEFAULT 16)..\n";
   errmsg << " -- <corrlen>   REAL      OPTIONAL: correlation length is the scaling parameter in the covariance functions. "
             "                          The smaller corrlen the faster the entries of G will go to 0. (DEFAULT 0.2)\n";
   int ndim;
   int nx = 16;
   double corlen = 0.2;
   double *mat;

   // switch(argc)
   // {
   // case 2:
	  //  ndim = atoi(argv[1]);
	  //  break;
   // case 3:
   //     ndim = atoi(argv[1]);
	  //  nx = atoi(argv[2]);
	  //  break;
   // case 4:
    ndim = atoi(argv[1]);
    nx = atoi(argv[2]);
    corlen = atof(argv[3]);
	  //  break;
   // default:
   //     cout << "Wrong Number of parameters.\n Usage:\n";
	  //  cout << errmsg.str();
	  //  return NULL;
   // }

   cout << "ndim = " << ndim << endl;
   cout << "nx = " << nx << endl;
   cout << "corlen = " << corlen << endl;

   if(ndim != 2 && ndim != 3)
   {
       std::cout << "USAGE ERROR: ndim parameter should be either 2 or 3.\n";
       std::cout << errmsg.str();
       return NULL;
   }
   if(nx < 2)
   {
       std::cout << "USAGE ERROR: nelem parameter should be bigger than or equal to 2.\n";
       std::cout << errmsg.str();
       return NULL;
   }
   if(corlen < 0)
   {
       std::cout << "USAGE ERROR: corrlen parameter should be positive.\n";
       std::cout << errmsg.str();
       return NULL;
   }
   std::cout << "Correlation: " << corlen << std::endl;


    Mesh *mesh;
    if(ndim == 2)
        mesh = new Mesh(nx,nx, Element::QUADRILATERAL, 1);
    else
        mesh = new Mesh(nx,nx,nx, Element::HEXAHEDRON, 1);

   // 3. Define a linear finite element space on the mesh.
   FiniteElementCollection *fec = new LinearFECollection;
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   FiniteElementSpace *vfespace = new FiniteElementSpace(mesh, fec, mesh->Dimension(), Ordering::byVDIM);
   cout << "Number of unknowns: " << fespace->GetVSize() << endl;

   // 4. Build the matrix X (size(X) = [nDimensions, nVertices]), with the vertices coordinates. X(:,i) = coordinates of vertex i.
   GridFunction x(vfespace);
   VectorFunctionCoefficient id_coeff(mesh->Dimension(), idfun);
   x.ProjectCoefficient(id_coeff);
   double * data;
   x.StealData(&data);
   DenseMatrix X( data, mesh->Dimension(), fespace->GetNDofs() );

   //5. Build the covariance matrix
   ExponentialKernel k(corlen, mesh->Dimension());
   int n = fespace->GetNDofs();
   *size=n;
   std::size_t n2 = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
   DenseMatrix G(new double[n2], n,n);
   k.Eval(X,G);

   // 6. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the reaction-diffusion operator 1/corlen^2-Delta
   ConstantCoefficient one(1.);
   ConstantCoefficient sigma(1./(corlen*corlen));
   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   a->AddDomainIntegrator(new MassIntegrator(sigma));
   a->Assemble();
   a->Finalize();
   //const SparseMatrix &A = a->SpMat();


   //// 7. Here G_lr, such that G_lr_ij = G_ij if A_ij = 0, and G_lr_ij = 0 if A_ij != 0.
   //int * i_A = A.GetI();
   //int * j_A = A.GetJ();
   //for(int irow = 0; irow < A.Size(); ++irow)
   //    for(int jpos = i_A[irow]; jpos < i_A[irow+1]; ++jpos)
   //        G(irow, j_A[jpos] ) = 0.0;

   mat=new double[n2];
   strumpack::blas::lacpy('N',n,n,G.Data(),n,mat,n);
   return mat;

   // 8. Free the used memory.
   delete a;
   delete vfespace;
   delete fespace;
   delete fec;
   delete mesh;

   return 0;
}
