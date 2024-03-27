#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
#include <fstream>
#include <random>
#include <chrono>

#include "HSS/HSSMatrixMPI.hpp"

#include "mfem.hpp"
using namespace mfem;

class ExponentialKernel {
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

strumpack::DenseMatrix<double>
generateMatrix(int ndim, int nx, double corlen=0.2)
{
  std::stringstream errmsg;
  errmsg << "./covariance <ndim> [<nref>] [<corrlen>]\n";
  errmsg << " -- <ndim>      INTEGER   REQUIRED whenever we are in 2D or in 3D (2 or 3).\n";
  errmsg << " -- <nelem>     INTEGER   OPTIONAL: number of elements in each direction (DEFAULT 16)..\n";
  errmsg << " -- <corrlen>   REAL      OPTIONAL: correlation length is the scaling parameter in the covariance functions. "
    "                          The smaller corrlen the faster the entries of G will go to 0. (DEFAULT 0.2)\n";

  assert(ndim == 2 || ndim == 3);
  assert(nx >= 2);
  assert(corlen >= 0);
  std::cout << "Correlation: " << corlen << std::endl;

  Mesh cart_mesh = ndim == 2 ?
    Mesh::MakeCartesian2D(nx, nx, Element::QUADRILATERAL, 1, 1., 1., false) :
    Mesh::MakeCartesian3D(nx, nx, nx, Element::HEXAHEDRON, 1., 1., 1., false);
  Mesh* mesh = &cart_mesh;

  // 3. Define a linear finite element space on the mesh.
  FiniteElementCollection *fec = new LinearFECollection;
  FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
  FiniteElementSpace *vfespace = new FiniteElementSpace(mesh, fec, mesh->Dimension(), Ordering::byVDIM);
  std::cout << "Number of unknowns: " << fespace->GetVSize() << std::endl;

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
  DenseMatrix G(new double[n*n], n,n);
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

  strumpack::DenseMatrix<double> M(n, n);
  strumpack::blas::lacpy('N', n, n, G.Data(), n, M.data(), n);

  // 8. Free the used memory.
  delete a;
  delete vfespace;
  delete fespace;
  delete fec;
  // delete mesh;

  return M;
}

/* Modified to not do dissection, but bisection. */
void ND1D(int bx, int ex, int by, int ey, int bz, int ez,
          int nx, int ny, int nz, int offset, int* order) {
  /* 1D Nested Dissection of a line of an nx x ny x nz cuboid mesh                  */
  /* Ordering is written in array order with starting positing "offset".            */
  int sx = ex-bx+1, sy = ey-by+1, sz = ez-bz+1;
  if (sx<=0 || sy<=0 || sz<=0)
    return;
  if (sx==1 && sy==1 && sz==1) {
    order[offset] = (bx-1)*ny*nz+(by-1)*nz+bz;
    return;
  }
  if (sy==1 && sz==1) {
    int hx = bx+sx/2;
    ND1D(bx,hx-1,by,ey,bz,ez,nx,ny,nz,offset        ,order);
    ND1D(hx,ex  ,by,ey,bz,ez,nx,ny,nz,offset+(hx-bx),order);
    // order[offset+ex-bx] = (hx-1)*ny*nz+(by-1)*nz+bz;
  } else if (sx==1 && sz==1) {
    int hy = by+sy/2;
    ND1D(bx,ex,by,hy-1,bz,ez,nx,ny,nz,offset        ,order);
    ND1D(bx,ex,hy,ey  ,bz,ez,nx,ny,nz,offset+(hy-by),order);
    //order[offset+ey-by] = (bx-1)*ny*nz+(hy-1)*nz+bz;
  } else if (sx==1 && sy==1) {
    int hz = bz+sz/2;
    ND1D(bx,ex,by,ey,bz,hz-1,nx,ny,nz,offset        ,order);
    ND1D(bx,ex,by,ey,hz,ez  ,nx,ny,nz,offset+(hz-bz),order);
    //order[offset+ez-bz] = (bx-1)*ny*nz+(by-1)*nz+hz;
  } else {
    std::cerr << "Internal error ND1D\n";
    exit(1);
  }
}

void ND2D(int bx, int ex, int by, int ey, int bz, int ez,
          int nx, int ny, int nz, int offset, int cut, int* order) {
  /* 2D Nested Dissection of a rectangle of an nx x ny x nz rectangular mesh       */
  /* Cut defines how to cut the domain (e.g., if constant z, 0: x-wise, 1: y-wise) */
  /* Ordering is written in array order with starting positing "offset".           */
  int sx = ex-bx+1, sy = ey-by+1, sz=ez-bz+1;
  if (sx>1 && sy>1 && sz>1) {
    std::cerr << "Internal error ND2D\n";
    exit(1);
  }
  if (sx<=0 || sy<=0 || sz<=0) return;
  if (sx==2 && sy==2) {
    order[offset]   = (bx-1)*nx*nz+(by-1)*nz+bz;
    order[offset+1] = (bx)  *nx*nz+(by)  *nz+bz;
    order[offset+2] = (bx)  *nx*nz+(by-1)*nz+bz;
    order[offset+3] = (bx-1)*nx*nz+(by)  *nz+bz;
    return;
  }
  if (sx==2 && sz==2) {
    order[offset]   = (bx-1)*nx*nz+(by-1)*nz+bz;
    order[offset+1] = (bx)  *nx*nz+(by-1)*nz+bz+1;
    order[offset+2] = (bx)  *nx*nz+(by-1)*nz+bz;
    order[offset+3] = (bx-1)*nx*nz+(by-1)*nz+bz+1;
    return;
  }
  if (sy==2 && sz==2) {
    order[offset]   = (bx-1)*nx*nz+(by-1)*nz+bz;
    order[offset+1] = (bx-1)*nx*nz+(by-1)*nz+bz+1;
    order[offset+2] = (bx-1)*nx*nz+(by)  *nz+bz;
    order[offset+3] = (bx-1)*nx*nz+(by)  *nz+bz+1;
    return;
  }
  if ((sx==1 && sy==1) || (sx==1 && sz==1) || (sy==1 && sz==1)) {
    ND1D(bx,ex,by,ey,bz,ez,nx,ny,nz,offset,order);
    return;
  }
  if (sz==1) {
    if (cut==0) {
      int hx = bx+sx/2;
      ND2D(bx,hx-1,by,ey,bz,ez,nx,ny,nz,offset           ,1,order);
      ND2D(hx,ex  ,by,ey,bz,ez,nx,ny,nz,offset+(hx-bx)*sy,1,order);
      //ND1D(hx  ,hx  ,by,ey,bz,ez,nx,ny,nz,offset+(ex-bx)*sy,  order);
    } else {
      int hy = by+sy/2;
      ND2D(bx,ex,by,hy-1,bz,ez,nx,ny,nz,offset           ,0,order);
      ND2D(bx,ex,hy,ey  ,bz,ez,nx,ny,nz,offset+(hy-by)*sx,0,order);
      //ND1D(bx,ex,hy  ,hy  ,bz,ez,nx,ny,nz,offset+(ey-by)*sx,  order);
    }
  } else if (sy==1) {
    if (cut==0) {
      int hx = bx+sx/2;
      ND2D(bx,hx-1,by,ey,bz,ez,nx,ny,nz,offset           ,1,order);
      ND2D(hx,ex  ,by,ey,bz,ez,nx,ny,nz,offset+(hx-bx)*sz,1,order);
      //ND1D(hx  ,hx  ,by,ey,bz,ez,nx,ny,nz,offset+(ex-bx)*sz,  order);
    } else {
      int hz = bz+sz/2;
      ND2D(bx,ex,by,ey,bz,hz-1,nx,ny,nz,offset           ,0,order);
      ND2D(bx,ex,by,ey,hz,ez  ,nx,ny,nz,offset+(hz-bz)*sx,0,order);
      //ND1D(bx,ex,by,ey,hz  ,hz  ,nx,ny,nz,offset+(ez-bz)*sx,  order);
    }
  } else if (sx==1) {
    if (cut==0) {
      int hy = by+sy/2;
      ND2D(bx,ex,by,hy-1,bz,ez,nx,ny,nz,offset           ,0,order);
      ND2D(bx,ex,hy,ey  ,bz,ez,nx,ny,nz,offset+(hy-by)*sz,0,order);
      //ND1D(bx,ex,hy  ,hy  ,bz,ez,nx,ny,nz,offset+(ey-by)*sz,  order);
    } else {
      int hz = bz+sz/2;
      ND2D(bx,ex,by,ey,bz  ,hz-1,nx,ny,nz,offset           ,0,order);
      ND2D(bx,ex,by,ey,hz,ez  ,nx,ny,nz,offset+(hz-bz)*sy,0,order);
      //ND1D(bx,ex,by,ey,hz  ,hz  ,nx,ny,nz,offset+(ez-bz)*sy,  order);
    }
  }
}

void ND3D(int bx, int ex, int by, int ey, int bz, int ez,
          int nx, int ny, int nz, int offset, int cut, int* order) {
  /* 3D Nested Dissection of subdomain [bx:ex]x[by:ey]x[bz:ez] of an nx x ny x nz rectangular mesh */
  /* Cut defines how to cut the domain (0: x-wise, 1: y-wise, 2: z-wise)                           */
  /* Points of the domain are assumed to be numbered y-wise.                                       */
  /* Ordering is written in array order with starting positing "offset".                           */
  int sx = ex-bx+1, sy = ey-by+1, sz = ez-bz+1;
  if (sx<=0 || sy<=0 || sz<=0) return;
  if (sx==2 && sy==2 && sz==2) {
    order[offset]   = (bx-1)*nx*nz+(by-1)*nz+bz;
    order[offset+1] = (bx-1)*nx*nz+(by-1)*nz+bz+1;
    order[offset+2] = (bx-1)*nx*nz+(by)  *nz+bz;
    order[offset+3] = (bx-1)*nx*nz+(by)  *nz+bz+1;;
    order[offset+4] = (bx)  *nx*nz+(by-1)*nz+bz;
    order[offset+5] = (bx)  *nx*nz+(by-1)*nz+bz+1;
    order[offset+6] = (bx)  *nx*nz+(by)  *nz+bz;
    order[offset+7] = (bx)  *nx*nz+(by)  *nz+bz+1;;
    return;
  }
  if ((sx==1 && sy==1) || (sx==1 && sz==1) || (sy==1 && sz==1)) {
    ND1D(bx,ex,by,ey,bz,ez,nx,ny,nz,offset,order);
    return;
  }
  if (sx==1 || sy==1 || sz==1) {
    ND2D(bx,ex,by,ey,bz,ez,nx,ny,nz,offset,0,order);
    return;
  }
  if(cut==0) {
    int hx = bx+sx/2;
    ND3D(bx,hx-1,by,ey,bz,ez,nx,ny,nz,offset              ,1,order);
    ND3D(hx,ex  ,by,ey,bz,ez,nx,ny,nz,offset+(hx-bx)*sy*sz,1,order);
    //ND2D(hx  ,hx  ,by,ey,bz,ez,nx,ny,nz,offset+(ex-bx)*sy*sz,0,order);
  } else if(cut==1) {
    int hy = by+sy/2;
    ND3D(bx,ex,by,hy-1,bz,ez,nx,ny,nz,offset              ,2,order);
    ND3D(bx,ex,hy,ey  ,bz,ez,nx,ny,nz,offset+(hy-by)*sx*sz,2,order);
    //ND2D(bx,ex,hy  ,hy  ,bz,ez,nx,ny,nz,offset+(ey-by)*sx*sz,0,order);
  } else {
    int hz = bz+sz/2;
    ND3D(bx,ex,by,ey,bz,hz-1,nx,ny,nz,offset              ,0,order);
    ND3D(bx,ex,by,ey,hz,ez  ,nx,ny,nz,offset+(hz-bz)*sx*sy,0,order);
    //ND2D(bx,ex,by,ey,hz  ,hz  ,nx,ny,nz,offset+(ez-bz)*sx*sy,0,order);
  }
}

std::vector<int> generatePermutation(int dim, int k) {
  std::vector<int> order;
  k++;
  switch(dim) {
  case 1:
    order.resize(k);
    ND1D(1, k, 1, 1, 1, 1, k, 1, 1, 0, order.data());
    break;
  case 2:
    order.resize(k*k);
    ND2D(1, k, 1, k, 1, 1, k, k, 1, 0, 0, order.data());
    break;
  case 3:
    order.resize(k*k*k);
    ND3D(1, k, 1, k, 1, k, k, k, k, 0, 0, order.data());
    break;
  }
  return order;
}


int main(int argc, char* argv[]) {

MPI_Init(&argc, &argv);

  int ndim = 2, nx = 16;
  double corlen = 0.2;
  switch(argc) {
  case 2:
    ndim = std::atoi(argv[1]);
    break;
  case 3:
    ndim = std::atoi(argv[1]);
    nx = std::atoi(argv[2]);
    break;
  default:
    ndim = std::atoi(argv[1]);
    nx = std::atoi(argv[2]);
    corlen = std::atof(argv[3]);
    break;
  // default:
  //   std::cout << "Wrong Number of parameters.\n Usage:\n";
  }

{
  strumpack::MPIComm c;
  strumpack::BLACSGrid grid(c);
  strumpack::DenseMatrix<double> Aseq;
  strumpack::DistributedMatrix<double> A;
  int m = 0;


  if(!strumpack::mpi_rank()){

  
  auto C = generateMatrix(ndim, nx, corlen);
  auto P = generatePermutation(ndim, nx);

  {
    auto Pc = P;
    std::sort(Pc.begin(), Pc.end());
    for (std::size_t i=0; i<Pc.size(); i++) {
      if (i+1 != Pc[i]) std::cerr << "invalid permutation!" << std::endl;
    }
  }

  Aseq = C;
  for (int j=0; j<Aseq.cols(); j++)
    for (int i=0; i<Aseq.rows(); i++)
      Aseq(i, j) = C(P[i]-1, P[j]-1);

  m = Aseq.rows();
  std::cout << "# matrix generated with " << m << " rows" << std::endl;
  }

  MPI_Bcast(&m, 1, strumpack::mpi_type<int>(), 0, MPI_COMM_WORLD);
  
  A = strumpack::DistributedMatrix<double>(&grid, m, m);
  // A.scatter(Aseq);
  std::size_t B = 10000;
  for (std::size_t r=0; r<m; r+=B) {
    auto nr = std::min(B, m-r);
    strumpack::DenseMatrixWrapper<double> Ar(nr, m, Aseq, r, 0);
    strumpack::copy(nr, m, Ar, 0, A, r, 0, grid.ctxt_all());
  }
  
  std::cout << "# scatter success" << std::endl;
  Aseq.clear();


   strumpack::HSS::HSSOptions<double> hss_opts;
   hss_opts.set_from_command_line(argc, argv);


    std::vector<int> ranks;
    std::vector<double> times, errors;

    for (int r=0; r<10; r++) {
      auto begin = std::chrono::steady_clock::now();
      strumpack::HSS::HSSMatrixMPI<double> H(A, hss_opts);
      auto end = std::chrono::steady_clock::now();
      if (H.is_compressed()) {
        auto max_levels = H.max_levels();
        if (c.is_root())
          std::cout << "# created M matrix of dimension "
                    << H.rows() << " x " << H.cols()
                    << " with " << max_levels << " levels" << std::endl;
        auto T = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        times.push_back(T);
        if (c.is_root())
          std::cout << "# total compression time = " << T << " [10e-3s]" << std::endl
                    << "# compression succeeded!" << std::endl;
      } else {
        if (c.is_root())
          std::cout << "# compression failed!!!!!!!!" << std::endl;
        return 1;
      }
      auto rk = H.max_rank();
      if (c.is_root())
        std::cout << "# rank(H) = " << rk << std::endl;
      ranks.push_back(rk);

      auto tot_mem_H = H.total_memory();
      if (c.is_root())
        std::cout << "# memory(H) = " << tot_mem_H/1e6 << " MB, "
                  << 100. * tot_mem_H / A.total_memory() << "% of dense" << std::endl;

      // H.print_info();
      auto Hdense = H.dense();
      Hdense.scaled_add(-1., A);
      auto rel_err = Hdense.normF() / A.normF();
      errors.push_back(rel_err);
      auto Hdnorm = Hdense.normF();
      if (c.is_root())
        std::cout << "# relative error = ||A-H*I||_F/||A||_F = "
                  << rel_err << std::endl
                  << "# absolute error = ||A-H*I||_F = "
                  << Hdnorm << std::endl;
    }

    std::sort(ranks.begin(), ranks.end());
    std::sort(times.begin(), times.end());
    std::sort(errors.begin(), errors.end());

    if (c.is_root())
      std::cout << "min, median, max" << std::endl
                << "ranks: " << ranks[0] << " "
                << ranks[ranks.size()/2] << " "
                << ranks[ranks.size()-1] << std::endl
                << "times: " << times[0] << " "
                << times[times.size()/2] << " "
                << times[times.size()-1] << std::endl
                << "errors: " << errors[0] << " "
                << errors[errors.size()/2] << " "
                << errors[errors.size()-1] << std::endl;

    // strumpack::TimerList::Finalize();
  }

MPI_Finalize();

  return 0;
}
