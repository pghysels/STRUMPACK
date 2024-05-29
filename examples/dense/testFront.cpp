#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
#include <fstream>
#include <random>
#include <chrono>

#include "HSS/HSSMatrix.hpp"
#define ERROR_TOLERANCE 1e2
#define SOLVE_TOLERANCE 1e-12

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

  auto M = strumpack::DenseMatrix<double>::read(argv[1]);
  int n = M.rows();
  {
    auto nx = std::sqrt(n);
    auto P = generatePermutation(2, nx-1); // why -1 here??
    M.lapmt(P, true);   // permute columns
    M.lapmr(P, true);   // permute rows
  }

  strumpack::HSS::HSSOptions<double> hss_opts;
  hss_opts.set_from_command_line(argc, argv);

  std::vector<int> ranks;
  std::vector<double> times, errors;

  for (int r=0; r<3; r++) {
    auto begin = std::chrono::steady_clock::now();
    strumpack::HSS::HSSMatrix<double> H(M, hss_opts);
    auto end = std::chrono::steady_clock::now();
    if (H.is_compressed()) {
      std::cout << "# created M matrix of dimension "
                << H.rows() << " x " << H.cols()
                << " with " << H.levels() << " levels" << std::endl;
      auto T = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
      times.push_back(T);
      std::cout << "# total compression time = " << T << " [10e-3s]" << std::endl;
      std::cout << "# compression succeeded!" << std::endl;
    } else {
      std::cout << "# compression failed!!!!!!!!" << std::endl;
      return 1;
    }
    std::cout << "# rank(H) = " << H.rank() << std::endl;
    ranks.push_back(H.rank());
    std::cout << "# memory(H) = " << H.memory()/1e6 << " MB, "
              << 100. * H.memory() / M.memory() << "% of dense" << std::endl;

    // H.print_info();
    auto Hdense = H.dense();
    Hdense.scaled_add(-1., M);
    auto rel_err = Hdense.normF() / M.normF();
    errors.push_back(rel_err);
    std::cout << "# relative error = ||A-H*I||_F/||A||_F = "
              << rel_err << std::endl;
    std::cout << "# absolute error = ||A-H*I||_F = " << Hdense.normF() << std::endl;
    // if (Hdense.normF() / M.normF() > ERROR_TOLERANCE
    //     * std::max(hss_opts.rel_tol(),hss_opts.abs_tol())) {
    //   std::cout << "ERROR: compression error too big!!" << std::endl;
    //   return 1;
    // }
  }

  std::sort(ranks.begin(), ranks.end());
  std::sort(times.begin(), times.end());
  std::sort(errors.begin(), errors.end());

  std::cout << "min, median, max" << std::endl;
  std::cout << "ranks: " << ranks[ranks.size()/2] << " "
            << ranks[ranks.size()/2]-ranks[0] << " "
            << ranks.back() - ranks[ranks.size()/2] << std::endl;
  std::cout << "times: " << times[times.size()/2] << " "
            << times[times.size()/2]-times[0] << " "
            << times.back() - times[times.size()/2] << std::endl;
  std::cout << "errors: " << errors[errors.size()/2] << " "
            << errors[errors.size()/2]-errors[0] << " "
            << errors.back() - errors[errors.size()/2] << std::endl;

  // strumpack::TimerList::Finalize();

  return 0;
}
