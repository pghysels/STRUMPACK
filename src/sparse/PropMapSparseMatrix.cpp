/*
 * STRUMPACK -- STRUctured Matrices PACKage, Copyright (c) 2014, The
 * Regents of the University of California, through Lawrence Berkeley
 * National Laboratory (subject to receipt of any required approvals
 * from the U.S. Dept. of Energy).  All rights reserved.
 *
 * If you have questions about your rights to use or distribute this
 * software, please contact Berkeley Lab's Technology Transfer
 * Department at TTD@lbl.gov.
 *
 * NOTICE. This software is owned by the U.S. Department of Energy. As
 * such, the U.S. Government has been granted for itself and others
 * acting on its behalf a paid-up, nonexclusive, irrevocable,
 * worldwide license in the Software to reproduce, prepare derivative
 * works, and perform publicly and display publicly.  Beginning five
 * (5) years after the date permission to assert copyright is obtained
 * from the U.S. Department of Energy, and subject to any subsequent
 * five (5) year renewals, the U.S. Government is granted for itself
 * and others acting on its behalf a paid-up, nonexclusive,
 * irrevocable, worldwide license in the Software to reproduce,
 * prepare derivative works, distribute copies to the public, perform
 * publicly and display publicly, and to permit others to do so.
 *
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li.
 *             (Lawrence Berkeley National Lab, Computational Research
 *             Division).
 *
 */
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <tuple>

#include "PropMapSparseMatrix.hpp"
#include "dense/DistributedMatrix.hpp"
#include "misc/Triplet.hpp"
#include "ordering/MatrixReorderingMPI.hpp"
#include "CSRMatrixMPI.hpp"
#include "EliminationTreeMPIDist.hpp"


namespace strumpack {

  template<typename scalar_t,typename integer_t>
  PropMapSparseMatrix<scalar_t,integer_t>::PropMapSparseMatrix()
    : CompressedSparseMatrix<scalar_t,integer_t>() { }

  template<typename scalar_t,typename integer_t> void
  PropMapSparseMatrix<scalar_t,integer_t>::setup
  (const CSRMatrixMPI<scalar_t,integer_t>& Ampi,
   const MatrixReorderingMPI<scalar_t,integer_t>& nd,
   const EliminationTreeMPIDist<scalar_t,integer_t>& et,
   bool duplicate_fronts) {
    n_ = Ampi.size();
    nnz_ = Ampi.nnz();
    const auto& comm = et.Comm();
    auto P = comm.size();
    auto eps = blas::lamch<real_t>('E');

    std::vector<std::tuple<int,int,int>> dest(Ampi.local_nnz());
    std::vector<std::size_t> scnts(P);
#pragma omp parallel for
    for (integer_t r=0; r<Ampi.local_rows(); r++) {
      auto r_perm = nd.perm()[r + Ampi.begin_row()];
      auto hij = Ampi.ptr(r+1) - Ampi.ptr(0);
      for (integer_t j=Ampi.ptr(r)-Ampi.ptr(0); j<hij; j++) {
        if (std::abs(Ampi.val(j)) > eps) {
          auto indj = Ampi.ind(j);
          auto c_perm = nd.perm()[indj];
          dest[j] = et.get_sparse_mapped_destination
            (Ampi, r, indj, r_perm, c_perm, duplicate_fronts);
          auto& d = dest[j];
          auto hip = std::get<0>(d) + std::get<1>(d);
          for (int p=std::get<0>(d); p<hip; p+=std::get<2>(d))
#pragma omp atomic
            scnts[p]++;
        }
      }
    }

    using Triplet = Triplet<scalar_t,integer_t>;
    std::vector<std::vector<Triplet>> sbuf(P);
    for (int p=0; p<P; p++)
      sbuf[p].reserve(scnts[p]);
    for (integer_t r=0; r<Ampi.local_rows(); r++) {
      auto r_perm = nd.perm()[r + Ampi.begin_row()];
      auto hij = Ampi.ptr(r+1) - Ampi.ptr(0);
      for (integer_t j=Ampi.ptr(r)-Ampi.ptr(0); j<hij; j++) {
        auto a = Ampi.val(j);
        if (std::abs(a) > eps) {
          Triplet t = {r_perm, nd.perm()[Ampi.ind(j)], a};
          auto& d = dest[j];
          auto hip = std::get<0>(d) + std::get<1>(d);
          for (int p=std::get<0>(d); p<hip; p+=std::get<2>(d))
            sbuf[p].push_back(t); // do NOT use emplace
        }
      }
    }
    auto triplets = comm.all_to_all_v(sbuf);
    Triplet::free_mpi_type();

    // TODO this sort can be avoided? first make the CSR/CSC
    // representation, then sort that row per row in parallel
    std::sort
      (triplets.begin(), triplets.end(),
       [](const Triplet& a, const Triplet& b) {
         // sort according to column, then rows
         if (a.c != b.c) return (a.c < b.c);
         return (a.r < b.r);
       });

    integer_t lnnz = triplets.size();
    local_cols_ = lnnz ? 1 : 0;
    for (integer_t t=1; t<lnnz; t++)
      if (triplets[t].c != triplets[t-1].c) local_cols_++;
    ptr_.resize(local_cols_+1);
    global_col_.resize(local_cols_);
    ind_.resize(lnnz);
    val_.resize(lnnz);
    integer_t col = 0;
    ptr_[col] = 0;
    if (local_cols_) {
      ptr_[1] = 0;
      global_col_[col] = triplets[0].c;
    }
    for (integer_t j=0; j<lnnz; j++) {
      ind_[j] = triplets[j].r;
      val_[j] = triplets[j].v;
      if (j > 0 && (triplets[j].c != triplets[j-1].c)) {
        col++;
        ptr_[col+1] = ptr_[col];
        global_col_[col] = triplets[j].c;
      }
      ptr_[col+1]++;
    }
  }

  template<typename scalar_t,typename integer_t> void
  PropMapSparseMatrix<scalar_t,integer_t>::
  print_dense(const std::string& name) const {
    DenseMatrix<scalar_t> M(n_, n_);
    M.fill(scalar_t(0.));
    for (integer_t c=0; c<local_cols_; c++)
      for (integer_t j=ptr_[c]; j<ptr_[c+1]; j++)
        M(ind_[j], global_col_[c]) = val_[j];
    M.print(name);
  }

  template<typename scalar_t,typename integer_t> void
  PropMapSparseMatrix<scalar_t,integer_t>::extract_separator
  (integer_t shi, const std::vector<std::size_t>& I,
   const std::vector<std::size_t>& J, DenseM_t& B, int depth) const {
    integer_t m = I.size(), n = J.size();
    if (m == 0 || n == 0) return;
    for (integer_t j=0; j<n; j++) {
      integer_t c = find_global(J[j]);
      if (c == local_cols_ || global_col_[c] != integer_t(J[j])) {
        std::fill(B.ptr(0,j), B.ptr(m, j), scalar_t(0.));
        continue;
      }
      auto rmin = ind_[ptr_[c]];
      auto rmax = ind_[ptr_[c+1]-1];
      for (integer_t i=0; i<m; i++) {
        integer_t r = I[i];
        if (r >= rmin && r <= rmax &&
            (global_col_[c] < shi || r < shi)) {
          auto a_pos = ptr_[c];
          auto a_max = ptr_[c+1];
          while (a_pos < a_max-1 && ind_[a_pos] < r) a_pos++;
          B(i,j) = (ind_[a_pos] == r) ?
            val_[a_pos] : scalar_t(0.);
        } else B(i,j) = scalar_t(0.);
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  PropMapSparseMatrix<scalar_t,integer_t>::front_multiply
  (integer_t slo, integer_t shi, const std::vector<integer_t>& upd,
   const DenseM_t& R, DenseM_t& Sr, DenseM_t& Sc, int depth) const {
    const integer_t dupd = upd.size();
    const std::size_t clo = find_global(slo);
    const std::size_t chi = find_global(shi);
    const auto ds = shi - slo;
    const auto nbvec = R.cols();

    const auto B = 4; // blocking parameter
#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop default(shared)                    \
  if(depth < params::task_recursion_cutoff_level)
#endif
    for (std::size_t k=0; k<nbvec; k+=B) {
      long long int local_flops = 0;
      for (std::size_t c=clo; c<chi; c++) {
        const auto col = global_col_[c];
        integer_t row_upd = 0;
        const auto hij = ptr_[c+1];
        for (auto j=ptr_[c]; j<hij; j++) {
          const auto row = ind_[j];
          if (row >= slo) {
            if (row < shi) {
              const auto hikk = std::min(k+B, nbvec);
              const auto a = val_[j];
              for (std::size_t kk=k; kk<hikk; kk++) {
                Sr(row-slo, kk) += a * R(col-slo, kk);
                Sc(col-slo, kk) += a * R(row-slo, kk);
              }
              local_flops += 4 * (hikk - k);
            } else {
              while (row_upd < dupd && upd[row_upd] < row) row_upd++;
              if (row_upd == dupd) break;
              if (upd[row_upd] == row) {
                const auto hikk = std::min(k+B, nbvec);
                const auto a = val_[j];
                for (std::size_t kk=k; kk<hikk; kk++) {
                  Sr(ds+row_upd, kk) += a * R(col-slo, kk);
                  Sc(col-slo, kk) += blas::my_conj(a) * R(ds+row_upd, kk);
                }
                local_flops += 4 * (hikk - k);
              }
            }
          }
        }
      }
      STRUMPACK_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
      STRUMPACK_SPARSE_SAMPLE_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
    }

#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop default(shared)                    \
  if(depth < params::task_recursion_cutoff_level)
#endif
    for (std::size_t k=0; k<nbvec; k+=B) {
      long long int local_flops = 0;
      for (integer_t i=0, c=chi; i<dupd; i++) { // update columns
        c = find_global(upd[i], c);
        if (c == local_cols_ || global_col_[c] != upd[i]) continue;
        const auto hij = ptr_[c+1];
        for (auto j=ptr_[c]; j<hij; j++) {
          const auto row = ind_[j];
          if (row >= slo) {
            if (row < shi) {
              const auto a = val_[j];
              const auto hikk = std::min(k+B, nbvec);
              for (std::size_t kk=k; kk<hikk; kk++) {
                Sr(row-slo, kk) += a * R(ds+i, kk);
                Sc(ds+i, kk) += blas::my_conj(a) * R(row-slo, kk);
              }
              local_flops += 4 * (hikk - k);
            } else break;
          }
        }
      }
      STRUMPACK_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
      STRUMPACK_SPARSE_SAMPLE_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
    }
  }

  template<typename scalar_t,typename integer_t> void
  PropMapSparseMatrix<scalar_t,integer_t>::front_multiply_F11
  (Trans op, integer_t slo, integer_t shi,
   const DenseM_t& R, DenseM_t& S, int depth) const {
    const std::size_t clo = find_global(slo);
    const std::size_t chi = find_global(shi);
    const auto nbvec = R.cols();
    const auto B = 4; // blocking parameter
#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop default(shared)                    \
  if(depth < params::task_recursion_cutoff_level)
#endif
    for (std::size_t k=0; k<nbvec; k+=B) {
      long long int local_flops = 0;
      for (std::size_t c=clo; c<chi; c++) {
        const auto col = global_col_[c];
        const auto hij = ptr_[c+1];
        for (auto j=ptr_[c]; j<hij; j++) {
          const auto row = ind_[j];
          if (row >= slo) {
            if (row < shi) {
              const auto hikk = std::min(k+B, nbvec);
              const auto a = val_[j];
              for (std::size_t kk=k; kk<hikk; kk++) {
                if (op == Trans::N)
                  S(row-slo, kk) += a * R(col-slo, kk);
                else if (op == Trans::T)
                  S(col-slo, kk) += a * R(row-slo, kk);
                else
                  S(col-slo, kk) += blas::my_conj(a) * R(row-slo, kk);
              }
              local_flops += 2 * (hikk - k);
            } else break;
          }
        }
      }
      STRUMPACK_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
      STRUMPACK_SPARSE_SAMPLE_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
    }
  }

  template<typename scalar_t,typename integer_t> void
  PropMapSparseMatrix<scalar_t,integer_t>::front_multiply_F21
  (Trans op, integer_t slo, integer_t shi, const std::vector<integer_t>& upd,
   const DenseM_t& R, DenseM_t& S, int depth) const {
    const integer_t dupd = upd.size();
    const std::size_t clo = find_global(slo);
    const std::size_t chi = find_global(shi);
    const auto nbvec = R.cols();
    const auto B = 4; // blocking parameter
#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop default(shared)                    \
  if(depth < params::task_recursion_cutoff_level)
#endif
    for (std::size_t k=0; k<nbvec; k+=B) {
      long long int local_flops = 0;
      for (std::size_t c=clo; c<chi; c++) {
        const auto col = global_col_[c];
        integer_t row_upd = 0;
        const auto hij = ptr_[c+1];
        for (auto j=ptr_[c]; j<hij; j++) {
          const auto row = ind_[j];
          if (row >= shi) {
            while (row_upd < dupd && upd[row_upd] < row) row_upd++;
            if (row_upd == dupd) break;
            if (upd[row_upd] == row) {
              const auto hikk = std::min(k+B, nbvec);
              const auto a = val_[j];
              for (std::size_t kk=k; kk<hikk; kk++) {
                if (op == Trans::N)
                  S(row_upd, kk) += a * R(col-slo, kk);
                else if (op == Trans::T)
                  S(col-slo, kk) += a * R(row_upd, kk);
                else
                  S(col-slo, kk) += blas::my_conj(a) * R(row_upd, kk);
              }
              local_flops += 2 * (hikk - k);
            }
          }
        }
      }
      STRUMPACK_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
      STRUMPACK_SPARSE_SAMPLE_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
    }
  }

  template<typename scalar_t,typename integer_t> void
  PropMapSparseMatrix<scalar_t,integer_t>::front_multiply_F12
  (Trans op, integer_t slo, integer_t shi, const std::vector<integer_t>& upd,
   const DenseM_t& R, DenseM_t& S, int depth) const {
    const integer_t dupd = upd.size();
    //const std::size_t clo = find_global(slo);
    const std::size_t chi = find_global(shi);
    //const auto ds = shi - slo;
    const auto nbvec = R.cols();
    const auto B = 4; // blocking parameter
#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop default(shared)                    \
  if(depth < params::task_recursion_cutoff_level)
#endif
    for (std::size_t k=0; k<nbvec; k+=B) {
      long long int local_flops = 0;
      for (integer_t i=0, c=chi; i<dupd; i++) { // update columns
        c = find_global(upd[i], c);
        if (c == local_cols_ || global_col_[c] != upd[i]) continue;
        const auto hij = ptr_[c+1];
        for (auto j=ptr_[c]; j<hij; j++) {
          const auto row = ind_[j];
          if (row >= slo) {
            if (row < shi) {
              const auto a = val_[j];
              const auto hikk = std::min(k+B, nbvec);
              for (std::size_t kk=k; kk<hikk; kk++) {
                if (op == Trans::N)
                  S(row-slo, kk) += a * R(i, kk);
                else if (op == Trans::T)
                  S(i, kk) += a * R(row-slo, kk);
                else
                  S(i, kk) += blas::my_conj(a) * R(row-slo, kk);
              }
              local_flops += 2 * (hikk - k);
            } else break;
          }
        }
      }
      STRUMPACK_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
      STRUMPACK_SPARSE_SAMPLE_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
    }
  }

  template<typename scalar_t,typename integer_t> void
  PropMapSparseMatrix<scalar_t,integer_t>::extract_front
  (DenseM_t& F11, DenseM_t& F12, DenseM_t& F21, integer_t slo,
   integer_t shi, const std::vector<integer_t>& upd, int depth) const {
    integer_t dim_upd = upd.size();
    auto c = find_global(slo);
    auto chi = find_global(shi, c);
    for (; c<chi; c++) {
      auto col = global_col_[c];
      integer_t row_ptr = 0;
      auto hij = ptr_[c+1];
      for (integer_t j=ptr_[c]; j<hij; j++) {
        auto row = ind_[j];
        if (row >= slo) {
          if (row < shi)
            F11(row-slo, col-slo) = val_[j];
          else {
            while (row_ptr<dim_upd && upd[row_ptr]<row)
              row_ptr++;
            if (row_ptr == dim_upd) break;
            if (upd[row_ptr] == row)
              F21(row_ptr, col-slo) = val_[j];
          }
        }
      }
    }
    for (integer_t i=0; i<dim_upd; ++i) { // update columns
      //while (c < local_cols_ && global_col_[c] < upd[i]) c++;
      c = find_global(upd[i], c);
      if (c == local_cols_ || global_col_[c] != upd[i]) continue;
      for (integer_t j=ptr_[c]; j<ptr_[c+1]; j++) {
        auto row = ind_[j];
        if (row >= slo) {
          if (row < shi)
            F12(row-slo, i) = val_[j];
          else break;
        }
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  PropMapSparseMatrix<scalar_t,integer_t>::push_front_elements
  (integer_t slo, integer_t shi, const std::vector<integer_t>& upd,
   std::vector<Triplet<scalar_t>>& e11, std::vector<Triplet<scalar_t>>& e12,
   std::vector<Triplet<scalar_t>>& e21) const {
    integer_t dim_upd = upd.size();
    auto c = find_global(slo);
    auto chi = find_global(shi, c);
    for (; c<chi; c++) {
      auto col = global_col_[c];
      integer_t row_ptr = 0;
      auto hij = ptr_[c+1];
      for (integer_t j=ptr_[c]; j<hij; j++) {
        auto row = ind_[j];
        if (row >= slo) {
          if (row < shi)
            e11.emplace_back(row-slo, col-slo, val_[j]);
          else {
            while (row_ptr<dim_upd && upd[row_ptr]<row)
              row_ptr++;
            if (row_ptr == dim_upd) break;
            if (upd[row_ptr] == row)
              e21.emplace_back(row_ptr, col-slo, val_[j]);
          }
        }
      }
    }
    for (integer_t i=0; i<dim_upd; ++i) { // update columns
      //while (c < local_cols_ && global_col_[c] < upd[i]) c++;
      c = find_global(upd[i], c);
      if (c == local_cols_ || global_col_[c] != upd[i]) continue;
      for (integer_t j=ptr_[c]; j<ptr_[c+1]; j++) {
        auto row = ind_[j];
        if (row >= slo) {
          if (row < shi)
            e12.emplace_back(row-slo, i, val_[j]);
          else break;
        }
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  PropMapSparseMatrix<scalar_t,integer_t>::set_front_elements
  (integer_t slo, integer_t shi, const std::vector<integer_t>& upd,
   Triplet<scalar_t>* e11, Triplet<scalar_t>* e12,
   Triplet<scalar_t>* e21) const {
    integer_t dim_upd = upd.size();
    auto c = find_global(slo);
    auto chi = find_global(shi, c);
    for (; c<chi; c++) {
      auto col = global_col_[c];
      integer_t row_ptr = 0;
      auto hij = ptr_[c+1];
      for (integer_t j=ptr_[c]; j<hij; j++) {
        auto row = ind_[j];
        if (row >= slo) {
          if (row < shi)
            *e11++ = Triplet<scalar_t>(row-slo, col-slo, val_[j]);
          else {
            while (row_ptr<dim_upd && upd[row_ptr]<row)
              row_ptr++;
            if (row_ptr == dim_upd) break;
            if (upd[row_ptr] == row)
              *e21++ = Triplet<scalar_t>(row_ptr, col-slo, val_[j]);
          }
        }
      }
    }
    for (integer_t i=0; i<dim_upd; ++i) { // update columns
      //while (c < local_cols_ && global_col_[c] < upd[i]) c++;
      c = find_global(upd[i], c);
      if (c == local_cols_ || global_col_[c] != upd[i]) continue;
      for (integer_t j=ptr_[c]; j<ptr_[c+1]; j++) {
        auto row = ind_[j];
        if (row >= slo) {
          if (row < shi)
            *e12++ = Triplet<scalar_t>(row-slo, i, val_[j]);
          else break;
        }
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  PropMapSparseMatrix<scalar_t,integer_t>::count_front_elements
  (integer_t slo, integer_t shi, const std::vector<integer_t>& upd,
   std::size_t& e11, std::size_t& e12, std::size_t& e21) const {
    integer_t dim_upd = upd.size();
    auto c = find_global(slo);
    auto chi = find_global(shi, c);
    for (; c<chi; c++) {
      integer_t row_ptr = 0;
      auto hij = ptr_[c+1];
      for (integer_t j=ptr_[c]; j<hij; j++) {
        auto row = ind_[j];
        if (row >= slo) {
          if (row < shi) e11++;
          else {
            while (row_ptr<dim_upd && upd[row_ptr]<row)
              row_ptr++;
            if (row_ptr == dim_upd) break;
            if (upd[row_ptr] == row) e21++;
          }
        }
      }
    }
    for (integer_t i=0; i<dim_upd; ++i) { // update columns
      //while (c < local_cols_ && global_col_[c] < upd[i]) c++;
      c = find_global(upd[i], c);
      if (c == local_cols_ || global_col_[c] != upd[i]) continue;
      for (integer_t j=ptr_[c]; j<ptr_[c+1]; j++) {
        auto row = ind_[j];
        if (row >= slo) {
          if (row < shi) e12++;
          else break;
        }
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  PropMapSparseMatrix<scalar_t,integer_t>::extract_F11_block
  (scalar_t* F, integer_t ldF, integer_t row, integer_t nr_rows,
   integer_t col, integer_t nr_cols) const {
    auto c = find_global(col);
    auto chi = find_global(col+nr_cols, c);
    for (; c<chi; c++) {
      for (integer_t j=ptr_[c]; j<ptr_[c+1]; j++) {
        auto r = ind_[j];
        if (r >= row) {
          if (r < row + nr_rows)
            F[r-row + (global_col_[c]-col)*ldF] = val_[j];
          else break;
        }
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  PropMapSparseMatrix<scalar_t,integer_t>::extract_F12_block
  (scalar_t* F, integer_t ldF, integer_t row, integer_t nr_rows,
   integer_t col, integer_t nr_cols, const integer_t* upd) const {
    if (nr_cols == 0 || nr_rows == 0) return;
    auto c = find_global(upd[0]);
    for (integer_t i=0; i<nr_cols; i++) {
      //while (c < local_cols_ && global_col_[c] < upd[i]) c++;
      if (i > 0)
        c = find_global(upd[i], c);
      if (c == local_cols_ || global_col_[c] != upd[i]) continue;
      for (integer_t j=ptr_[c]; j<ptr_[c+1]; j++) {
        auto r = ind_[j];
        if (r >= row) {
          if (r < row+nr_rows) F[r-row + i*ldF] = val_[j];
          else break;
        }
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  PropMapSparseMatrix<scalar_t,integer_t>::extract_F21_block
  (scalar_t* F, integer_t ldF, integer_t row, integer_t nr_rows,
   integer_t col, integer_t nr_cols, const integer_t* upd) const{
    if (nr_rows == 0 || nr_cols == 0) return;
    auto c = find_global(col);
    auto chi = find_global(col+nr_cols, c);
    for (; c<chi; c++) {
      integer_t row_upd = 0;
      for (integer_t j=ptr_[c]; j<ptr_[c+1]; j++) {
        auto r = ind_[j];
        while (row_upd<nr_rows && upd[row_upd]<r) row_upd++;
        if (row_upd == nr_rows) break;
        if (upd[row_upd] == r)
          F[row_upd + (global_col_[c]-col)*ldF] = val_[j];
      }
    }
  }

  /** mat should be defined on the same communicator as F11 */
  template<typename scalar_t,typename integer_t> void
  PropMapSparseMatrix<scalar_t,integer_t>::extract_separator_2d
  (integer_t shi, const std::vector<std::size_t>& I,
   const std::vector<std::size_t>& J, DistM_t& B) const {
    if (!B.active()) return;
    integer_t m = B.rows(), n = B.cols();
    if (m == 0 || n == 0) return;
    B.zero();
    for (integer_t j=0; j<n; j++) {
      integer_t jj = J[j];
      integer_t c = find_global(jj);
      if (c == local_cols_ || global_col_[c] != jj) continue;
      for (integer_t i=0; i<m; i++) {
        integer_t ii = I[i];
        if (jj >= shi && ii >= shi) break;
        for (integer_t k=ptr_[c]; k<ptr_[c+1]; k++) {
          if (ind_[k] == ii) {
            B.global(i, j, val_[k]);
            break;
          }
        }
      }
    }
  }


  /**
   * Multiply the front F = [ A(I^{sep},I^{sep})  A(I^{sep},I^{upd}) ]
   *                        [ A(I^{upd},I^{sep})  0                  ]
   * with the random matrix R. Compute both Srow = F R and Scol = F^T R.
   *
   * R, Srow and Scol are distributed in 2D block cyclic format, on a
   * p_rows x p_cols grid. There is only communication within columns
   * of the processor grid.
   *
   * The algorithm is as follows:
   * There are 3 loops over all the nonzeros in F.
   *
   *  # first loop to count what needs to be communicated
   *  - for all nnz fij in F do
   *  -     if I own row j of R
   *  -         I will need to send row j of R to the owner of row i in the same processor column as me
   *  -     if I own row i of R  # for the transposed multiply
   *  -         I will need to send row i of R to the owner of row j in the same processor column as me
   *  -     if I own row i of Srow
   *  -         I will need to receive aij times row j of R from the owner of row j of R in the same processor column as me
   *  -     if I own row j of Scol
   *  -         I will need to receive aji times row i of R from the owner of row i of R in the same processor column as me
   *
   *  # second loop to do the actual multiplication and copy results to send buffers
   *  - for all nnz fij in F do
   *  -     if I own row j of R
   *  -         multiply row j of R with aij and copy the result in sendbuffer for the owner of row i in the same processor column as me
   *  -     if I own row i of R  # for the transposed multiply
   *  -         multiply row i of R with aji and copy the result in sendbuffer for the owner of row j in the same processor column as me
   *
   *  # communicate with all the other processes in the same processor column
   *  - for all p in my processor column: MPI_Isend()
   *  - for all p in my processor column: MPI_Irecv()
   *
   *  - wait for all incoming messages
   *  # final loop to receive data and add it to Srow and Scol
   *  - for all nnz fij in F do
   *  -     if I own row i of Srow
   *  -         copy aij times row j of R from the recvbuffer of the owner of row j of R in the same processor column as me to row i in Srow
   *  -     if I own row j of Scol
   *  -         copy aji times row i of R from the recvbuffer of the owner of row i of R in the same processor column as me to row j in Scol
   *
   *  - wait for all outgoing messages to complete
   */
  template<typename scalar_t,typename integer_t> void
  PropMapSparseMatrix<scalar_t,integer_t>::front_multiply_2d
  (integer_t slo, integer_t shi, const std::vector<integer_t>& upd,
   const DistM_t& R, DistM_t& Srow, DistM_t& Scol, int depth) const {
    assert(R.fixed());
    assert(Srow.fixed());
    assert(Scol.fixed());
    if (!R.active()) return;
    const integer_t dim_upd = upd.size();
    long long int local_flops = 0;
    const auto dim_sep = shi - slo;
    const auto cols = R.cols();
    const auto lcols = R.lcols();
    const auto p_rows = R.nprows();
    const auto p_row  = R.prow();
    auto rows_to = new integer_t[2*p_rows];
    auto rows_from = rows_to + p_rows;
    std::fill(rows_to, rows_to+2*p_rows, 0);

    auto clo = find_global(slo);
    auto chi = find_global(shi);

    for (integer_t c=clo; c<chi; c++) { // separator columns
      auto row_j_rank = R.rowg2p_fixed(global_col_[c] - slo);
      integer_t row_upd = 0;
      auto hij = ptr_[c+1];
      for (integer_t j=ptr_[c]; j<hij; j++) {
        auto row = ind_[j];
        if (row >= slo) {
          if (row < shi) {
            auto row_i_rank = R.rowg2p_fixed(row - slo);
            if (row_j_rank == p_row) {
              rows_to[row_i_rank]++;
              rows_from[row_i_rank]++;
            }
            if (row_i_rank == p_row) { // transpose
              rows_to[row_j_rank]++;
              rows_from[row_j_rank]++;
            }
            local_flops += 4 * cols;
          } else {
            while (row_upd < dim_upd && upd[row_upd] < row) row_upd++;
            if (row_upd == dim_upd) break;
            if (upd[row_upd] == row) {
              auto row_i_rank = R.rowg2p_fixed(dim_sep + row_upd);
              if (row_j_rank == p_row) {
                rows_to[row_i_rank]++;
                rows_from[row_i_rank]++;
              }
              if (row_i_rank == p_row) { // transpose
                rows_to[row_j_rank]++;
                rows_from[row_j_rank]++;
              }
              local_flops += 4 * cols;
            }
          }
        }
      }
    }
    for (integer_t i=0, c=chi; i<dim_upd; i++) { // update columns
      c = find_global(upd[i], c);
      if (c == local_cols_ || global_col_[c] != upd[i]) continue;
      auto row_j_rank = R.rowg2p_fixed(dim_sep + i);
      auto hij = ptr_[c+1];
      for (integer_t j=ptr_[c]; j<hij; j++) {
        integer_t row = ind_[j];
        if (row >= slo) {
          if (row < shi) {
            auto row_i_rank = R.rowg2p_fixed(row - slo);
            if (row_j_rank == p_row) {
              rows_to[row_i_rank]++;
              rows_from[row_i_rank]++;
            }
            if (row_i_rank == p_row) { // transpose
              rows_to[row_j_rank]++;
              rows_from[row_j_rank]++;
            }
            local_flops += 4 * cols;
          } else break;
        }
      }
    }
    if (R.is_master()) {
      STRUMPACK_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
      STRUMPACK_SPARSE_SAMPLE_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
    }

    std::size_t ssize = std::accumulate(rows_to, rows_to+p_rows, 0);
    std::size_t rsize = std::accumulate(rows_from, rows_from+p_rows, 0);
    auto sbuf = new scalar_t[(ssize+rsize)*lcols];
    auto rbuf = sbuf+ssize*lcols;
    std::fill(sbuf, sbuf+ssize*lcols, scalar_t(0.));
    auto pp = new scalar_t*[p_rows];
    pp[0] = sbuf;
    for (integer_t p=1; p<p_rows; p++)
      pp[p] = pp[p-1] + rows_to[p-1]*lcols;

#pragma omp parallel for
    for (int t=0; t<p_rows; t++) {
      for (integer_t c=clo; c<chi; c++) { // separator columns
        auto Aj = global_col_[c] - slo;
        auto row_j_rank = R.rowg2p_fixed(Aj);
        integer_t row_upd = 0;
        const auto hij = ptr_[c+1];
        for (integer_t j=ptr_[c]; j<hij; j++) {
          auto row = ind_[j];
          if (row >= slo) {
            if (row < shi) {
              auto Ai = row - slo;
              auto row_i_rank = R.rowg2p_fixed(Ai);
              if (row_i_rank == t && row_j_rank == p_row) {
                const auto a = val_[j];
                for (integer_t k=0, r=R.rowg2l_fixed(Aj); k<lcols; k++)
                  pp[row_i_rank][k] += a * R(r, k);
                pp[row_i_rank] += lcols;
              }
              if (row_j_rank == t && row_i_rank == p_row) { // transpose
                const auto a = blas::my_conj(val_[j]);
                for (integer_t k=0, r=R.rowg2l_fixed(Ai); k<lcols; k++)
                  pp[row_j_rank][k] += a * R(r, k);
                pp[row_j_rank] += lcols;
              }
            } else {
              while (row_upd < dim_upd && upd[row_upd] < row)
                row_upd++;
              if (row_upd == dim_upd) break;
              if (upd[row_upd] == row) {
                auto Ai = dim_sep + row_upd;
                auto row_i_rank = R.rowg2p_fixed(Ai);
                if (row_i_rank == t && row_j_rank == p_row) {
                  const auto a = val_[j];
                  for (integer_t k=0, r=R.rowg2l_fixed(Aj); k<lcols; k++)
                    pp[row_i_rank][k] += a * R(r, k);
                  pp[row_i_rank] += lcols;
                }
                if (row_j_rank == t && row_i_rank == p_row) { // transpose
                  const auto a = blas::my_conj(val_[j]);
                  for (integer_t k=0, r=R.rowg2l_fixed(Ai); k<lcols; k++)
                    pp[row_j_rank][k] += a * R(r, k);
                  pp[row_j_rank] += lcols;
                }
              }
            }
          }
        }
      }
    }
#pragma omp parallel for
    for (int t=0; t<p_rows; t++) {
      for (integer_t i=0, c=chi; i<dim_upd; i++) { // update columns
        c = find_global(upd[i], c);
        if (c == local_cols_ || global_col_[c] != upd[i]) continue;
        auto Aj = dim_sep + i;
        auto row_j_rank = R.rowg2p_fixed(Aj);
        auto hij = ptr_[c+1];
        for (integer_t j=ptr_[c]; j<hij; j++) {
          integer_t row = ind_[j];
          if (row >= slo) {
            if (row < shi) {
              const auto a = blas::my_conj(val_[j]);
              auto Ai = row - slo;
              auto row_i_rank = R.rowg2p_fixed(Ai);
              if (row_i_rank == t && row_j_rank == p_row) {
                for (integer_t k=0, r=R.rowg2l_fixed(Aj); k<lcols; k++)
                  pp[row_i_rank][k] += a * R(r, k);
                pp[row_i_rank] += lcols;
              }
              if (row_j_rank == t && row_i_rank == p_row) { // transpose
                for (integer_t k=0, r=R.rowg2l_fixed(Ai); k<lcols; k++)
                  pp[row_j_rank][k] += a * R(r, k);
                pp[row_j_rank] += lcols;
              }
            } else break;
          }
        }
      }
    }

    // TODO instead of a send/receive loop, get the column
    // communicator from BLACS?

    pp[0] = sbuf;
    for (integer_t p=1; p<p_rows; p++)
      pp[p] = pp[p-1] + rows_to[p-1]*lcols;
    auto p_col  = R.pcol();
    auto sreq = new MPI_Request[p_rows*2];
    auto rreq = sreq + p_rows;
    for (integer_t p=0; p<p_rows; p++)
      MPI_Isend(pp[p], rows_to[p]*lcols, mpi_type<scalar_t>(),
                p+p_col*p_rows, 0, R.comm(), sreq+p);

    pp[0] = rbuf;
    for (integer_t p=1; p<p_rows; p++)
      pp[p] = pp[p-1] + rows_from[p-1]*lcols;
    for (integer_t p=0; p<p_rows; p++)
      MPI_Irecv(pp[p], rows_from[p]*lcols, mpi_type<scalar_t>(),
                p+p_col*p_rows, 0, R.comm(), rreq+p);

    // wait for all incoming messages
    MPI_Waitall(p_rows, rreq, MPI_STATUSES_IGNORE);
    for (integer_t c=clo; c<chi; c++) { // separator columns
      auto Aj = global_col_[c] - slo;
      auto row_j_rank = R.rowg2p_fixed(Aj);
      integer_t row_upd = 0;
      auto hij = ptr_[c+1];
      for (integer_t j=ptr_[c]; j<hij; j++) {
        auto row = ind_[j];
        if (row >= slo) {
          if (row < shi) {
            auto Ai = row - slo;
            auto row_i_rank = R.rowg2p_fixed(Ai);
            if (row_i_rank == p_row) {
              for (integer_t k=0, r=Srow.rowg2l_fixed(Ai); k<lcols; k++)
                Srow(r, k) += pp[row_j_rank][k];
              pp[row_j_rank] += lcols;
            }
            if (row_j_rank == p_row) { // transpose
              for (integer_t k=0, r=Srow.rowg2l_fixed(Aj); k<lcols; k++)
                Scol(r, k) += pp[row_i_rank][k];
              pp[row_i_rank] += lcols;
            }
          } else {
            while (row_upd < dim_upd && upd[row_upd] < row)
              row_upd++;
            if (row_upd == dim_upd) break;
            if (upd[row_upd] == row) {
              auto Ai = dim_sep + row_upd;
              auto row_i_rank = R.rowg2p_fixed(Ai);
              if (row_i_rank == p_row) {
                for (integer_t k=0, r=Srow.rowg2l_fixed(Ai); k<lcols; k++)
                  Srow(r, k) += pp[row_j_rank][k];
                pp[row_j_rank] += lcols;
              }
              if (row_j_rank == p_row) { // transpose
                for (integer_t k=0, r=Scol.rowg2l_fixed(Aj); k<lcols; k++)
                  Scol(r, k) += pp[row_i_rank][k];
                pp[row_i_rank] += lcols;
              }
            }
          }
        }
      }
    }
    for (integer_t i=0, c=chi; i<dim_upd; i++) { // update columns
      c = find_global(upd[i], c);
      if (c == local_cols_ || global_col_[c] != upd[i])
        continue;
      auto Aj = dim_sep + i;
      auto row_j_rank = R.rowg2p_fixed(Aj);
      auto hij = ptr_[c+1];
      for (integer_t j=ptr_[c]; j<hij; j++) {
        integer_t row = ind_[j];
        if (row >= slo) {
          if (row < shi) {
            auto Ai = row - slo;
            auto row_i_rank = R.rowg2p_fixed(Ai);
            if (row_i_rank == p_row) {
              for (integer_t k=0, r=Srow.rowg2l_fixed(Ai); k<lcols; k++)
                Srow(r, k) += pp[row_j_rank][k];
              pp[row_j_rank] += lcols;
            }
            if (row_j_rank == p_row) { // transpose
              for (integer_t k=0, r=Scol.rowg2l_fixed(Aj); k<lcols; k++)
                Scol(r, k) += pp[row_i_rank][k];
              pp[row_i_rank] += lcols;
            }
          } else break;
        }
      }
    }
    delete[] pp;
    delete[] rows_to;

    // wait for sends to finish
    MPI_Waitall(p_rows, sreq, MPI_STATUSES_IGNORE);
    delete[] sreq;
    delete[] sbuf;
  }

  template<typename scalar_t,typename integer_t> void
  PropMapSparseMatrix<scalar_t,integer_t>::
  front_multiply_2d_N
  (integer_t slo, integer_t shi, const std::vector<integer_t>& upd,
   const DistM_t& R, DistM_t& S, int depth) const {
    // DistM_t Sdummy(S.grid(), S.rows(), S.cols());
    // front_multiply_2d(slo, shi, upd, R, S, Sdummy, 0);
    // return;

    assert(R.fixed());
    assert(S.fixed());
    if (!R.active()) return;
    const integer_t dim_upd = upd.size();
    long long int local_flops = 0;
    const auto dim_sep = shi - slo;
    const auto cols = R.cols();
    const auto lcols = R.lcols();
    const auto p_rows = R.nprows();
    const auto p_row  = R.prow();
    auto rows_to = new integer_t[2*p_rows];
    auto rows_from = rows_to + p_rows;
    std::fill(rows_to, rows_to+2*p_rows, 0);

    auto clo = find_global(slo);
    auto chi = find_global(shi);

    for (integer_t c=clo; c<chi; c++) { // separator columns
      auto row_j_rank = R.rowg2p_fixed(global_col_[c] - slo);
      integer_t row_upd = 0;
      auto hij = ptr_[c+1];
      for (integer_t j=ptr_[c]; j<hij; j++) {
        auto row = ind_[j];
        if (row >= slo) {
          if (row < shi) {
            auto row_i_rank = R.rowg2p_fixed(row - slo);
            if (row_j_rank == p_row) {
              rows_to[row_i_rank]++;
              rows_from[row_i_rank]++;
            }
            local_flops += 2 * cols;
          } else {
            while (row_upd < dim_upd && upd[row_upd] < row) row_upd++;
            if (row_upd == dim_upd) break;
            if (upd[row_upd] == row) {
              auto row_i_rank = R.rowg2p_fixed(dim_sep + row_upd);
              if (row_j_rank == p_row) {
                rows_to[row_i_rank]++;
                rows_from[row_i_rank]++;
              }
              local_flops += 2 * cols;
            }
          }
        }
      }
    }
    for (integer_t i=0, c=chi; i<dim_upd; i++) { // update columns
      c = find_global(upd[i], c);
      if (c == local_cols_ || global_col_[c] != upd[i]) continue;
      auto row_j_rank = R.rowg2p_fixed(dim_sep + i);
      auto hij = ptr_[c+1];
      for (integer_t j=ptr_[c]; j<hij; j++) {
        integer_t row = ind_[j];
        if (row >= slo) {
          if (row < shi) {
            auto row_i_rank = R.rowg2p_fixed(row - slo);
            if (row_j_rank == p_row) {
              rows_to[row_i_rank]++;
              rows_from[row_i_rank]++;
            }
            local_flops += 2 * cols;
          } else break;
        }
      }
    }
    if (R.is_master()) {
      STRUMPACK_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
      STRUMPACK_SPARSE_SAMPLE_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
    }

    std::size_t ssize = std::accumulate(rows_to, rows_to+p_rows, 0);
    std::size_t rsize = std::accumulate(rows_from, rows_from+p_rows, 0);
    auto sbuf = new scalar_t[(ssize+rsize)*lcols];
    auto rbuf = sbuf+ssize*lcols;
    std::fill(sbuf, sbuf+ssize*lcols, scalar_t(0.));
    auto pp = new scalar_t*[p_rows];
    pp[0] = sbuf;
    for (integer_t p=1; p<p_rows; p++)
      pp[p] = pp[p-1] + rows_to[p-1]*lcols;

#pragma omp parallel for
    for (int t=0; t<p_rows; t++) {
      for (integer_t c=clo; c<chi; c++) { // separator columns
        auto Aj = global_col_[c] - slo;
        auto row_j_rank = R.rowg2p_fixed(Aj);
        integer_t row_upd = 0;
        const auto hij = ptr_[c+1];
        for (integer_t j=ptr_[c]; j<hij; j++) {
          auto row = ind_[j];
          if (row >= slo) {
            if (row < shi) {
              auto Ai = row - slo;
              auto row_i_rank = R.rowg2p_fixed(Ai);
              if (row_i_rank == t && row_j_rank == p_row) {
                const auto a = val_[j];
                for (integer_t k=0, r=R.rowg2l_fixed(Aj); k<lcols; k++)
                  pp[row_i_rank][k] += a * R(r, k);
                pp[row_i_rank] += lcols;
              }
            } else {
              while (row_upd < dim_upd && upd[row_upd] < row)
                row_upd++;
              if (row_upd == dim_upd) break;
              if (upd[row_upd] == row) {
                auto Ai = dim_sep + row_upd;
                auto row_i_rank = R.rowg2p_fixed(Ai);
                if (row_i_rank == t && row_j_rank == p_row) {
                  const auto a = val_[j];
                  for (integer_t k=0, r=R.rowg2l_fixed(Aj); k<lcols; k++)
                    pp[row_i_rank][k] += a * R(r, k);
                  pp[row_i_rank] += lcols;
                }
              }
            }
          }
        }
      }
    }
#pragma omp parallel for
    for (int t=0; t<p_rows; t++) {
      for (integer_t i=0, c=chi; i<dim_upd; i++) { // update columns
        c = find_global(upd[i], c);
        if (c == local_cols_ || global_col_[c] != upd[i]) continue;
        auto Aj = dim_sep + i;
        auto row_j_rank = R.rowg2p_fixed(Aj);
        auto hij = ptr_[c+1];
        for (integer_t j=ptr_[c]; j<hij; j++) {
          integer_t row = ind_[j];
          if (row >= slo) {
            if (row < shi) {
              auto a = val_[j];
              auto Ai = row - slo;
              auto row_i_rank = R.rowg2p_fixed(Ai);
              if (row_i_rank == t && row_j_rank == p_row) {
                for (integer_t k=0, r=R.rowg2l_fixed(Aj); k<lcols; k++)
                  pp[row_i_rank][k] += a * R(r, k);
                pp[row_i_rank] += lcols;
              }
            } else break;
          }
        }
      }
    }

    // TODO instead of a send/receive loop, get the column
    // communicator from BLACS?

    pp[0] = sbuf;
    for (integer_t p=1; p<p_rows; p++)
      pp[p] = pp[p-1] + rows_to[p-1]*lcols;
    auto p_col  = R.pcol();
    auto sreq = new MPI_Request[p_rows*2];
    auto rreq = sreq + p_rows;
    for (integer_t p=0; p<p_rows; p++)
      MPI_Isend(pp[p], rows_to[p]*lcols, mpi_type<scalar_t>(),
                p+p_col*p_rows, 0, R.comm(), sreq+p);

    pp[0] = rbuf;
    for (integer_t p=1; p<p_rows; p++)
      pp[p] = pp[p-1] + rows_from[p-1]*lcols;
    for (integer_t p=0; p<p_rows; p++)
      MPI_Irecv(pp[p], rows_from[p]*lcols, mpi_type<scalar_t>(),
                p+p_col*p_rows, 0, R.comm(), rreq+p);

    // wait for all incoming messages
    MPI_Waitall(p_rows, rreq, MPI_STATUSES_IGNORE);
    for (integer_t c=clo; c<chi; c++) { // separator columns
      auto Aj = global_col_[c] - slo;
      auto row_j_rank = R.rowg2p_fixed(Aj);
      integer_t row_upd = 0;
      auto hij = ptr_[c+1];
      for (integer_t j=ptr_[c]; j<hij; j++) {
        auto row = ind_[j];
        if (row >= slo) {
          if (row < shi) {
            auto Ai = row - slo;
            auto row_i_rank = R.rowg2p_fixed(Ai);
            if (row_i_rank == p_row) {
              for (integer_t k=0, r=S.rowg2l_fixed(Ai); k<lcols; k++)
                S(r, k) += pp[row_j_rank][k];
              pp[row_j_rank] += lcols;
            }
          } else {
            while (row_upd < dim_upd && upd[row_upd] < row)
              row_upd++;
            if (row_upd == dim_upd) break;
            if (upd[row_upd] == row) {
              auto Ai = dim_sep + row_upd;
              auto row_i_rank = R.rowg2p_fixed(Ai);
              if (row_i_rank == p_row) {
                for (integer_t k=0, r=S.rowg2l_fixed(Ai); k<lcols; k++)
                  S(r, k) += pp[row_j_rank][k];
                pp[row_j_rank] += lcols;
              }
            }
          }
        }
      }
    }
    for (integer_t i=0, c=chi; i<dim_upd; i++) { // update columns
      c = find_global(upd[i], c);
      if (c == local_cols_ || global_col_[c] != upd[i])
        continue;
      auto Aj = dim_sep + i;
      auto row_j_rank = R.rowg2p_fixed(Aj);
      auto hij = ptr_[c+1];
      for (integer_t j=ptr_[c]; j<hij; j++) {
        integer_t row = ind_[j];
        if (row >= slo) {
          if (row < shi) {
            auto Ai = row - slo;
            auto row_i_rank = R.rowg2p_fixed(Ai);
            if (row_i_rank == p_row) {
              for (integer_t k=0, r=S.rowg2l_fixed(Ai); k<lcols; k++)
                S(r, k) += pp[row_j_rank][k];
              pp[row_j_rank] += lcols;
            }
          } else break;
        }
      }
    }
    delete[] pp;
    delete[] rows_to;

    // wait for sends to finish
    MPI_Waitall(p_rows, sreq, MPI_STATUSES_IGNORE);
    delete[] sreq;
    delete[] sbuf;
  }

  template<typename scalar_t,typename integer_t> void
  PropMapSparseMatrix<scalar_t,integer_t>::front_multiply_2d_TC
  (integer_t slo, integer_t shi, const std::vector<integer_t>& upd,
   const DistM_t& R, DistM_t& S, int depth) const {
    // DistM_t Sdummy(S.grid(), S.rows(), S.cols());
    // front_multiply_2d(slo, shi, upd, R, Sdummy, S, 0);
    // return;

    assert(R.fixed());
    assert(S.fixed());
    if (!R.active()) return;
    const integer_t dim_upd = upd.size();
    long long int local_flops = 0;
    const auto dim_sep = shi - slo;
    const auto cols = R.cols();
    const auto lcols = R.lcols();
    const auto p_rows = R.nprows();
    const auto p_row  = R.prow();
    auto rows_to = new integer_t[2*p_rows];
    auto rows_from = rows_to + p_rows;
    std::fill(rows_to, rows_to+2*p_rows, 0);

    auto clo = find_global(slo);
    auto chi = find_global(shi);

    for (integer_t c=clo; c<chi; c++) { // separator columns
      auto row_j_rank = R.rowg2p_fixed(global_col_[c] - slo);
      integer_t row_upd = 0;
      auto hij = ptr_[c+1];
      for (integer_t j=ptr_[c]; j<hij; j++) {
        auto row = ind_[j];
        if (row >= slo) {
          if (row < shi) {
            if (R.rowg2p_fixed(row - slo) == p_row) { // transpose
              rows_to[row_j_rank]++;
              rows_from[row_j_rank]++;
            }
            local_flops += 2 * cols;
          } else {
            while (row_upd < dim_upd && upd[row_upd] < row) row_upd++;
            if (row_upd == dim_upd) break;
            if (upd[row_upd] == row) {
              if (R.rowg2p_fixed(dim_sep + row_upd) == p_row) { // transpose
                rows_to[row_j_rank]++;
                rows_from[row_j_rank]++;
              }
              local_flops += 2 * cols;
            }
          }
        }
      }
    }
    for (integer_t i=0, c=chi; i<dim_upd; i++) { // update columns
      c = find_global(upd[i], c);
      if (c == local_cols_ || global_col_[c] != upd[i]) continue;
      auto row_j_rank = R.rowg2p_fixed(dim_sep + i);
      auto hij = ptr_[c+1];
      for (integer_t j=ptr_[c]; j<hij; j++) {
        integer_t row = ind_[j];
        if (row >= slo) {
          if (row < shi) {
            if (R.rowg2p_fixed(row - slo) == p_row) { // transpose
              rows_to[row_j_rank]++;
              rows_from[row_j_rank]++;
            }
            local_flops += 2 * cols;
          } else break;
        }
      }
    }
    if (R.is_master()) {
      STRUMPACK_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
      STRUMPACK_SPARSE_SAMPLE_FLOPS((is_complex<scalar_t>() ? 4 : 1) * local_flops);
    }

    std::size_t ssize = std::accumulate(rows_to, rows_to+p_rows, 0);
    std::size_t rsize = std::accumulate(rows_from, rows_from+p_rows, 0);
    auto sbuf = new scalar_t[(ssize+rsize)*lcols];
    auto rbuf = sbuf+ssize*lcols;
    std::fill(sbuf, sbuf+ssize*lcols, scalar_t(0.));
    auto pp = new scalar_t*[p_rows];
    pp[0] = sbuf;
    for (integer_t p=1; p<p_rows; p++)
      pp[p] = pp[p-1] + rows_to[p-1]*lcols;

#pragma omp parallel for
    for (int t=0; t<p_rows; t++) {
      for (integer_t c=clo; c<chi; c++) { // separator columns
        auto Aj = global_col_[c] - slo;
        auto row_j_rank = R.rowg2p_fixed(Aj);
        integer_t row_upd = 0;
        const auto hij = ptr_[c+1];
        for (integer_t j=ptr_[c]; j<hij; j++) {
          auto row = ind_[j];
          if (row >= slo) {
            if (row < shi) {
              auto Ai = row - slo;
              auto row_i_rank = R.rowg2p_fixed(Ai);
              if (row_j_rank == t && row_i_rank == p_row) { // transpose
                const auto a = blas::my_conj(val_[j]);
                for (integer_t k=0, r=R.rowg2l_fixed(Ai); k<lcols; k++)
                  pp[row_j_rank][k] += a * R(r, k);
                pp[row_j_rank] += lcols;
              }
            } else {
              while (row_upd < dim_upd && upd[row_upd] < row)
                row_upd++;
              if (row_upd == dim_upd) break;
              if (upd[row_upd] == row) {
                auto Ai = dim_sep + row_upd;
                auto row_i_rank = R.rowg2p_fixed(Ai);
                if (row_j_rank == t && row_i_rank == p_row) { // transpose
                  const auto a = blas::my_conj(val_[j]);
                  for (integer_t k=0, r=R.rowg2l_fixed(Ai); k<lcols; k++)
                    pp[row_j_rank][k] += a * R(r, k);
                  pp[row_j_rank] += lcols;
                }
              }
            }
          }
        }
      }
    }
#pragma omp parallel for
    for (int t=0; t<p_rows; t++) {
      for (integer_t i=0, c=chi; i<dim_upd; i++) { // update columns
        c = find_global(upd[i], c);
        if (c == local_cols_ || global_col_[c] != upd[i]) continue;
        auto Aj = dim_sep + i;
        auto row_j_rank = R.rowg2p_fixed(Aj);
        auto hij = ptr_[c+1];
        for (integer_t j=ptr_[c]; j<hij; j++) {
          integer_t row = ind_[j];
          if (row >= slo) {
            if (row < shi) {
              const auto a = blas::my_conj(val_[j]);
              auto Ai = row - slo;
              auto row_i_rank = R.rowg2p_fixed(Ai);
              if (row_j_rank == t && row_i_rank == p_row) { // transpose
                for (integer_t k=0, r=R.rowg2l_fixed(Ai); k<lcols; k++)
                  pp[row_j_rank][k] += a * R(r, k);
                pp[row_j_rank] += lcols;
              }
            } else break;
          }
        }
      }
    }

    // TODO instead of a send/receive loop, get the column
    // communicator from BLACS?

    pp[0] = sbuf;
    for (integer_t p=1; p<p_rows; p++)
      pp[p] = pp[p-1] + rows_to[p-1]*lcols;
    auto p_col  = R.pcol();
    auto sreq = new MPI_Request[p_rows*2];
    auto rreq = sreq + p_rows;
    for (integer_t p=0; p<p_rows; p++)
      MPI_Isend(pp[p], rows_to[p]*lcols, mpi_type<scalar_t>(),
                p+p_col*p_rows, 0, R.comm(), sreq+p);

    pp[0] = rbuf;
    for (integer_t p=1; p<p_rows; p++)
      pp[p] = pp[p-1] + rows_from[p-1]*lcols;
    for (integer_t p=0; p<p_rows; p++)
      MPI_Irecv(pp[p], rows_from[p]*lcols, mpi_type<scalar_t>(),
                p+p_col*p_rows, 0, R.comm(), rreq+p);

    // wait for all incoming messages
    MPI_Waitall(p_rows, rreq, MPI_STATUSES_IGNORE);
    for (integer_t c=clo; c<chi; c++) { // separator columns
      auto Aj = global_col_[c] - slo;
      auto row_j_rank = R.rowg2p_fixed(Aj);
      integer_t row_upd = 0;
      auto hij = ptr_[c+1];
      for (integer_t j=ptr_[c]; j<hij; j++) {
        auto row = ind_[j];
        if (row >= slo) {
          if (row < shi) {
            auto Ai = row - slo;
            auto row_i_rank = R.rowg2p_fixed(Ai);
            if (row_j_rank == p_row) { // transpose
              for (integer_t k=0, r=S.rowg2l_fixed(Aj); k<lcols; k++)
                S(r, k) += pp[row_i_rank][k];
              pp[row_i_rank] += lcols;
            }
          } else {
            while (row_upd < dim_upd && upd[row_upd] < row)
              row_upd++;
            if (row_upd == dim_upd) break;
            if (upd[row_upd] == row) {
              auto Ai = dim_sep + row_upd;
              auto row_i_rank = R.rowg2p_fixed(Ai);
              if (row_j_rank == p_row) { // transpose
                for (integer_t k=0, r=S.rowg2l_fixed(Aj); k<lcols; k++)
                  S(r, k) += pp[row_i_rank][k];
                pp[row_i_rank] += lcols;
              }
            }
          }
        }
      }
    }
    for (integer_t i=0, c=chi; i<dim_upd; i++) { // update columns
      c = find_global(upd[i], c);
      if (c == local_cols_ || global_col_[c] != upd[i])
        continue;
      auto Aj = dim_sep + i;
      auto row_j_rank = R.rowg2p_fixed(Aj);
      auto hij = ptr_[c+1];
      for (integer_t j=ptr_[c]; j<hij; j++) {
        integer_t row = ind_[j];
        if (row >= slo) {
          if (row < shi) {
            auto Ai = row - slo;
            auto row_i_rank = R.rowg2p_fixed(Ai);
            if (row_j_rank == p_row) { // transpose
              for (integer_t k=0, r=S.rowg2l_fixed(Aj); k<lcols; k++)
                S(r, k) += pp[row_i_rank][k];
              pp[row_i_rank] += lcols;
            }
          } else break;
        }
      }
    }
    delete[] pp;
    delete[] rows_to;

    // wait for sends to finish
    MPI_Waitall(p_rows, sreq, MPI_STATUSES_IGNORE);
    delete[] sreq;
    delete[] sbuf;
  }

  template<typename scalar_t,typename integer_t> CSRGraph<integer_t>
  PropMapSparseMatrix<scalar_t,integer_t>::extract_graph
  (int ordering_level, integer_t lo, integer_t hi) const {
    const auto clo = find_global(lo);
    const auto chi = find_global(hi);
    const auto n = hi - lo;
    std::vector<integer_t> gptr, gind;
    gptr.reserve(n+1);
    gptr.push_back(0);
    for (integer_t c=clo; c<chi; c++) {
      gptr.push_back(gptr.back());
      const auto phij = ind_.data() + ptr_[c+1];
      for (auto pj=ind_.data()+ptr_[c]; pj!=phij; pj++) {
        auto j = *pj;
        const auto row = j - lo;
        if (row >= 0 && row < n) {
          gind.push_back(row);
          gptr.back()++;
        }
      }
    }
    return CSRGraph<integer_t>(std::move(gptr), std::move(gind));
  }

  template<typename scalar_t,typename integer_t> CSRGraph<integer_t>
  PropMapSparseMatrix<scalar_t,integer_t>::extract_graph_sep_CB
  (int ordering_level, integer_t lo, integer_t hi,
   const std::vector<integer_t>& upd) const {
    if (upd.empty() || hi == lo)
      return CSRGraph<integer_t>(hi-lo, 0);
    const auto clo = find_global(lo);
    const auto chi = find_global(hi);
    const integer_t n = hi - lo, dupd = upd.size();
    std::vector<integer_t> gptr, gind;
    gptr.reserve(n+1);
    gptr.push_back(0);
    for (integer_t c=clo; c<chi; c++) {
      gptr.push_back(gptr.back());
      integer_t rupd = 0;
      const auto phij = ind_.data() + ptr_[c+1];
      for (auto pj=ind_.data() + ptr_[c]; pj!=phij; pj++) {
        const auto r = *pj;
        while (rupd < dupd && upd[rupd] < r) rupd++;
        if (rupd == dupd) break;
        if (upd[rupd] == r) {
          gind.push_back(rupd);
          gptr.back()++;
        }
      }
    }
    return CSRGraph<integer_t>(std::move(gptr), std::move(gind));
  }

  template<typename scalar_t,typename integer_t> CSRGraph<integer_t>
  PropMapSparseMatrix<scalar_t,integer_t>::extract_graph_CB_sep
  (int ordering_level, integer_t lo, integer_t hi,
   const std::vector<integer_t>& upd) const {
    if (upd.empty() || hi == lo)
      return CSRGraph<integer_t>(upd.size(), 0);
    integer_t dupd = upd.size(), dsep = hi - lo;
    std::vector<integer_t> gptr, gind;
    gptr.reserve(dupd+1);
    gptr.push_back(0);
    integer_t c = 0;
    for (integer_t i=0; i<dupd; i++) {
      gptr.push_back(gptr.back());
      c = find_global(upd[i], c);
      auto phij = ind_.data() + ptr_[c+1];
      for (auto pj=ind_.data()+ptr_[c]; pj!=phij; pj++) {
        auto r = *pj - lo;
        if (r >= 0 && r < dsep) {
          gind.push_back(r);
          gptr.back()++;
        }
      }
    }
    return CSRGraph<integer_t>(std::move(gptr), std::move(gind));
  }

  template<typename scalar_t,typename integer_t> CSRGraph<integer_t>
  PropMapSparseMatrix<scalar_t,integer_t>::extract_graph_CB
  (int ordering_level, const std::vector<integer_t>& upd) const {
    if (upd.empty()) return CSRGraph<integer_t>(upd.size(), 0);
    integer_t dupd = upd.size();
    std::vector<integer_t> gptr, gind;
    gptr.reserve(dupd+1);
    gptr.push_back(0);
    integer_t c = 0;
    for (integer_t i=0; i<dupd; i++) {
      gptr.push_back(gptr.back());
      c = find_global(upd[i], c);
      integer_t rupd = 0;
      auto phij = ind_.data() + ptr_[c+1];
      for (auto pj=ind_.data()+ptr_[c]; pj!=phij; pj++) {
        auto r = *pj;
        while (rupd < dupd && upd[rupd] < r) rupd++;
        if (rupd == dupd) break;
        if (upd[rupd] == r) {
          gind.push_back(rupd);
          gptr.back()++;
        }
      }
    }
    return CSRGraph<integer_t>(std::move(gptr), std::move(gind));
  }

  template<typename scalar_t,typename integer_t> void
  PropMapSparseMatrix<scalar_t,integer_t>::permute
  (const integer_t* iorder, const integer_t* order) {
    // don't actually permute, the matrix will be destroyed and build
    // again with the new permutation in the EliminationTreeMPIDist
    // class
  }

  // explicit template instantiations
  template class PropMapSparseMatrix<float,int>;
  template class PropMapSparseMatrix<double,int>;
  template class PropMapSparseMatrix<std::complex<float>,int>;
  template class PropMapSparseMatrix<std::complex<double>,int>;

  template class PropMapSparseMatrix<float,long int>;
  template class PropMapSparseMatrix<double,long int>;
  template class PropMapSparseMatrix<std::complex<float>,long int>;
  template class PropMapSparseMatrix<std::complex<double>,long int>;

  template class PropMapSparseMatrix<float,long long int>;
  template class PropMapSparseMatrix<double,long long int>;
  template class PropMapSparseMatrix<std::complex<float>,long long int>;
  template class PropMapSparseMatrix<std::complex<double>,long long int>;

} // end namespace strumpack
