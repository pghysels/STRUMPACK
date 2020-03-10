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
#include <vector>
#include <algorithm>
#include <tuple>
#include <cstdio>
#include <cstring>

#include "CompressedSparseMatrix.hpp"
#include "misc/Tools.hpp"
#include "CSRGraph.hpp"
#include "StrumpackConfig.hpp"
#include "dense/DenseMatrix.hpp"
#if defined(STRUMPACK_USE_MPI)
#include "dense/DistributedMatrix.hpp"
#endif


namespace strumpack {

  template<typename scalar_t,typename integer_t>
  CompressedSparseMatrix<scalar_t,integer_t>::CompressedSparseMatrix()
    : n_(0), nnz_(0), symm_sparse_(false) { }

  template<typename scalar_t,typename integer_t>
  CompressedSparseMatrix<scalar_t,integer_t>::CompressedSparseMatrix
  (integer_t n, integer_t nnz, bool symm_sparse)
    : n_(n), nnz_(nnz), symm_sparse_(symm_sparse) {
    ptr_.resize(n_+1);
    ind_.resize(nnz_);
    val_.resize(nnz_);
    ptr_[0] = 0;
    ptr_[n_] = nnz_;
  }

  template<typename scalar_t,typename integer_t>
  CompressedSparseMatrix<scalar_t,integer_t>::CompressedSparseMatrix
  (integer_t n, const integer_t* row_ptr, const integer_t* col_ind,
   const scalar_t* values, bool symm_sparsity)
    : n_(n), nnz_(row_ptr[n_]-row_ptr[0]), ptr_(n_+1),
      ind_(nnz_), val_(nnz_), symm_sparse_(symm_sparsity) {
    if (row_ptr)
      std::copy(row_ptr, row_ptr+n_+1, ptr());
    if (col_ind)
      std::copy(col_ind, col_ind+nnz_, ind());
    if (values)
      std::copy(values, values+nnz_, val());
  }

  template<typename scalar_t,typename integer_t> void
  CompressedSparseMatrix<scalar_t,integer_t>::print() const {
    std::cout << "size: " << size() << std::endl;
    std::cout << "nnz: " << nnz() << std::endl;
    std::cout << "ptr: " << std::endl << "\t";
    for (integer_t i=0; i<=size(); i++)
      std::cout << ptr_[i] << " ";
    std::cout << std::endl << "ind: ";
    for (integer_t i=0; i<nnz(); i++)
      std::cout << ind_[i] << " ";
    std::cout << std::endl << "val: ";
    for (integer_t i=0; i<nnz(); i++)
      std::cout << val_[i] << " ";
    std::cout << std::endl;
  }

  template<typename scalar_t,typename integer_t> int
  CompressedSparseMatrix<scalar_t,integer_t>::permute_and_scale
  (MatchingJob job, std::vector<integer_t>& perm, std::vector<scalar_t>& Dr,
   std::vector<scalar_t>& Dc, bool apply) {
    if (job == MatchingJob::NONE) return 1;
    if (job == MatchingJob::COMBBLAS) {
      std::cerr << "# ERROR: CombBLAS matching only supported in parallel."
                << std::endl;
      return 1;
    }
    perm.resize(n_);
    int_t liw = 0;
    switch (job) {
    case MatchingJob::MAX_SMALLEST_DIAGONAL: liw = 4*n_; break;
    case MatchingJob::MAX_SMALLEST_DIAGONAL_2: liw = 10*n_ + nnz_; break;
    case MatchingJob::MAX_CARDINALITY:
    case MatchingJob::MAX_DIAGONAL_SUM:
    case MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING:
    default: liw = 5*n_;
    }
    auto iw_ = std::unique_ptr<int_t[]>(new int_t[liw]);
    auto iw = iw_.get();
    int_t ldw = 0;
    switch (job) {
    case MatchingJob::MAX_CARDINALITY: ldw = 0; break;
    case MatchingJob::MAX_SMALLEST_DIAGONAL: ldw = n_; break;
    case MatchingJob::MAX_SMALLEST_DIAGONAL_2: ldw = nnz_; break;
    case MatchingJob::MAX_DIAGONAL_SUM: ldw = 2*n_ + nnz_; break;
    case MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING:
    default: ldw = 3*n_ + nnz_; break;
    }
    auto dw_ = std::unique_ptr<double[]>(new double[ldw]);
    auto dw = dw_.get();
    int_t icntl[10], info[10];
    int_t num;
    strumpack_mc64id_(icntl);
    //icntl[2] = 6; // print diagnostics
    //icntl[3] = 1; // no checking of input should be (slightly) faster
    strumpack_mc64(get_matching(job), &num, perm.data(),
                   liw, iw, ldw, dw, icntl, info);
    switch (info[0]) {
    case  0: break;
    case  1:
      std::cerr << "# ERROR: matrix is structurally singular" << std::endl;
      return 1;
      break;
    case  2:
        std::cerr << "# WARNING: mc64 scaling produced"
                  << " large scaling factors which may cause overflow!"
                  << std::endl;
      break;
    default:
      std::cerr << "# ERROR: mc64 failed with info[0]=" << info[0]
                << std::endl;
      return 1;
      break;
    }
    if (job == MatchingJob::MAX_DIAGONAL_PRODUCT_SCALING) { // scaling
      Dr.resize(n_);
      Dc.resize(n_);
#pragma omp parallel for
      for (integer_t i=0; i<n_; i++) {
        Dr[i] = scalar_t(std::exp(dw[i]));
        Dc[i] = scalar_t(std::exp(dw[n_+i]));
      }
      if (apply) apply_scaling(Dr, Dc);
    }
    if (apply) apply_column_permutation(perm);
    if (apply) symm_sparse_ = false;
    return 0;
  }

  template<typename scalar_t,typename integer_t> void
  CompressedSparseMatrix<scalar_t,integer_t>::symmetrize_sparsity() {
    if (symm_sparse_) return;
    std::vector<integer_t> a2_ctr(n_);
#pragma omp parallel for
    for (integer_t i=0; i<n_; i++)
      a2_ctr[i] = ptr_[i+1]-ptr_[i];

    bool change = false;
#pragma omp parallel for
    for (integer_t i=0; i<n_; i++)
      for (integer_t jj=ptr_[i]; jj<ptr_[i+1]; jj++) {
        integer_t kb = ptr_[ind_[jj]], ke = ptr_[ind_[jj]+1];
        if (std::find(ind()+kb, ind()+ke, i) == ind()+ke) {
#pragma omp critical
          {
            a2_ctr[ind_[jj]]++;
            change = true;
          }
        }
      }
    if (change) {
      std::vector<integer_t> a2_ptr(n_+1);
      a2_ptr[0] = 0;
      for (integer_t i=0; i<n_; i++) a2_ptr[i+1] = a2_ptr[i] + a2_ctr[i];
      auto new_nnz = a2_ptr[n_] - a2_ptr[0];
      std::vector<integer_t> a2_ind(new_nnz);
      std::vector<scalar_t> a2_val(new_nnz);
      nnz_ = new_nnz;
#pragma omp parallel for
      for (integer_t i=0; i<n_; i++) {
        a2_ctr[i] = a2_ptr[i] + ptr_[i+1] - ptr_[i];
        for (integer_t jj=ptr_[i], k=a2_ptr[i]; jj<ptr_[i+1]; jj++) {
          a2_ind[k  ] = ind_[jj];
          a2_val[k++] = val_[jj];
        }
      }
#pragma omp parallel for
      for (integer_t i=0; i<n_; i++)
        for (integer_t jj=ptr_[i]; jj<ptr_[i+1]; jj++) {
          integer_t kb = ptr_[ind_[jj]], ke = ptr_[ind_[jj]+1];
          if (std::find(ind()+kb,ind()+ke, i) == ind()+ke) {
            integer_t t = ind_[jj];
#pragma omp critical
            {
              a2_ind[a2_ctr[t]] = i;
              a2_val[a2_ctr[t]] = scalar_t(0.);
              a2_ctr[t]++;
            }
          }
        }
      std::swap(ptr_, a2_ptr);
      std::swap(ind_, a2_ind);
      std::swap(val_, a2_val);
    }
    symm_sparse_ = true;
  }

  template<typename scalar_t> scalar_t get_scalar(double vr, double vi) {
    return scalar_t(vr);
  }
  template<> inline std::complex<double> get_scalar(double vr, double vi) {
    return std::complex<double>(vr, vi);
  }
  template<> inline std::complex<float> get_scalar(double vr, double vi) {
    return std::complex<float>(vr, vi);
  }

  template<typename scalar_t,typename integer_t>
  std::vector<std::tuple<integer_t,integer_t,scalar_t>>
  CompressedSparseMatrix<scalar_t,integer_t>::read_matrix_market_entries
  (const std::string& filename) {
    std::cout << "# opening file \'" << filename << "\'" << std::endl;
    FILE *fp = fopen(filename.c_str(), "r");
    if (fp == NULL) {
      std::cerr << "ERROR: could not read file";
      exit(1);
    }
    const int max_cline = 256;
    char cline[max_cline];
    if (fgets(cline, max_cline, fp) == NULL) {
      std::cerr << "ERROR: could not read from file" << std::endl;
      exit(1);
    }
    printf("# %s", cline);
    if (strstr(cline, "pattern")) {
      std::cerr << "ERROR: This is not a matrix,"
                << " but just a sparsity pattern" << std::endl;
      exit(1);
    }
    else if (strstr(cline, "complex")) {
      if (!is_complex<scalar_t>()) {
        fclose(fp);
        throw "ERROR: Complex matrix";
      }
    }
    MMsym s = GENERAL;
    if (strstr(cline, "skew-symmetric")) {
      s = SKEWSYMMETRIC;
      symm_sparse_ = true;
    } else if (strstr(cline, "symmetric")) {
      s = SYMMETRIC;
      symm_sparse_ = true;
    } else if (strstr(cline, "hermitian")) {
      s = HERMITIAN;
      symm_sparse_ = true;
    }

    while (fgets(cline, max_cline, fp)) {
      if (cline[0] != '%') { // first line should be: m n nnz
        int m, in, innz;
        sscanf(cline, "%d %d %d", &m, &in, &innz);
        nnz_ = static_cast<integer_t>(innz);
        n_ = static_cast<integer_t>(in);
        std::cout << "# reading " << number_format_with_commas(m) << " by "
                  << number_format_with_commas(n_) << " matrix with "
                  << number_format_with_commas(nnz_) << " nnz's from "
                  << filename << std::endl;
        if (s != GENERAL) nnz_ = 2 * nnz_;
        if (m != n_) {
          std::cerr << "ERROR: matrix is not square!" << std::endl;
          exit(1);
        }
        break;
      }
    }
    std::vector<std::tuple<integer_t,integer_t,scalar_t>> A;
    A.reserve(nnz_);
    bool zero_based = false;
    if (!is_complex<scalar_t>()) {
      int ir, ic;
      double dv;
      while (fscanf(fp, "%d %d %lf\n", &ir, &ic, &dv) != EOF) {
        scalar_t v = static_cast<scalar_t>(dv);
        integer_t r = static_cast<integer_t>(ir);
        integer_t c = static_cast<integer_t>(ic);
        if (r==0 || c==0) zero_based = true;
        A.push_back(std::make_tuple(r, c, v));
        if (r != c) {
          switch (s) {
          case SKEWSYMMETRIC:
            A.push_back(std::make_tuple(c, r, -v));
            break;
          case SYMMETRIC:
            A.push_back(std::make_tuple(c, r, v));
            break;
          case HERMITIAN:
            A.push_back(std::make_tuple(c, r, blas::my_conj(v)));
            break;
          default: break;
          }
        }
      }
    } else {
      double vr=0, vi=0;
      int ir, ic;
      while (fscanf(fp, "%d %d %lf %lf\n", &ir, &ic, &vr, &vi) != EOF) {
        scalar_t v = get_scalar<scalar_t>(vr, vi);
        integer_t r = static_cast<integer_t>(ir),
          c = static_cast<integer_t>(ic);
        if (r==0 || c==0) zero_based = true;
        A.push_back(std::make_tuple(r, c, v));
        if (r != c) {
          switch (s) {
          case SKEWSYMMETRIC:
            A.push_back(std::make_tuple(c, r, -v));
            break;
          case SYMMETRIC:
            A.push_back(std::make_tuple(c, r, v));
            break;
          case HERMITIAN:
            A.push_back(std::make_tuple(c, r, blas::my_conj(v)));
            break;
          default: break;
          }
        }
      }
    }
    nnz_ = A.size();
    fclose(fp);
    if (!zero_based)
      for (auto& t : A) {
        std::get<0>(t)--;
        std::get<1>(t)--;
      }
    return A;
  }

  template<typename scalar_t,typename integer_t> void
  CompressedSparseMatrix<scalar_t,integer_t>::permute
  (const integer_t* iorder, const integer_t* order) {
    std::vector<integer_t> ptr(n_+1), ind(nnz_);
    std::vector<scalar_t> val(nnz_);
    integer_t nnz = 0;
    for (integer_t i=0; i<n_; i++) {
      auto ub = ptr_[iorder[i]+1];
      for (integer_t j=ptr_[iorder[i]]; j<ub; j++) {
        ind[nnz] = order[ind_[j]];
        val[nnz++] = val_[j];
      }
      ptr[i+1] = nnz;
    }
#pragma omp parallel for
    for (integer_t i=0; i<n_; i++)
      sort_indices_values
        (ind.data()+ptr[i], val.data()+ptr[i], integer_t(0), ptr[i+1]-ptr[i]);
    std::swap(ptr_, ptr);
    std::swap(ind_, ind);
    std::swap(val_, val);
  }

  template<typename scalar_t,typename integer_t> long long
  CompressedSparseMatrix<scalar_t,integer_t>::spmv_flops() const {
    return (is_complex<scalar_t>() ? 4 : 1 ) * (2ll * nnz_ - n_);
  }

  template<typename scalar_t,typename integer_t> long long
  CompressedSparseMatrix<scalar_t,integer_t>::spmv_bytes() const {
    // read   ind  nnz  integer_t
    //        val  nnz  scalar_t
    //        ptr  n    integer_t
    //        x    n    scalar_t
    //        y    n    scalar_t
    // write  y    n    scalar_t
    return (sizeof(scalar_t) * 3 + sizeof(integer_t)) * n_
      + (sizeof(scalar_t) + sizeof(integer_t)) * nnz_;
  }


  // explicit template instantiations
  template class CompressedSparseMatrix<float,int>;
  template class CompressedSparseMatrix<double,int>;
  template class CompressedSparseMatrix<std::complex<float>,int>;
  template class CompressedSparseMatrix<std::complex<double>,int>;

  template class CompressedSparseMatrix<float,long int>;
  template class CompressedSparseMatrix<double,long int>;
  template class CompressedSparseMatrix<std::complex<float>,long int>;
  template class CompressedSparseMatrix<std::complex<double>,long int>;

  template class CompressedSparseMatrix<float,long long int>;
  template class CompressedSparseMatrix<double,long long int>;
  template class CompressedSparseMatrix<std::complex<float>,long long int>;
  template class CompressedSparseMatrix<std::complex<double>,long long int>;

} //end namespace strumpack
