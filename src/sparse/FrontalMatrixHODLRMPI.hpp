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
 * five (5) year renewals, the U.S. Government igs granted for itself
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
#ifndef FRONTAL_MATRIX_HODLR_MPI_HPP
#define FRONTAL_MATRIX_HODLR_MPI_HPP

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include "misc/TaskTimer.hpp"
#include "misc/MPIWrapper.hpp"
#include "dense/DistributedMatrix.hpp"
#include "CompressedSparseMatrix.hpp"
#include "MatrixReordering.hpp"
#include "FrontalMatrixMPI.hpp"
#include "HODLR/HODLRMatrix.hpp"
#include "HODLR/LRBFMatrix.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t>
  class FrontalMatrixHODLRMPI : public FrontalMatrixMPI<scalar_t,integer_t> {
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using FMPI_t = FrontalMatrixMPI<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DistM_t = DistributedMatrix<scalar_t>;
    using DistMW_t = DistributedMatrixWrapper<scalar_t>;
    using ExtAdd = ExtendAdd<scalar_t,integer_t>;
    using Opts_t = SPOptions<scalar_t>;
    using VecVec_t = std::vector<std::vector<std::size_t>>;
    template<typename _scalar_t,typename _integer_t> friend class ExtendAdd;

  public:
    FrontalMatrixHODLRMPI
    (integer_t sep, integer_t sep_begin, integer_t sep_end,
     std::vector<integer_t>& upd, const MPIComm& comm, int _total_procs);
    FrontalMatrixHODLRMPI(const FrontalMatrixHODLRMPI&) = delete;
    FrontalMatrixHODLRMPI& operator=(FrontalMatrixHODLRMPI const&) = delete;

    void release_work_memory() override;

    void sample_CB
    (Trans op, const DistM_t& R, DistM_t& S, F_t* pa) const override;
    void sample_children_CB(Trans op, const DistM_t& R, DistM_t& S);

    void skinny_extend_add(DistM_t& cSl, DistM_t& cSr, DistM_t& S);

    void multifrontal_factorization
    (const SpMat_t& A, const Opts_t& opts,
     int etree_level=0, int task_depth=0) override;

    void forward_multifrontal_solve
    (DenseM_t& bloc, DistM_t* bdist, DistM_t& bupd, DenseM_t& seqbupd,
     int etree_level=0) const override;
    void backward_multifrontal_solve
    (DenseM_t& yloc, DistM_t* ydist, DistM_t& yupd, DenseM_t& seqyupd,
     int etree_level=0) const override;

    long long node_factor_nonzeros() const;
    integer_t maximum_rank(int task_depth) const;
    std::string type() const override { return "FrontalMatrixHODLRMPI"; }

    void extract_CB_sub_matrix_2d
    (const std::vector<std::vector<std::size_t>>& I,
     const std::vector<std::vector<std::size_t>>& J,
     std::vector<DistM_t>& B) const override;

    void partition
    (const Opts_t& opts, const SpMat_t& A, integer_t* sorder,
     bool is_root=true, int task_depth=0) override;

  private:
    HODLR::HODLRMatrix<scalar_t> F11_;
    HODLR::LRBFMatrix<scalar_t> F12_, F21_;
    std::unique_ptr<HODLR::HODLRMatrix<scalar_t>> F22_;
    HSS::HSSPartitionTree sep_tree_;

    void construct_hierarchy(const SpMat_t& A, const Opts_t& opts);
    void compress_sampling(const SpMat_t& A, const Opts_t& opts);
    void compress_extraction(const SpMat_t& A, const Opts_t& opts);
    void compress_flops_F11();
    void compress_flops_F12_F21();
    void compress_flops_F22();
    void compress_flops_Schur();

    using FrontalMatrix<scalar_t,integer_t>::lchild_;
    using FrontalMatrix<scalar_t,integer_t>::rchild_;
    using FrontalMatrix<scalar_t,integer_t>::sep_begin_;
    using FrontalMatrix<scalar_t,integer_t>::sep_end_;
    using FrontalMatrix<scalar_t,integer_t>::dim_sep;
    using FrontalMatrix<scalar_t,integer_t>::dim_upd;
    using FrontalMatrix<scalar_t,integer_t>::dim_blk;
    using FrontalMatrixMPI<scalar_t,integer_t>::visit;
    using FrontalMatrixMPI<scalar_t,integer_t>::grid;
    using FrontalMatrixMPI<scalar_t,integer_t>::Comm;
  };

  template<typename scalar_t,typename integer_t>
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::FrontalMatrixHODLRMPI
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd, const MPIComm& comm, int total_procs)
    : FrontalMatrixMPI<scalar_t,integer_t>
    (sep, sep_begin, sep_end, upd, comm, total_procs) {
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::release_work_memory() {
    F22_.reset(nullptr);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::sample_CB
  (Trans op, const DistM_t& R, DistM_t& S, F_t* pa) const {
    if (!dim_upd()) return;
    if (Comm().is_null()) return;
    S = DistM_t(grid(), dim_upd(), R.cols());
    TIMER_TIME(TaskType::F22_MULT, 2, t_sprod);
    F22_->mult(op, R, S);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::sample_children_CB
  (Trans op, const DistM_t& R, DistM_t& S) {
    if (!lchild_ && !rchild_) return;
    DistM_t Sl, Sr, Rl, Rr;
    DenseM_t seqSl, seqSr, seqRl, seqRr;
    if (lchild_) {
      lchild_->extract_from_R2D(R, Rl, seqRl, this, visit(lchild_));
      seqSl = DenseM_t(seqRl.rows(), seqRl.cols());
      seqSl.zero();
    }
    if (rchild_) {
      rchild_->extract_from_R2D(R, Rr, seqRr, this, visit(rchild_));
      seqSr = DenseM_t(seqRr.rows(), seqRr.cols());
      seqSr.zero();
    }
    if (visit(lchild_)) lchild_->sample_CB(op, Rl, Sl, seqRl, seqSl, this);
    if (visit(rchild_)) rchild_->sample_CB(op, Rr, Sr, seqRr, seqSr, this);
    std::vector<std::vector<scalar_t>> sbuf(this->P());
    if (visit(lchild_)) lchild_->skinny_ea_to_buffers(Sl, seqSl, sbuf, this);
    if (visit(rchild_)) rchild_->skinny_ea_to_buffers(Sr, seqSr, sbuf, this);
    std::vector<scalar_t> rbuf;
    std::vector<scalar_t*> pbuf;
    Comm().all_to_all_v(sbuf, rbuf, pbuf);
    if (lchild_)
      lchild_->skinny_ea_from_buffers(S, pbuf.data(), this);
    if (rchild_)
      rchild_->skinny_ea_from_buffers
        (S, pbuf.data()+this->master(rchild_), this);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::extract_CB_sub_matrix_2d
  (const VecVec_t& I, const VecVec_t& J, std::vector<DistM_t>& B) const {
    if (Comm().is_null() || !dim_upd()) return;
    TIMER_TIME(TaskType::HSS_EXTRACT_SCHUR, 3, t_ex_schur);
    auto nB = I.size();
    std::vector<std::vector<std::size_t>> lI(nB), lJ(nB), oI(nB), oJ(nB);
    std::vector<DistM_t> Bloc(nB);
    for (std::size_t i=0; i<nB; i++) {
      this->find_upd_indices(I[i], lI[i], oI[i]);
      this->find_upd_indices(J[i], lJ[i], oJ[i]);
      Bloc[i] = DistM_t(grid(), lI[i].size(), lJ[i].size());
    }
    F22_->extract_elements(lI, lJ, Bloc);
    std::vector<std::vector<scalar_t>> sbuf(this->P());
    for (std::size_t i=0; i<nB; i++)
      ExtAdd::extend_copy_to_buffers(Bloc[i], oI[i], oJ[i], B[i], sbuf);
    std::vector<scalar_t> rbuf;
    std::vector<scalar_t*> pbuf;
    Comm().all_to_all_v(sbuf, rbuf, pbuf);
    for (std::size_t i=0; i<nB; i++)
      ExtAdd::extend_copy_from_buffers(B[i], oI[i], oJ[i], Bloc[i], pbuf);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::multifrontal_factorization
  (const SpMat_t& A, const Opts_t& opts, int etree_level, int task_depth) {
    if (visit(lchild_))
      lchild_->multifrontal_factorization(A, opts, etree_level+1, task_depth);
    if (visit(rchild_))
      rchild_->multifrontal_factorization(A, opts, etree_level+1, task_depth);
    if (!dim_blk()) return;
    TaskTimer t("");
    if (/*etree_level == 0 && */opts.print_root_front_stats()) t.start();
    construct_hierarchy(A, opts);
    switch (opts.HODLR_options().compression_algorithm()) {
    case HODLR::CompressionAlgorithm::RANDOM_SAMPLING:
      compress_sampling(A, opts); break;
    case HODLR::CompressionAlgorithm::ELEMENT_EXTRACTION:
      compress_extraction(A, opts); break;
    }
    if (/*etree_level == 0 && */opts.print_root_front_stats()) {
      auto time = t.elapsed();
      float perbyte = 1.0e6 / sizeof(scalar_t);
      float tmp[5];
      float& F11nnzH = tmp[0];
      float& F11nnzFactors = tmp[1];
      float& F12nnzH = tmp[2];
      float& F21nnzH = tmp[3];
      float& F22nnzH = tmp[4];
      auto F11rank = F11_.get_stat("Rank_max");
      F11nnzH = F11_.get_stat("Mem_Fill") * perbyte;
      F11nnzFactors = F11_.get_stat("Mem_Factor") * perbyte;
      auto F12rank = F12_.get_stat("Rank_max");
      F12nnzH = F12_.get_stat("Mem_Fill") * perbyte;
      auto F21rank = F21_.get_stat("Rank_max");
      F21nnzH = F21_.get_stat("Mem_Fill") * perbyte;
      int F22rank = 0;
      if (F22_) {
        F22rank = F22_->get_stat("Rank_max");
        F22nnzH = F22_->get_stat("Mem_Fill") * perbyte;
      } else F22nnzH = 0.;
      Comm().reduce(tmp, 5, MPI_SUM);
      if (Comm().is_root())
        std::cout << "#   - HODLRMPI front: Nsep= " << dim_sep()
                  << " , Nupd= " << dim_upd() << "\n#       "
                  << " nnz(F11)= " << F11nnzH << " , nnz(factor(F11))= "
                  << F11nnzFactors << " , rank(F11)= " << F11rank << " ,\n#       "
                  << " nnz(F12)= " << F12nnzH << " , rank(F12)= " << F12rank << " , "
                  << " nnz(F21)= " << F21nnzH << " , rank(F21)= " << F21rank << " ,\n#       "
                  << " nnz(F22)= " << F22nnzH << " , rank(F22)= " << F22rank << " , "
                  << (float(tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4])
                      / (float(dim_blk())*dim_blk()) * 100.)
                  << " %compression, time= " << time
                  << " sec" << std::endl;
    }
    if (lchild_) lchild_->release_work_memory();
    if (rchild_) rchild_->release_work_memory();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::compress_extraction
  (const SpMat_t& A, const Opts_t& opts) {
    auto extract_F11 =
      [&](VecVec_t& I, VecVec_t& J, std::vector<DistMW_t>& B,
          HODLR::ExtractionMeta&) {
        for (auto& Ik : I) for (auto& i : Ik) i += sep_begin_;
        for (auto& Jk : J) for (auto& j : Jk) j += sep_begin_;
        this->extract_2d(A, I, J, B);
      };
    { TIMER_TIME(TaskType::HSS_COMPRESS, 0, t_f11_compress);
      F11_.compress(extract_F11); }
    { TIMER_TIME(TaskType::HSS_FACTOR, 0, t_fact);
      F11_.factor(); }
    compress_flops_F11();
    if (dim_upd()) {
      auto extract_F12 =
        [&](VecVec_t& I, VecVec_t& J, std::vector<DistMW_t>& B,
            HODLR::ExtractionMeta&) {
          for (auto& Ik : I) for (auto& i : Ik) i += sep_begin_;
          for (auto& Jk : J) for (auto& j : Jk) j = this->upd_[j];
          this->extract_2d(A, I, J, B);
        };
      auto extract_F21 =
        [&](VecVec_t& I, VecVec_t& J, std::vector<DistMW_t>& B,
            HODLR::ExtractionMeta&) {
          for (auto& Ik : I) for (auto& i : Ik) i = this->upd_[i];
          for (auto& Jk : J) for (auto& j : Jk) j += sep_begin_;
          this->extract_2d(A, I, J, B);
        };
      { TIMER_TIME(TaskType::LRBF_COMPRESS, 0, t_lrbf_compress);
        F12_ = HODLR::LRBFMatrix<scalar_t>(F11_, *F22_);
        F21_ = HODLR::LRBFMatrix<scalar_t>(*F22_, F11_);
        F12_.compress(extract_F12);
        F21_.compress(extract_F21); }
      compress_flops_F12_F21();

      // first construct S=-F21*inv(F11)*F12 using matvecs, then
      // construct F22+S using element extraction
      auto sample_Schur =
        [&](Trans op, scalar_t a, const DenseM_t& R, scalar_t b, DenseM_t& S) {
          TIMER_TIME(TaskType::RANDOM_SAMPLING, 0, t_sampling);
          TIMER_TIME(TaskType::HSS_SCHUR_PRODUCT, 2, t_sprod);
          DenseM_t F12R(F12_.lrows(), R.cols()),
            invF11F12R(F11_.lrows(), R.cols());
          auto& F1 = (op == Trans::N) ? F12_ : F21_;
          auto& F2 = (op == Trans::N) ? F21_ : F12_;
          a = -a; // construct S=-F21*inv(F11)*F12
          if (a != scalar_t(1.)) {
            DenseM_t Rtmp(R);
            Rtmp.scale(a);
            F1.mult(op, Rtmp, F12R);
          } else F1.mult(op, R, F12R);
          F11_.inv_mult(op, F12R, invF11F12R);
          //F11_.solve(F12R, invF11F12R);
          if (b != scalar_t(0.)) {
            DenseM_t Stmp(S.rows(), S.cols());
            F2.mult(op, invF11F12R, Stmp);
            S.scale_and_add(b, Stmp);
          } else F2.mult(op, invF11F12R, S);
          compress_flops_Schur();
        };
      HODLR::LRBFMatrix<scalar_t> Schur(*F22_, *F22_);
      Schur.compress(sample_Schur);
      auto extract_F22 =
        [&](VecVec_t& I, VecVec_t& J, std::vector<DistMW_t>& B,
            HODLR::ExtractionMeta& e) {
          for (auto& Ik : I) for (auto& i : Ik) i = this->upd_[i];
          for (auto& Jk : J) for (auto& j : Jk) j = this->upd_[j];
          this->extract_2d(A, I, J, B);
          Schur.extract_add_elements(e, B);
        };
      TIMER_TIME(TaskType::HSS_COMPRESS, 0, t_f22_compress);
      F22_->compress(extract_F22);
      compress_flops_F22();
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::compress_sampling
  (const SpMat_t& A, const Opts_t& opts) {
    const auto dsep = dim_sep();
    const auto dupd = dim_upd();
    auto sample_front = [&](Trans op, const DistM_t& R, DistM_t& S) {
      S.zero();
      TIMER_TIME(TaskType::FRONT_MULTIPLY_2D, 1, t_fmult);
      A.front_multiply_2d(op, sep_begin_, sep_end_, this->upd_, R, S, 0);
      TIMER_STOP(t_fmult);
      TIMER_TIME(TaskType::UUTXR, 1, t_UUtxR);
      sample_children_CB(op, R, S);
      TIMER_STOP(t_UUtxR);
    };
    auto sample_F11 = [&](Trans op, const DenseM_t& R1, DenseM_t& S1) {
      TIMER_TIME(TaskType::RANDOM_SAMPLING, 0, t_sampling);
      auto n = R1.cols();
      DistM_t R(grid(), dsep+dupd, n), S(grid(), dsep+dupd, n);
      DistMW_t R1dist(dsep, n, R, 0, 0);
      DistMW_t(dupd, n, R, dsep, 0).zero();
      F11_.redistribute_1D_to_2D(R1, R1dist);
      sample_front(op, R, S);
      F11_.redistribute_2D_to_1D(DistMW_t(dsep, n, S, 0, 0), S1);
    };
    { TIMER_TIME(TaskType::HSS_COMPRESS, 0, t_f11_compress);
      F11_.compress(sample_F11); }
    { TIMER_TIME(TaskType::HSS_FACTOR, 0, t_fact);
      F11_.factor(); }
    compress_flops_F11();
    if (dupd) {
      auto sample_F12 = [&]
        (Trans op, scalar_t a, const DenseM_t& Rl, scalar_t b, DenseM_t& Sl) {
        TIMER_TIME(TaskType::RANDOM_SAMPLING, 0, t_sampling);
        auto n = Rl.cols();
        DistM_t R(grid(), dsep+dupd, n), S(grid(), dsep+dupd, n);
        if (op == Trans::N) {
          DistMW_t R2dist(dupd, n, R, dsep, 0);
          DistMW_t(dsep, n, R, 0, 0).zero();
          F12_.redistribute_1D_to_2D(Rl, R2dist, F12_.cdist());
          sample_front(op, R, S);
          F12_.redistribute_2D_to_1D
            (a, DistMW_t(dsep, n, S, 0, 0), b, Sl, F12_.rdist());
        } else {
          DistMW_t R1dist(dsep, n, R, 0, 0);
          DistMW_t(dupd, n, R, dsep, 0).zero();
          F12_.redistribute_1D_to_2D(Rl, R1dist, F12_.rdist());
          sample_front(op, R, S);
          F12_.redistribute_2D_to_1D
          (a, DistMW_t(dupd, n, S, dsep, 0), b, Sl, F12_.cdist());
        }
      };
      auto sample_F21 = [&]
        (Trans op, scalar_t a, const DenseM_t& Rl, scalar_t b, DenseM_t& Sl) {
        TIMER_TIME(TaskType::RANDOM_SAMPLING, 0, t_sampling);
        auto n = Rl.cols();
        DistM_t R(grid(), dsep+dupd, n), S(grid(), dsep+dupd, n);
        if (op == Trans::N) {
          DistMW_t R1dist(dsep, n, R, 0, 0);
          DistMW_t(dupd, n, R, dsep, 0).zero();
          F21_.redistribute_1D_to_2D(Rl, R1dist, F21_.cdist());
          sample_front(op, R, S);
          F12_.redistribute_2D_to_1D
            (a, DistMW_t(dupd, n, S, dsep, 0), b, Sl, F21_.rdist());
        } else {
          DistMW_t R2dist(dupd, n, R, dsep, 0);
          DistMW_t(dsep, n, R, 0, 0).zero();
          F21_.redistribute_1D_to_2D(Rl, R2dist, F21_.rdist());
          sample_front(op, R, S);
          F12_.redistribute_2D_to_1D
            (a, DistMW_t(dsep, n, S, 0, 0), b, Sl, F21_.cdist());
        }
      };
      { TIMER_TIME(TaskType::LRBF_COMPRESS, 0, t_lrbf_compress);
        F12_ = HODLR::LRBFMatrix<scalar_t>(F11_, *F22_);
        F12_.compress(sample_F12);
        F21_ = HODLR::LRBFMatrix<scalar_t>(*F22_, F11_);
        F21_.compress(sample_F21, F12_.get_stat("Rank_max")+10); }
      compress_flops_F12_F21();

      auto sample_F22 = [&](Trans op, const DenseM_t& R2, DenseM_t& S2) {
        TIMER_TIME(TaskType::RANDOM_SAMPLING, 0, t_sampling);
        auto n = R2.cols();
        DistM_t R(grid(), dsep+dupd, n), S(grid(), dsep+dupd, n);
        DistMW_t(dsep, n, R, 0, 0).zero();
        DistMW_t R2dist(dupd, n, R, dsep, 0);
        F22_->redistribute_1D_to_2D(R2, R2dist);
        sample_front(op, R, S);
        F22_->redistribute_2D_to_1D(DistMW_t(dupd, n, S, dsep, 0), S2);
        TIMER_TIME(TaskType::HSS_SCHUR_PRODUCT, 2, t_sprod);
        DenseM_t F12R2(F12_.lrows(), R2.cols()),
        invF11F12R2(F11_.lrows(), R2.cols()), S2tmp(S2.rows(), S2.cols());
        if (op == Trans::N) {
          F12_.mult(op, R2, F12R2);
          F11_.inv_mult(op, F12R2, invF11F12R2);
          //F11_.solve(F12R, invF11F12R);
          F21_.mult(op, invF11F12R2, S2tmp);
        } else {
          F21_.mult(op, R2, F12R2);
          F11_.inv_mult(op, F12R2, invF11F12R2);
          F12_.mult(op, invF11F12R2, S2tmp);
        }
        S2.scaled_add(scalar_t(-1.), S2tmp);
        compress_flops_Schur();
      };
      { TIMER_TIME(TaskType::HSS_COMPRESS, 0, t_f22_compress);
        F22_->compress(sample_F22, std::max(F12_.get_stat("Rank_max"),
                                            F21_.get_stat("Rank_max"))); }
      compress_flops_F22();
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::compress_flops_F11() {
#if defined(STRUMPACK_COUNT_FLOPS)
    long long int f11_fill_flops = F11_.get_stat("Flop_Fill");
    long long int f11_factor_flops = F11_.get_stat("Flop_Factor");
    STRUMPACK_HODLR_F11_FILL_FLOPS(f11_fill_flops);
    STRUMPACK_ULV_FACTOR_FLOPS(f11_factor_flops);
    STRUMPACK_FLOPS(f11_fill_flops + f11_factor_flops);
#endif
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::compress_flops_F12_F21() {
#if defined(STRUMPACK_COUNT_FLOPS)
    long long int f12_fill_flops = F12_.get_stat("Flop_Fill");
    long long int f21_fill_flops = F21_.get_stat("Flop_Fill");
    STRUMPACK_HODLR_F12_FILL_FLOPS(f12_fill_flops);
    STRUMPACK_HODLR_F21_FILL_FLOPS(f21_fill_flops);
    STRUMPACK_FLOPS(f12_fill_flops + f21_fill_flops);
#endif
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::compress_flops_F22() {
#if defined(STRUMPACK_COUNT_FLOPS)
    long long int f22_fill_flops = F22_->get_stat("Flop_Fill");
    STRUMPACK_HODLR_F22_FILL_FLOPS(f22_fill_flops);
    STRUMPACK_FLOPS(f22_fill_flops);
#endif
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::compress_flops_Schur() {
#if defined(STRUMPACK_COUNT_FLOPS)
    long long int f21_mult_flops = F21_.get_stat("Flop_C_Mult"),
      invf11_mult_flops = F11_.get_stat("Flop_C_Mult"),
      f12_mult_flops = F12_.get_stat("Flop_C_Mult"),
      schur_flops = f21_mult_flops + invf11_mult_flops
      + f12_mult_flops; // + S.rows()*S.cols(); // extra scaling?
    STRUMPACK_SCHUR_FLOPS(schur_flops);
    STRUMPACK_FLOPS(schur_flops);
    STRUMPACK_HODLR_F21_MULT_FLOPS(f21_mult_flops);
    STRUMPACK_HODLR_F12_MULT_FLOPS(f12_mult_flops);
    STRUMPACK_HODLR_INVF11_MULT_FLOPS(invf11_mult_flops);
#endif
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::forward_multifrontal_solve
  (DenseM_t& bloc, DistM_t* bdist, DistM_t& bupd, DenseM_t& seqbupd,
   int etree_level) const {
    DistM_t CBl, CBr;
    DenseM_t seqCBl, seqCBr;
    if (visit(lchild_))
      lchild_->forward_multifrontal_solve
        (bloc, bdist, CBl, seqCBl, etree_level+1);
    if (visit(rchild_))
      rchild_->forward_multifrontal_solve
        (bloc, bdist, CBr, seqCBr, etree_level+1);
    DistM_t& b = bdist[this->sep_];
    bupd = DistM_t(grid(), dim_upd(), b.cols());
    bupd.zero();
    this->extend_add_b(b, bupd, CBl, CBr, seqCBl, seqCBr);
    if (this->dim_sep()) {
      TIMER_TIME(TaskType::SOLVE_LOWER, 0, t_s);
      DistM_t rhs(b);
      F11_.solve(rhs, b);
      STRUMPACK_FLOPS(F11_.get_stat("Flop_Solve"));
      if (this->dim_upd()) {
        DistM_t tmp(bupd.grid(), bupd.rows(), bupd.cols());
        F21_.mult(Trans::N, b, tmp);
        bupd.scaled_add(scalar_t(-1.), tmp);
        STRUMPACK_FLOPS(F21_.get_stat("Flop_C_Mult") +
                        2*bupd.rows()*bupd.cols());
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::backward_multifrontal_solve
  (DenseM_t& yloc, DistM_t* ydist, DistM_t& yupd, DenseM_t& seqyupd,
   int etree_level) const {
    DistM_t& y = ydist[this->sep_];
    if (this->dim_sep() && this->dim_upd()) {
      TIMER_TIME(TaskType::SOLVE_UPPER, 0, t_s);
      DistM_t tmp(y.grid(), y.rows(), y.cols()), tmp2(y.grid(), y.rows(), y.cols());
      F12_.mult(Trans::N, yupd, tmp);
      F11_.solve(tmp, tmp2);
      y.scaled_add(scalar_t(-1.), tmp2);
      STRUMPACK_FLOPS(F12_.get_stat("Flop_C_Mult") +
                      F11_.get_stat("Flop_Solve") +
                      2*yloc.rows()*yloc.cols());
    }
    DistM_t CBl, CBr;
    DenseM_t seqCBl, seqCBr;
    this->extract_b(y, yupd, CBl, CBr, seqCBl, seqCBr);
    if (visit(lchild_))
      lchild_->backward_multifrontal_solve
        (yloc, ydist, CBl, seqCBl, etree_level+1);
    if (visit(rchild_))
      rchild_->backward_multifrontal_solve
        (yloc, ydist, CBr, seqCBr, etree_level+1);
  }

  template<typename scalar_t,typename integer_t> integer_t
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::maximum_rank
  (int task_depth) const {
    integer_t mr = std::max(F11_.get_stat("Rank_max"),
                            std::max(F12_.get_stat("Rank_max"),
                                     F21_.get_stat("Rank_max")));
    if (visit(lchild_))
      mr = std::max(mr, lchild_->maximum_rank(task_depth));
    if (visit(rchild_))
      mr = std::max(mr, rchild_->maximum_rank(task_depth));
    return mr;
  }

  template<typename scalar_t,typename integer_t> long long
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::node_factor_nonzeros() const {
    return (F11_.get_stat("Mem_Fill") + F11_.get_stat("Mem_Factor")
            + F12_.get_stat("Mem_Fill") + F21_.get_stat("Mem_Fill"))
      * 1.0e6 / sizeof(scalar_t);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::partition
  (const Opts_t& opts, const SpMat_t& A,
   integer_t* sorder, bool is_root, int task_depth) {
    if (Comm().is_null()) return;
    auto g = A.extract_graph
      (opts.separator_ordering_level(), sep_begin_, sep_end_);
    sep_tree_ = g.recursive_bisection
      (opts.compression_leaf_size(), 0,
       sorder+sep_begin_, nullptr, 0, 0, dim_sep());
    std::vector<integer_t> siorder(dim_sep());
    for (integer_t i=sep_begin_; i<sep_end_; i++)
      siorder[sorder[i]] = i - sep_begin_;
    for (integer_t i=sep_begin_; i<sep_end_; i++)
      sorder[i] += sep_begin_;
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixHODLRMPI<scalar_t,integer_t>::construct_hierarchy
  (const SpMat_t& A, const Opts_t& opts) {
    auto g = A.extract_graph
      (opts.separator_ordering_level(), sep_begin_, sep_end_);
    auto adm = g.admissibility(sep_tree_.template leaf_sizes<int>());
    F11_ = std::move
      (HODLR::HODLRMatrix<scalar_t>
       (Comm(), sep_tree_, adm, g, opts.HODLR_options()));
    if (dim_upd()) {
      HSS::HSSPartitionTree CB_tree(dim_upd());
      CB_tree.refine(opts.HODLR_options().leaf_size());
      if (opts.HODLR_options().compression_algorithm() ==
          HODLR::CompressionAlgorithm::ELEMENT_EXTRACTION &&
          opts.HODLR_options().geo() == 2) {
        {
          auto gCB = A.extract_graph_CB(opts.separator_ordering_level(), this->upd());
          auto admCB = gCB.admissibility(CB_tree.template leaf_sizes<int>());
          F22_ = std::unique_ptr<HODLR::HODLRMatrix<scalar_t>>
            (new HODLR::HODLRMatrix<scalar_t>
             (Comm(), CB_tree, admCB, gCB, opts.HODLR_options()));
        } {
          auto g12 = A.extract_graph_sep_CB
            (opts.separator_ordering_level(), sep_begin_, sep_end_, this->upd());
          auto adm12 = g12.admissibility
            (sep_tree_.template leaf_sizes<int>(),
             CB_tree.template leaf_sizes<int>());
          F12_ = HODLR::LRBFMatrix<scalar_t>
            (F11_, sep_tree_, *F22_, CB_tree, adm12, g12, opts.HODLR_options());
        } {
          auto g21 = A.extract_graph_CB_sep
            (opts.separator_ordering_level(), sep_begin_, sep_end_, this->upd());
          auto adm21 = g21.admissibility
            (CB_tree.template leaf_sizes<int>(),
             sep_tree_.template leaf_sizes<int>());
          F21_ = HODLR::LRBFMatrix<scalar_t>
            (*F22_, CB_tree, F11_, sep_tree_, adm21, g21, opts.HODLR_options());
        }
      } else {
        F22_ = std::unique_ptr<HODLR::HODLRMatrix<scalar_t>>
          (new HODLR::HODLRMatrix<scalar_t>
           (Comm(), CB_tree, opts.HODLR_options()));
        F12_ = HODLR::LRBFMatrix<scalar_t>(F11_, *F22_);
        F21_ = HODLR::LRBFMatrix<scalar_t>(*F22_, F11_);
      }
    }
  }

} // end namespace strumpack

#endif //FRONTAL_MATRIX_HODLR_MPI_HPP
