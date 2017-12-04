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
#ifndef FRONTAL_MATRIX_MPI_HPP
#define FRONTAL_MATRIX_MPI_HPP

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include "misc/MPIWrapper.hpp"
#include "misc/TaskTimer.hpp"
#include "dense/DistributedMatrix.hpp"
#include "CompressedSparseMatrix.hpp"
#include "MatrixReordering.hpp"
#include "ExtendAdd.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t>
  class FrontalMatrixMPI : public FrontalMatrix<scalar_t,integer_t> {
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using FMPI_t = FrontalMatrixMPI<scalar_t,integer_t>;
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DistM_t = DistributedMatrix<scalar_t>;
    using DistMW_t = DistributedMatrixWrapper<scalar_t>;
    using ExtAdd = ExtendAdd<scalar_t,integer_t>;
    template<typename _scalar_t,typename _integer_t> friend class ExtendAdd;

  public:
    FrontalMatrixMPI
    (integer_t _sep, integer_t _sep_begin, integer_t _sep_end,
     std::vector<integer_t>& _upd, MPI_Comm _front_comm, int _total_procs);
    FrontalMatrixMPI(const FrontalMatrixMPI&) = delete;
    FrontalMatrixMPI& operator=(FrontalMatrixMPI const&) = delete;
    virtual ~FrontalMatrixMPI();

    void sample_CB
    (const SPOptions<scalar_t>& opts, const DenseM_t& R, DenseM_t& Sr,
     DenseM_t& Sc, F_t* parent, int task_depth=0) override {};
    virtual void sample_CB
    (const SPOptions<scalar_t>& opts, const DistM_t& R, DistM_t& Sr,
     DistM_t& Sc, F_t* pa) const = 0;

    void extract_2d
    (const SpMat_t& A, const std::vector<std::size_t>& I,
     const std::vector<std::size_t>& J, DistM_t& B) const;
    void get_child_submatrix_2d
    (const F_t* ch, const std::vector<std::size_t>& I,
     const std::vector<std::size_t>& J,
     DistM_t& B) const;
    void extract_CB_sub_matrix
    (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
     DenseM_t& B, int task_depth) const {};
    virtual void extract_CB_sub_matrix_2d
    (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
     DistM_t& B) const = 0;

    void extend_add_b
    (DistM_t& b, DistM_t& bupd, const DistM_t& CBl, const DistM_t& CBr,
     const DenseM_t& seqCBl, const DenseM_t& seqCBr) const;
    void extract_b
    (const DistM_t& b, const DistM_t& bupd, DistM_t& CBl, DistM_t& CBr,
     DenseM_t& seqCBl, DenseM_t& seqCBr) const;

    inline bool visit(const F_t* ch) const;
    inline int child_master(const F_t* ch) const;
    inline int blacs_context() const { return ctxt; }
    inline int blacs_context_all() const { return ctxt_all; }
    inline bool active() const {
      return front_comm != MPI_COMM_NULL &&
        mpi_rank(front_comm) < proc_rows*proc_cols;
    }
    static inline void processor_grid
    (int np_procs, int& np_rows, int& np_cols) {
      np_cols = std::floor(std::sqrt((float)np_procs));
      np_rows = np_procs / np_cols;
    }
    inline int np_rows() const { return proc_rows; }
    inline int np_cols() const { return proc_cols; }
    inline int find_rank(integer_t r, integer_t c, const DistM_t& F) const;
    inline int find_rank_fixed
    (integer_t r, integer_t c, const DistM_t& F) const;
    virtual long long dense_factor_nonzeros(int task_depth=0) const;
    virtual std::string type() const { return "FrontalMatrixMPI"; }
    virtual bool isMPI() const { return true; }
    MPI_Comm comm() const { return front_comm; }
    virtual void bisection_partitioning
    (const SPOptions<scalar_t>& opts, integer_t* sorder,
     bool isroot=true, int task_depth=0);

    friend class FrontalMatrixDenseMPI<scalar_t,integer_t>;
    friend class FrontalMatrixHSSMPI<scalar_t,integer_t>;

  protected:
    // this is a blacs context with only the active process for this front
    int ctxt;
    // this is a blacs context with all process for this front
    int ctxt_all;
    // number of process rows in the blacs ctxt
    int proc_rows;
    // number of process columns in the blacs ctxt
    int proc_cols;
    // number of processes that work on the subtree belonging to this front,
    int total_procs;
    // this can be more than the processes in the blacs context ctxt
    // and is not necessarily the same as mpi_nprocs(front_comm),
    // because if this rank is not part of front_comm,
    // mpi_nprocs(front_comm) == 0
    MPI_Comm front_comm;  // MPI communicator for this front
  };

  template<typename scalar_t,typename integer_t>
  FrontalMatrixMPI<scalar_t,integer_t>::FrontalMatrixMPI
  (integer_t _sep, integer_t _sep_begin, integer_t _sep_end,
   std::vector<integer_t>& _upd, MPI_Comm _front_comm, int _total_procs)
    : F_t(NULL, NULL,_sep, _sep_begin, _sep_end, _upd),
    total_procs(_total_procs), front_comm(_front_comm) {
    processor_grid(total_procs, proc_rows, proc_cols);
    if (front_comm != MPI_COMM_NULL) {
      int active_procs = proc_rows * proc_cols;
      if (active_procs < total_procs) {
        auto active_front_comm = mpi_sub_comm(front_comm, 0, active_procs);
        if (mpi_rank(front_comm) < active_procs) {
          ctxt = scalapack::Csys2blacs_handle(active_front_comm);
          scalapack::Cblacs_gridinit(&ctxt, "C", proc_rows, proc_cols);
        } else ctxt = -1;
        mpi_free_comm(&active_front_comm);
      } else {
        ctxt = scalapack::Csys2blacs_handle(front_comm);
        scalapack::Cblacs_gridinit(&ctxt, "C", proc_rows, proc_cols);
      }
      ctxt_all = scalapack::Csys2blacs_handle(front_comm);
      scalapack::Cblacs_gridinit(&ctxt_all, "R", 1, total_procs);
    } else ctxt = ctxt_all = -1;
  }

  template<typename scalar_t,typename integer_t>
  FrontalMatrixMPI<scalar_t,integer_t>::~FrontalMatrixMPI() {
    if (ctxt != -1) scalapack::Cblacs_gridexit(ctxt);
    if (ctxt_all != -1) scalapack::Cblacs_gridexit(ctxt_all);
    mpi_free_comm(&front_comm);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixMPI<scalar_t,integer_t>::extract_2d
  (const SpMat_t& A, const std::vector<std::size_t>& I,
   const std::vector<std::size_t>& J, DistM_t& B) const {
    auto m = I.size();
    auto n = J.size();
    TIMER_TIME(TaskType::EXTRACT_SEP_2D, 2, t_ex_sep);
    {
      DistM_t tmp(ctxt, m, n);
      A.extract_separator_2d(this->sep_end, I, J, tmp, front_comm);
      // TODO why this copy???
      strumpack::copy(m, n, tmp, 0, 0, B, 0, 0, ctxt_all);
    }
    TIMER_STOP(t_ex_sep);
    TIMER_TIME(TaskType::GET_SUBMATRIX_2D, 2, t_getsub);
    DistM_t Bl, Br;
    get_child_submatrix_2d(this->lchild, I, J, Bl);
    get_child_submatrix_2d(this->rchild, I, J, Br);
    DistM_t tmp(B.ctxt(), m, n);
    strumpack::copy(m, n, Bl, 0, 0, tmp, 0, 0, ctxt_all);
    B.add(tmp);
    strumpack::copy(m, n, Br, 0, 0, tmp, 0, 0, ctxt_all);
    B.add(tmp);
    TIMER_STOP(t_getsub);
  }

  // this should not be necessary with proper polymorphic code
  template<typename scalar_t,typename integer_t> void
  FrontalMatrixMPI<scalar_t,integer_t>::get_child_submatrix_2d
  (const F_t* ch, const std::vector<std::size_t>& I,
   const std::vector<std::size_t>& J, DistM_t& B) const {
    if (!ch) return;
    auto m = I.size();
    auto n = J.size();
    if (auto mpi_child = dynamic_cast<const FMPI_t*>(ch)) {
      // TODO check if this really needs to be the child context,
      // maybe we can do directly to this context??
      // check in both FrontalMatrixDenseMPI and HSSMPI
      B = DistM_t(mpi_child->blacs_context(), m, n);
      B.zero();
      mpi_child->extract_CB_sub_matrix_2d(I, J, B);
    } else {
      TIMER_TIME(TaskType::GET_SUBMATRIX, 2, t_getsub);
      auto pch = child_master(ch);
      B = DistM_t(ctxt, m, n);
      DenseM_t lB;
      if (mpi_rank(front_comm) == pch) {
        lB = DenseM_t(m, n);
        lB.zero();
        ch->extract_CB_sub_matrix(I, J, lB, 0);
      }
      strumpack::copy(m, n, lB, pch, B, 0, 0, ctxt_all);
    }
  }

  /** return the rank in front_comm where element r,c in matrix F is
      located */
  template<typename scalar_t,typename integer_t> inline int
  FrontalMatrixMPI<scalar_t,integer_t>::find_rank
  (integer_t r, integer_t c, const DistM_t& F) const {
    // the blacs grid is column major
    return ((r / F.MB()) % proc_rows) + ((c / F.NB()) % proc_cols)
      * proc_rows;
  }

  template<typename scalar_t,typename integer_t> inline int
  FrontalMatrixMPI<scalar_t,integer_t>::find_rank_fixed
  (integer_t r, integer_t c, const DistM_t& F) const {
    assert(F.fixed());
    return ((r / DistM_t::default_MB) % proc_rows)
      + ((c / DistM_t::default_NB) % proc_cols) * proc_rows;
  }

  /**
   * Check if the child needs to be visited not necessary when this rank
   * is not part of the processes assigned to the child.
   */
  template<typename scalar_t,typename integer_t> bool
  FrontalMatrixMPI<scalar_t,integer_t>::visit(const F_t* ch) const {
    if (!ch) return false;
    if (auto mpi_child = dynamic_cast<const FMPI_t*>(ch)) {
      if (mpi_child->front_comm == MPI_COMM_NULL)
        return false; // child is MPI
    } else if (mpi_rank(front_comm) != child_master(ch))
      return false; // child is sequential
    return true;
  }

  template<typename scalar_t,typename integer_t> int
  FrontalMatrixMPI<scalar_t,integer_t>::child_master(const F_t* ch) const {
    int ch_master;
    if (auto mpi_ch = dynamic_cast<const FMPI_t*>(ch))
      ch_master = (ch == this->lchild) ? 0 :
        total_procs - mpi_ch->total_procs;
    else ch_master = (ch == this->lchild) ? 0 : total_procs - 1;
    return ch_master;
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixMPI<scalar_t,integer_t>::extend_add_b
  (DistM_t& b, DistM_t& bupd, const DistM_t& CBl, const DistM_t& CBr,
   const DenseM_t& seqCBl, const DenseM_t& seqCBr) const {
    if (mpi_rank(front_comm) == 0) {
      STRUMPACK_FLOPS
        (static_cast<long long int>(CBl.rows()*b.cols()));
      STRUMPACK_FLOPS
        (static_cast<long long int>(CBr.rows()*b.cols()));
    }
    auto P = mpi_nprocs(this->front_comm);
    std::vector<std::vector<scalar_t>> sbuf(P);
    if (this->visit(this->lchild)) {
      if (this->lchild->isMPI())
        ExtAdd::extend_add_column_copy_to_buffers
          (CBl, sbuf, this, this->lchild->upd_to_parent(this));
      else ExtAdd::extend_add_column_seq_copy_to_buffers
             (seqCBl, sbuf, this, this->lchild);
    }
    if (this->visit(this->rchild)) {
      if (this->rchild->isMPI())
        ExtAdd::extend_add_column_copy_to_buffers
          (CBr, sbuf, this, this->rchild->upd_to_parent(this));
      else ExtAdd::extend_add_column_seq_copy_to_buffers
             (seqCBr, sbuf, this, this->rchild);
    }
    scalar_t *rbuf = nullptr, **pbuf = nullptr;
    all_to_all_v(sbuf, rbuf, pbuf, this->front_comm);
    for (auto ch : {this->lchild, this->rchild}) {
      if (!ch) continue;
      if (auto ch_mpi = dynamic_cast<FMPI_t*>(ch))
        ExtAdd::extend_add_column_copy_from_buffers
          (b, bupd, pbuf+this->child_master(ch), this, ch_mpi);
      else ExtAdd::extend_add_column_seq_copy_from_buffers
             (b, bupd, pbuf[this->child_master(ch)], this, ch);
    }
    delete[] pbuf;
    delete[] rbuf;
  }


  template<typename scalar_t,typename integer_t> void
  FrontalMatrixMPI<scalar_t,integer_t>::extract_b
  (const DistM_t& b, const DistM_t& bupd, DistM_t& CBl, DistM_t& CBr,
   DenseM_t& seqCBl, DenseM_t& seqCBr) const {
    auto P = mpi_nprocs(this->front_comm);
    std::vector<std::vector<scalar_t>> sbuf(P);
    for (auto ch : {this->lchild, this->rchild}) {
      if (!ch) continue;
      if (auto ch_mpi = dynamic_cast<FMPI_t*>(ch))
        ExtAdd::extract_column_copy_to_buffers(b, bupd, sbuf, this, ch_mpi);
      else ExtAdd::extract_column_seq_copy_to_buffers
             (b, bupd, sbuf[this->child_master(ch)], this, ch);
    }
    scalar_t *rbuf = nullptr, **pbuf = nullptr;
    all_to_all_v(sbuf, rbuf, pbuf, this->front_comm);
    if (visit(this->lchild)) {
      if (auto ch_mpi = dynamic_cast<FMPI_t*>(this->lchild)) {
        CBl = DistM_t
          (ch_mpi->ctxt, this->lchild->dim_upd(), b.cols());
        ExtAdd::extract_column_copy_from_buffers
          (CBl, pbuf, this, this->lchild);
      } else {
        seqCBl = DenseM_t(this->lchild->dim_upd(), b.cols());
        ExtAdd::extract_column_seq_copy_from_buffers
          (seqCBl, pbuf, this, this->lchild);
      }
    }
    if (visit(this->rchild)) {
      if (auto ch_mpi = dynamic_cast<FMPI_t*>(this->rchild)) {
        CBr = DistM_t
          (ch_mpi->ctxt, this->rchild->dim_upd(), b.cols());
        ExtAdd::extract_column_copy_from_buffers(CBr, pbuf, this, this->rchild);
      } else {
        seqCBr = DenseM_t(this->rchild->dim_upd(), b.cols());
        ExtAdd::extract_column_seq_copy_from_buffers
          (seqCBr, pbuf, this, this->rchild);
      }
    }
    delete[] pbuf;
    delete[] rbuf;
  }


  // template<typename scalar_t,typename integer_t> void
  // FrontalMatrixMPI<scalar_t,integer_t>::extract_b
  // (const F_t* ch, const DistM_t& b, const DistM_t& bupd,
  //  DistM_t& CB) const {
  //   if (mpi_rank(front_comm) == 0) {
  //     STRUMPACK_FLOPS
  //       (static_cast<long long int>(ch->dim_upd())*ch->dim_upd());
  //   }
  //   if (auto mpi_child = dynamic_cast<FMPI_t*>(ch))
  //     extract_b_mpi_to_mpi(mpi_child, b, bupd, CB); // child is MPI
  //   else extract_b_seq_to_mpi(ch, b, bupd, CB);     // child is sequential
  // }

  // template<typename scalar_t,typename integer_t> void
  // FrontalMatrixMPI<scalar_t,integer_t>::extract_b_mpi_to_mpi
  // (FMPI_t* ch, DistM_t& b, scalar_t* wmem) {
  //   DistMW_t CB, bupd;
  //   if (visit(ch))
  //     CB = DistMW_t(ch->ctxt, ch->dim_upd(), 1, wmem+ch->p_wmem);
  //   if (front_comm != MPI_COMM_NULL)
  //     bupd = DistMW_t(ctxt, this->dim_upd(), 1, wmem+this->p_wmem);
  //   std::vector<MPI_Request> sreq;
  //   std::vector<std::vector<scalar_t>> sbuf;
  //   auto I = ch->upd_to_parent(this);
  //   if (b.pcol() == 0) {
  //     // send to all active processes in the child (only 1 column)
  //     sbuf.resize(ch->proc_rows);
  //     std::function<int(integer_t)> ch_rank = [&](integer_t r) {
  //       return ch->find_rank(r, 0, CB);
  //     };
  //     ExtAdd::extract_b_copy_to_buffers
  //       (b, bupd, sbuf, ch_rank, I, ch->proc_rows);
  //     sreq.resize(ch->proc_rows);
  //     for (int p=0; p<ch->proc_rows; p++)
  //       MPI_Isend(sbuf[p].data(), sbuf[p].size(), mpi_type<scalar_t>(),
  //                 p+child_master(ch), 0, front_comm, &sreq[p]);
  //   }
  //   if (CB.pcol() == 0) {
  //     // receive from all active processes in the parent (only 1 column)
  //     std::vector<std::vector<scalar_t>> rbuf(proc_rows);
  //     MPI_Status status;
  //     int msg_size;
  //     for (int p=0; p<proc_rows; p++) {
  //       MPI_Probe(MPI_ANY_SOURCE, 0, front_comm, &status);
  //       MPI_Get_count(&status, mpi_type<scalar_t>(), &msg_size);
  //       rbuf[status.MPI_SOURCE].resize(msg_size);
  //       MPI_Recv(rbuf[status.MPI_SOURCE].data(), msg_size,
  //                mpi_type<scalar_t>(), status.MPI_SOURCE, 0,
  //                front_comm, &status);
  //     }
  //     std::function<int(integer_t)> src_rank = [&](integer_t r) {
  //       return (r < this->dim_sep()) ? find_rank(r, 0, b) :
  //       find_rank(r-this->dim_sep(), 0, bupd);
  //     };
  //     ExtAdd::extract_b_copy_from_buffers(CB, rbuf, I, src_rank);
  //   }
  //   if (b.pcol() == 0)
  //     MPI_Waitall(sreq.size(), sreq.data(), MPI_STATUSES_IGNORE);
  // }

  // template<typename scalar_t,typename integer_t> void
  // FrontalMatrixMPI<scalar_t,integer_t>::extract_b_seq_to_mpi
  // (F_t* ch, DistM_t& b, scalar_t* wmem) {
  //   int child_rank = child_master(ch);
  //   scalar_t* CB = wmem+ch->p_wmem;
  //   DistMW_t bupd;
  //   if (front_comm != MPI_COMM_NULL)
  //     bupd = DistMW_t(ctxt, this->dim_upd(), 1, wmem+this->p_wmem);
  //   MPI_Request sreq;
  //   std::vector<std::vector<scalar_t>> sbuf(1);
  //   auto I = ch->upd_to_parent(this);
  //   if (b.pcol() == 0) {
  //     // send to all active processes in the child (only 1 column)
  //     std::function<int(integer_t)> dest_rank = [&](integer_t){ return 0; };
  //     ExtAdd::extract_b_copy_to_buffers(b, bupd, sbuf, dest_rank, I, 1);
  //     MPI_Isend(sbuf[0].data(), sbuf[0].size(), mpi_type<scalar_t>(),
  //               child_rank, 0, front_comm, &sreq);
  //   }
  //   if (mpi_rank(front_comm) == child_rank) {
  //     // receive from all active processes in the parent (only 1 column)
  //     std::vector<std::vector<scalar_t>> rbuf(proc_rows);
  //     MPI_Status status;
  //     int msg_size;
  //     for (int p=0; p<proc_rows; p++) {
  //       MPI_Probe(MPI_ANY_SOURCE, 0, front_comm, &status);
  //       MPI_Get_count(&status, mpi_type<scalar_t>(), &msg_size);
  //       rbuf[status.MPI_SOURCE].resize(msg_size);
  //       MPI_Recv(rbuf[status.MPI_SOURCE].data(), msg_size,
  //                mpi_type<scalar_t>(), status.MPI_SOURCE, 0,
  //                front_comm, &status);
  //     }
  //     std::vector<scalar_t*> pbuf(rbuf.size());
  //     for (std::size_t p=0; p<rbuf.size(); p++)
  //       pbuf[p] = rbuf[p].data();
  //     for (integer_t r=0; r<ch->dim_upd(); r++) {
  //       integer_t pa_r = I[r];
  //       integer_t rank = (pa_r < this->dim_sep()) ? find_rank(pa_r, 0, b)
  //         : find_rank(pa_r-this->dim_sep(), 0, bupd);
  //       CB[r] = *(pbuf[rank]);
  //       pbuf[rank]++;
  //     }
  //   }
  //   if (b.pcol() == 0) MPI_Wait(&sreq, MPI_STATUS_IGNORE);
  // }

  template<typename scalar_t,typename integer_t> long long
  FrontalMatrixMPI<scalar_t,integer_t>::dense_factor_nonzeros
  (int task_depth) const {
    long long nnz = 0;
    if (this->front_comm != MPI_COMM_NULL &&
        mpi_rank(this->front_comm) == 0) {
      auto dsep = this->dim_sep();
      auto dupd = this->dim_upd();
      nnz = dsep * (dsep + 2 * dupd);
    }
    if (visit(this->lchild))
      nnz += this->lchild->dense_factor_nonzeros(task_depth);
    if (visit(this->rchild))
      nnz += this->rchild->dense_factor_nonzeros(task_depth);
    return nnz;
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixMPI<scalar_t,integer_t>::bisection_partitioning
  (const SPOptions<scalar_t>& opts, integer_t* sorder,
   bool isroot, int task_depth) {
    for (integer_t i=this->sep_begin; i<this->sep_end; i++) sorder[i] = -i;

    if (visit(this->lchild))
      this->lchild->bisection_partitioning(opts, sorder, false, task_depth);
    if (visit(this->rchild))
      this->rchild->bisection_partitioning(opts, sorder, false, task_depth);
  }

} // end namespace strumpack

#endif //FRONTAL_MATRIX_MPI_HPP
