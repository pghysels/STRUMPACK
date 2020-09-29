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

#include "FrontalMatrixMPI.hpp"
#include "FrontalMatrixBLRMPI.hpp"
#include "ExtendAdd.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t>
  FrontalMatrixMPI<scalar_t,integer_t>::FrontalMatrixMPI
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd, const MPIComm& comm, int P)
    : F_t(nullptr, nullptr, sep, sep_begin, sep_end, upd),
      blacs_grid_(comm, P) {
  }

  template<typename scalar_t,typename integer_t> integer_t
  FrontalMatrixMPI<scalar_t,integer_t>::maximum_rank(int task_depth) const {
    auto mr = this->front_rank();
    if (visit(lchild_))
      mr = std::max(mr, lchild_->maximum_rank(task_depth));
    if (visit(rchild_))
      mr = std::max(mr, rchild_->maximum_rank(task_depth));
    return mr;
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixMPI<scalar_t,integer_t>::extract_2d
  (const SpMat_t& A, const VecVec_t& I, const VecVec_t& J,
   std::vector<DistMW_t>& B, bool skip_sparse) const {
    TIMER_TIME(TaskType::EXTRACT_2D, 1, t_ex);
    int rank = Comm().rank(), nprocs = P();
    std::vector<DistM_t> Bl, Br;
    std::vector<DenseM_t> Blseq, Brseq;
    if (visit(lchild_)) lchild_->get_submatrix_2d(I, J, Bl, Blseq);
    if (visit(rchild_)) rchild_->get_submatrix_2d(I, J, Br, Brseq);
    std::vector<int> bgrid(3*I.size(), -1);
    for (std::size_t i=0; i<I.size(); i++)
      if (B[i].grid() && B[i].prow() == 0 && B[i].pcol() == 0) {
        bgrid[3*i] = rank;
        bgrid[3*i+1] = B[i].nprows();
        bgrid[3*i+2] = B[i].npcols();
      }
    Comm().all_reduce(bgrid.data(), bgrid.size(), MPI_MAX);
    std::vector<std::vector<scalar_t>> sbuf(nprocs);
    std::size_t ssize = 0;
    for (std::size_t i=0; i<I.size(); i++)
      ssize += I[i].size() * J[i].size();
    for (int p=0; p<nprocs; p++)
      sbuf[p].reserve(std::round(1.2*float(ssize)/nprocs));
    bool vl = visit(lchild_), vr = visit(rchild_);
    for (std::size_t i=0; i<I.size(); i++) {
      if (!skip_sparse) {
        TIMER_TIME(TaskType::EXTRACT_SEP_2D, 2, t_ex_sep);
        DistM_t tmp(grid(), I[i].size(), J[i].size());
        A.extract_separator_2d(this->sep_end_, I[i], J[i], tmp);
        subgrid_copy_to_buffers
          (tmp, B[i], bgrid[3*i], bgrid[3*i+1], bgrid[3*i+2], sbuf);
      }
      if (vl) {
        if (lchild_->isMPI())
          subgrid_copy_to_buffers
            (Bl[i], B[i], bgrid[3*i], bgrid[3*i+1], bgrid[3*i+2], sbuf);
        else
          subproc_copy_to_buffers
            (Blseq[i], B[i], bgrid[3*i], bgrid[3*i+1], bgrid[3*i+2], sbuf);
      }
      if (vr) {
        if (rchild_->isMPI())
          subgrid_copy_to_buffers
            (Br[i], B[i], bgrid[3*i], bgrid[3*i+1], bgrid[3*i+2], sbuf);
        else
          subproc_copy_to_buffers
            (Brseq[i], B[i], bgrid[3*i], bgrid[3*i+1], bgrid[3*i+2], sbuf);
      }
    }
    std::vector<scalar_t,NoInit<scalar_t>> rbuf;
    std::vector<scalar_t*> pbuf;
    {
      TIMER_TIME(TaskType::EXTRACT_2D_A2A, 1, t_a2a);
      Comm().all_to_all_v(sbuf, rbuf, pbuf);
    }
    BLACSGrid *gl = nullptr, *gr = nullptr;
    if (lchild_) gl = lchild_->grid();
    if (rchild_) gr = rchild_->grid();
    int ml = master(lchild_), mr = master(rchild_);
    for (std::size_t i=0; i<I.size(); i++) {
      if (!B[i].active()) continue;
      B[i].zero();
      if (!skip_sparse) subgrid_add_from_buffers(grid(), 0, B[i], pbuf);
      if (lchild_) subgrid_add_from_buffers(gl, ml, B[i], pbuf);
      if (rchild_) subgrid_add_from_buffers(gr, mr, B[i], pbuf);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixMPI<scalar_t,integer_t>::get_submatrix_2d
  (const VecVec_t& I, const VecVec_t& J,
   std::vector<DistM_t>& Bdist, std::vector<DenseM_t>&) const {
    TIMER_TIME(TaskType::GET_SUBMATRIX_2D, 3, t_ex_schur);
    for (std::size_t i=0; i<I.size(); i++) {
      Bdist.emplace_back(grid(), I[i].size(), J[i].size());
      Bdist.back().zero();
    }
    extract_CB_sub_matrix_2d(I, J, Bdist);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixMPI<scalar_t,integer_t>::extract_CB_sub_matrix_2d
  (const VecVec_t& I, const VecVec_t& J, std::vector<DistM_t>& B) const {
    for (std::size_t i=0; i<I.size(); i++)
      extract_CB_sub_matrix_2d(I[i], J[i], B[i]);
  }

  /**
   * Check if the child needs to be visited. Not necessary when this
   * rank is not part of the processes assigned to the child.
   */
  template<typename scalar_t,typename integer_t> bool
  FrontalMatrixMPI<scalar_t,integer_t>::visit
  (const F_t* ch) const {
    if (!ch) return false;
    // TODO do this differently!!
    if (auto mpi_child = dynamic_cast<const FMPI_t*>(ch)) {
      if (mpi_child->Comm().is_null())
        return false; // child is MPI
    } else if (Comm().rank() != master(ch))
      return false; // child is sequential
    return true;
  }

  template<typename scalar_t,typename integer_t> bool
  FrontalMatrixMPI<scalar_t,integer_t>::visit
  (const std::unique_ptr<F_t>& ch) const {
    return visit(ch.get());
  }

  template<typename scalar_t,typename integer_t> int
  FrontalMatrixMPI<scalar_t,integer_t>::master
  (const std::unique_ptr<F_t>& ch) const {
    return master(ch.get());
  }

  template<typename scalar_t,typename integer_t> int
  FrontalMatrixMPI<scalar_t,integer_t>::master(const F_t* ch) const {
    return (ch == lchild_.get()) ? 0 : P() - ch->P();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixMPI<scalar_t,integer_t>::extend_add_b
  (DistM_t& b, DistM_t& bupd, const DistM_t& CBl, const DistM_t& CBr,
   const DenseM_t& seqCBl, const DenseM_t& seqCBr) const {
    if (Comm().is_root()) {
      STRUMPACK_FLOPS(static_cast<long long int>(CBl.rows()*b.cols()));
      STRUMPACK_FLOPS(static_cast<long long int>(CBr.rows()*b.cols()));
    }
    std::vector<std::vector<scalar_t>> sbuf(P());
    if (visit(lchild_))
      lchild_->extend_add_column_copy_to_buffers(CBl, seqCBl, sbuf, this);
    if (visit(rchild_))
      rchild_->extend_add_column_copy_to_buffers(CBr, seqCBr, sbuf, this);
    std::vector<scalar_t,NoInit<scalar_t>> rbuf;
    std::vector<scalar_t*> pbuf;
    Comm().all_to_all_v(sbuf, rbuf, pbuf);
    for (auto& ch : {lchild_.get(), rchild_.get()})
      if (ch) ch->extend_add_column_copy_from_buffers
                (b, bupd, pbuf.data()+master(ch), this);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixMPI<scalar_t,integer_t>::extract_from_R2D
  (const DistM_t& R, DistM_t& cR, DenseM_t& seqcR,
   const FMPI_t* pa, bool visit) const {
    auto I = this->upd_to_parent(pa);
    cR = DistM_t(grid(), I.size(), R.cols(),
                 R.extract_rows(I), pa->grid()->ctxt_all());
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixMPI<scalar_t,integer_t>::extract_b
  (const DistM_t& b, const DistM_t& bupd, DistM_t& CBl, DistM_t& CBr,
   DenseM_t& seqCBl, DenseM_t& seqCBr) const {
    std::vector<std::vector<scalar_t>> sbuf(P());
    for (auto ch : {lchild_.get(), rchild_.get()})
      if (ch) ch->extract_column_copy_to_buffers
                (b, bupd, master(ch), sbuf, this);
    std::vector<scalar_t,NoInit<scalar_t>> rbuf;
    std::vector<scalar_t*> pbuf;
    Comm().all_to_all_v(sbuf, rbuf, pbuf);
    if (visit(lchild_))
      lchild_->extract_column_copy_from_buffers(b, CBl, seqCBl, pbuf, this);
    if (visit(rchild_))
      rchild_->extract_column_copy_from_buffers(b, CBr, seqCBr, pbuf, this);
  }

  template<typename scalar_t,typename integer_t> long long
  FrontalMatrixMPI<scalar_t,integer_t>::dense_factor_nonzeros
  (int task_depth) const {
    long long nnz = 0;
    if (!Comm().is_null() && Comm().is_root()) {
      long long dsep = this->dim_sep();
      long long dupd = this->dim_upd();
      nnz = dsep * (dsep + 2 * dupd);
    }
    if (visit(lchild_))
      nnz += lchild_->dense_factor_nonzeros(task_depth);
    if (visit(rchild_))
      nnz += rchild_->dense_factor_nonzeros(task_depth);
    return nnz;
  }

  template<typename scalar_t,typename integer_t> long long
  FrontalMatrixMPI<scalar_t,integer_t>::factor_nonzeros
  (int task_depth) const {
    long long nnz = node_factor_nonzeros();
    if (visit(lchild_))
      nnz += lchild_->factor_nonzeros(task_depth);
    if (visit(rchild_))
      nnz += rchild_->factor_nonzeros(task_depth);
    return nnz;
  }

  template<typename scalar_t,typename integer_t> long long
  FrontalMatrixMPI<scalar_t,integer_t>::node_factor_nonzeros() const {
    long long dsep = this->dim_sep();
    long long dupd = this->dim_upd();
    return Comm().is_root() ? dsep * (dsep + 2 * dupd) : 0;
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixMPI<scalar_t,integer_t>::partition_fronts
  (const Opts_t& opts, const SpMat_t& A, integer_t* sorder,
   bool is_root, int task_depth) {
    if (visit(lchild_))
      lchild_->partition_fronts(opts, A, sorder, false, task_depth+1);
    if (visit(rchild_))
      rchild_->partition_fronts(opts, A, sorder, false, task_depth+1);
    this->partition(opts, A, sorder, is_root, task_depth);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixMPI<scalar_t,integer_t>::extend_add_copy_from_buffers
  (DistM_t& F11, DistM_t& F12, DistM_t& F21, DistM_t& F22, scalar_t** pbuf,
   const FrontalMatrixMPI<scalar_t,integer_t>* pa) const {
    ExtendAdd<scalar_t,integer_t>::extend_add_copy_from_buffers
      (F11, F12, F21, F22, pbuf, pa, this);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixMPI<scalar_t,integer_t>::extend_add_column_copy_to_buffers
  (const DistM_t& CB, const DenseM_t& seqCB,
   std::vector<std::vector<scalar_t>>& sbuf, const FMPI_t* pa) const {
    ExtendAdd<scalar_t,integer_t>::extend_add_column_copy_to_buffers
      (CB, sbuf, pa, this->upd_to_parent(pa));
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixMPI<scalar_t,integer_t>::extend_add_column_copy_from_buffers
  (DistM_t& B, DistM_t& Bupd, scalar_t** pbuf,
   const FrontalMatrixMPI<scalar_t,integer_t>* pa) const {
    ExtendAdd<scalar_t,integer_t>::extend_add_column_copy_from_buffers
      (B, Bupd, pbuf, pa, this);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixMPI<scalar_t,integer_t>::extract_column_copy_to_buffers
  (const DistM_t& b, const DistM_t& bupd, int ch_master,
   std::vector<std::vector<scalar_t>>& sbuf, const FMPI_t* pa) const {
    ExtendAdd<scalar_t,integer_t>::extract_column_copy_to_buffers
      (b, bupd, sbuf, pa, this);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixMPI<scalar_t,integer_t>::extract_column_copy_from_buffers
  (const DistM_t& b, DistM_t& CB, DenseM_t& seqCB,
   std::vector<scalar_t*>& pbuf, const FMPI_t* pa) const {
    CB = DistM_t(grid(), this->dim_upd(), b.cols());
    ExtendAdd<scalar_t,integer_t>::extract_column_copy_from_buffers
      (CB, pbuf, pa, this);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixMPI<scalar_t,integer_t>::skinny_ea_to_buffers
  (const DistM_t& S, const DenseM_t& seqS,
   std::vector<std::vector<scalar_t>>& sbuf, const FMPI_t* pa) const {
    ExtendAdd<scalar_t,integer_t>::skinny_extend_add_copy_to_buffers
      (S, sbuf, pa, this->upd_to_parent(pa));
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixMPI<scalar_t,integer_t>::skinny_ea_from_buffers
  (DistM_t& S, scalar_t** pbuf, const FMPI_t* pa) const {
    ExtendAdd<scalar_t,integer_t>::skinny_extend_add_copy_from_buffers
      (S, pbuf, pa, this);
  }

  // explicit template instantiations
  template class FrontalMatrixMPI<float,int>;
  template class FrontalMatrixMPI<double,int>;
  template class FrontalMatrixMPI<std::complex<float>,int>;
  template class FrontalMatrixMPI<std::complex<double>,int>;

  template class FrontalMatrixMPI<float,long int>;
  template class FrontalMatrixMPI<double,long int>;
  template class FrontalMatrixMPI<std::complex<float>,long int>;
  template class FrontalMatrixMPI<std::complex<double>,long int>;

  template class FrontalMatrixMPI<float,long long int>;
  template class FrontalMatrixMPI<double,long long int>;
  template class FrontalMatrixMPI<std::complex<float>,long long int>;
  template class FrontalMatrixMPI<std::complex<double>,long long int>;

} // end namespace strumpack
