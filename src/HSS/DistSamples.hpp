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
 */
#ifndef DIST_SAMPLES_HPP
#define DIST_SAMPLES_HPP

#include "HSSOptions.hpp"
#include "HSS/HSSMatrix.sketch.hpp"

namespace strumpack {
  namespace HSS {

    template<typename scalar_t> class HSSMatrixMPI;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    template<typename scalar_t> class DistSamples {
      using real_t = typename RealType<scalar_t>::value_type;
      using DistM_t = DistributedMatrix<scalar_t>;
      using DistMW_t = DistributedMatrixWrapper<scalar_t>;
      using DenseM_t = DenseMatrix<scalar_t>;
      using dmult_t = typename std::function
        <void(DistM_t& R, DistM_t& Sr, DistM_t& Sc)>;
      using opts_t = HSSOptions<scalar_t>;

    public:
      DistSamples(int d, const BLACSGrid* g, HSSMatrixMPI<scalar_t>& hss,
                  const dmult_t& Amult, const opts_t& opts,
                  bool hard_restart=false)
        : DistSamples(d, g, hss, opts, hard_restart) {
        Amult_ = &Amult;
        init(opts);
      }

      DistSamples(int d, const DistM_t& A, HSSMatrixMPI<scalar_t>& hss,
                  const opts_t& opts, bool hard_restart=false)
        : DistSamples(d, A.grid(), hss, opts, hard_restart) {
        A_= &A;
        init(opts);
      }

      void init(const opts_t& opts) {
        if (Amult_) {
          std::cout << "PDGEMM/Amult sampling" << std::endl;
          rgen_->seed(R.prow(), R.pcol());
          R.random(*rgen_);
          STRUMPACK_RANDOM_FLOPS
            (rgen_->flops_per_prng() * R.lrows() * R.lcols());
          (*Amult_)(R, Sr, Sc);
          hss_.to_block_row(R,  sub_Rr, leaf_R);
          sub_Rc = sub_Rr;
          hss_.to_block_row(Sr, sub_Sr, leaf_Sr);
          hss_.to_block_row(Sc, sub_Sc, leaf_Sc);
        } else {
          std::cout << "block row sampling!!" << std::endl;
          DenseM_t sub_A;
          DistM_t leaf_A;
          hss_.to_block_row(*A_, sub_A, leaf_A);
          if (leaf_A.rows())
            std::cout << "ERROR: Not supported. "
                      << " Make sure there are more MPI ranks than HSS leafs."
                      << std::endl;

          auto n = R.rows();
          auto d = R.cols();

          sub_Sr = DenseM_t(sub_A.rows(), d);
#if 1
          bool chunk = opts.SJLT_algo() == SJLTAlgo::CHUNK;
          SJLTGenerator<scalar_t,int> g;
          SJLTMatrix<scalar_t,int> S_sjlt(g, 0, n, 0, chunk);
          S_sjlt.add_columns(d, opts.nnz0());
          auto dup_R = S_sjlt.to_dense();
          matrix_times_SJLT(sub_A, S_sjlt, sub_Sr);
          // TODO transpose!
          // matrixT_times_SJLT(sub_A, S, Sc_new);
#else
          DenseM_t dup_R(n, d);
          rgen_->seed(0, 0);
          dup_R.random(*rgen_);
          // use regular gemm
          gemm(Trans::N, Trans::N, scalar_t(1.), sub_A, dup_R, scalar_t(0.), sub_Sr);
#endif

          // TODO transpose, assume for now matrix is symmetric
          sub_Sc = sub_Sr;

          auto rank = hss_.Comm().rank();
          auto r0 = hss_.tree_ranges().clo(rank);
          auto r1 = hss_.tree_ranges().chi(rank);
          sub_Rr = DenseM_t(r1-r0, d, dup_R, r0, 0);
          sub_Rc = sub_Rr;
        }
        if (hard_restart_) { // copies for when doing a hard restart
          sub_R2 = sub_Rr;
          sub_Sr2 = sub_Sr;
          sub_Sc2 = sub_Sc;
        }
      }

      const HSSMatrixMPI<scalar_t>& HSS() const { return hss_; }

      void add_columns(int d, const opts_t& opts) {

        auto n = R.rows();
        auto d_old = R.cols();
        auto dd = d-d_old;
        DistM_t Rnew(R.grid(), n, dd);
        Rnew.random(*rgen_);
        STRUMPACK_RANDOM_FLOPS
          (rgen_->flops_per_prng() * Rnew.lrows() * Rnew.lcols());
        DistM_t Srnew(Sr.grid(), n, dd);
        DistM_t Scnew(Sc.grid(), n, dd);

        // sample(Rnew, Srnew, Scnew);
        std::cout << "TODO Adding cols" << std::endl;

        R.hconcat(Rnew);
        Sr.hconcat(Srnew);
        Sc.hconcat(Scnew);
        DenseM_t subRnew, subSrnew, subScnew;
        DistM_t leafRnew, leafSrnew, leafScnew;
        hss_.to_block_row(Rnew,  subRnew, leafRnew);
        hss_.to_block_row(Srnew, subSrnew, leafSrnew);
        hss_.to_block_row(Scnew, subScnew, leafScnew);
        if (hard_restart_) {
          sub_Rr = hconcat(sub_R2,  subRnew);
          sub_Rc = hconcat(sub_R2,  subRnew);
          sub_Sr = hconcat(sub_Sr2, subSrnew);
          sub_Sc = hconcat(sub_Sc2, subScnew);
          sub_R2  = sub_Rr;
          sub_Sr2 = sub_Sr;
          sub_Sc2 = sub_Sc;
        } else {
          sub_Rr.hconcat(subRnew);
          sub_Rc.hconcat(subRnew);
          sub_Sr.hconcat(subSrnew);
          sub_Sc.hconcat(subScnew);
        }
        leaf_R.hconcat(leafRnew);
        leaf_Sr.hconcat(leafSrnew);
        leaf_Sc.hconcat(leafScnew);
      }

      // no need to store all these?
      DistM_t R, Sr, Sc, leaf_R, leaf_Sr, leaf_Sc;
      DenseM_t sub_Rr, sub_Rc, sub_Sr, sub_Sc;
      DenseM_t sub_R2, sub_Sr2, sub_Sc2;

    private:
      DistSamples(int d, const BLACSGrid* g, HSSMatrixMPI<scalar_t>& hss,
                  const opts_t& opts, bool hard_restart) :
        // TODO no need to construct R, Sr, and Sc when doing block
        // row multiply with SJLT
        R(g, hss.cols(), d), Sr(g, hss.cols(), d), Sc(g, hss.cols(), d),
        hss_(hss),
        rgen_(random::make_random_generator<real_t>
              (opts.random_engine(), opts.random_distribution())),
        hard_restart_(hard_restart) { }

      const dmult_t* Amult_ = nullptr;
      const DistM_t* A_ = nullptr;

      const HSSMatrixMPI<scalar_t>& hss_;
      std::unique_ptr<random::RandomGeneratorBase<real_t>> rgen_;
      bool hard_restart_ = false;
    };

#endif // DOXYGEN_SHOULD_SKIP_THIS

  } // end namespace HSS
} // end namespace strumpack

#endif // DIST_SAMPLES_HPP
