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
    private:
      const dmult_t& _Amult;
      const HSSMatrixMPI<scalar_t>& _hss;
      std::unique_ptr<random::RandomGeneratorBase<real_t>> _rgen;
      bool _hard_restart = false;
    public:
      DistM_t R, Sr, Sc, leaf_R, leaf_Sr, leaf_Sc;
      DenseM_t sub_Rr, sub_Rc, sub_Sr, sub_Sc;
      DenseM_t sub_R2, sub_Sr2, sub_Sc2;
      DistSamples(int d, const BLACSGrid* g, HSSMatrixMPI<scalar_t>& hss,
                  const dmult_t& Amult, const opts_t& opts,
                  bool hard_restart=false)
        : _Amult(Amult), _hss(hss),
          _rgen(random::make_random_generator<real_t>
                (opts.random_engine(), opts.random_distribution())),
          _hard_restart(hard_restart),
          R(g, _hss.cols(), d), Sr(g, _hss.cols(), d),
          Sc(g, _hss.cols(), d) {
        _rgen->seed(R.prow(), R.pcol());
        R.random(*_rgen);
        STRUMPACK_RANDOM_FLOPS
          (_rgen->flops_per_prng() * R.lrows() * R.lcols());
        _Amult(R, Sr, Sc);
        _hss.to_block_row(R,  sub_Rr, leaf_R);
        sub_Rc = DenseM_t(sub_Rr);
        _hss.to_block_row(Sr, sub_Sr, leaf_Sr);
        _hss.to_block_row(Sc, sub_Sc, leaf_Sc);
        if (_hard_restart) { // copies for when doing a hard restart
          sub_R2 = sub_Rr;
          sub_Sr2 = sub_Sr;
          sub_Sc2 = sub_Sc;
        }
      }
      const HSSMatrixMPI<scalar_t>& HSS() const { return _hss; }
      void add_columns(int d, const opts_t& opts) {
        auto n = R.rows();
        auto d_old = R.cols();
        auto dd = d-d_old;
        DistM_t Rnew(R.grid(), n, dd);
        Rnew.random(*_rgen);
        STRUMPACK_RANDOM_FLOPS
          (_rgen->flops_per_prng() * Rnew.lrows() * Rnew.lcols());
        DistM_t Srnew(Sr.grid(), n, dd);
        DistM_t Scnew(Sc.grid(), n, dd);
        _Amult(Rnew, Srnew, Scnew);
        R.hconcat(Rnew);
        Sr.hconcat(Srnew);
        Sc.hconcat(Scnew);
        DenseM_t subRnew, subSrnew, subScnew;
        DistM_t leafRnew, leafSrnew, leafScnew;
        _hss.to_block_row(Rnew,  subRnew, leafRnew);
        _hss.to_block_row(Srnew, subSrnew, leafSrnew);
        _hss.to_block_row(Scnew, subScnew, leafScnew);
        if (_hard_restart) {
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
    };
#endif // DOXYGEN_SHOULD_SKIP_THIS

  } // end namespace HSS
} // end namespace strumpack

#endif // DIST_SAMPLES_HPP
