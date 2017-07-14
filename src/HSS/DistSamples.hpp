#ifndef DIST_SAMPLES_HPP
#define DIST_SAMPLES_HPP

#include "HSSOptions.hpp"

namespace strumpack {
  namespace HSS {

    template<typename scalar_t> class HSSMatrixMPI;

    template<typename scalar_t> class DistSamples {
      using real_t = typename RealType<scalar_t>::value_type;
      using DistM_t = DistributedMatrix<scalar_t>;
      using DistMW_t = DistributedMatrixWrapper<scalar_t>;
      using DenseM_t = DenseMatrix<scalar_t>;
      using dmult_t = typename std::function<void(DistM_t& R, DistM_t& Sr, DistM_t& Sc)>;
      using opts_t = HSSOptions<scalar_t>;
    private:
      const dmult_t& _Amult;
      const HSSMatrixMPI<scalar_t>& _hss;
      std::unique_ptr<random::RandomGeneratorBase<real_t>> _rgen;
    public:
      DistM_t R, Sr, Sc, leaf_R, leaf_Sr, leaf_Sc;
      DenseM_t sub_Rr, sub_Rc, sub_Sr, sub_Sc;
      DistSamples(int d, int Actxt, HSSMatrixMPI<scalar_t>& hss, const dmult_t& Amult, const opts_t& opts)
	: _Amult(Amult), _hss(hss),
	  _rgen(random::make_random_generator<real_t>
		(mpi_rank(hss.comm())+1, opts.random_engine(), opts.random_distribution())),
	  R(Actxt, _hss.cols(), d), Sr(Actxt, _hss.cols(), d), Sc(Actxt, _hss.cols(), d) {
	R.random(*_rgen);
	_Amult(R, Sr, Sc);
	_hss.to_block_row(R,  sub_Rr, leaf_R);
	sub_Rc = DenseM_t(sub_Rr);
	_hss.to_block_row(Sr, sub_Sr, leaf_Sr);
	_hss.to_block_row(Sc, sub_Sc, leaf_Sc);
      }
      const HSSMatrixMPI<scalar_t>& HSS() const { return _hss; }
      void add_columns(int d, const opts_t& opts) {
	auto n = R.rows();
	auto d_old = R.cols();
	auto dd = d-d_old;
	DistM_t Rnew(R.ctxt(), n, dd);
	Rnew.random(*_rgen);
	DistM_t Srnew(Sr.ctxt(), n, dd);
	DistM_t Scnew(Sc.ctxt(), n, dd);
	_Amult(Rnew, Srnew, Scnew);
	R.hconcat(Rnew);
	Sr.hconcat(Srnew);
	Sc.hconcat(Scnew);
	DenseM_t subRnew, subSrnew, subScnew;
	DistM_t leafRnew, leafSrnew, leafScnew;
	_hss.to_block_row(Rnew,  subRnew, leafRnew);
	_hss.to_block_row(Srnew, subSrnew, leafSrnew);
	_hss.to_block_row(Scnew, subScnew, leafScnew);
	sub_Rr.hconcat(subRnew);
	sub_Rc.hconcat(subRnew);
	sub_Sr.hconcat(subSrnew);
	sub_Sc.hconcat(subScnew);
	leaf_R.hconcat(leafRnew);
	leaf_Sr.hconcat(leafSrnew);
	leaf_Sc.hconcat(leafScnew);
      }
    };

  } // end namespace HSS
} // end namespace strumpack

#endif // DIST_SAMPLES_HPP
