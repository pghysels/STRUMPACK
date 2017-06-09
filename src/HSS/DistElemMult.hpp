#ifndef HSS_DISTELEMMULT_HPP
#define HSS_DISTELEMMULT_HPP

namespace strumpack {
  namespace HSS {

    template<typename scalar_t> class DistElemMult {
      using DistM_t = DistributedMatrix<scalar_t>;
      using DenseM_t = DenseMatrix<scalar_t>;
    public:
      // TODO this stores a copy of the global matrix on every
      // process! avoid this!!
      DistElemMult(const DistM_t& A, int ctxt_all, MPI_Comm comm)
	: _A(A), _Ag(A.all_gather(ctxt_all)), _ctxt(A.ctxt()), _ctxt_all(ctxt_all), _comm(comm) { }
      const DistM_t& _A;
      const DenseM_t _Ag;
      const int _ctxt, _ctxt_all;
      const MPI_Comm _comm;
      void operator()(DistM_t& R, DistM_t& Sr, DistM_t& Sc) {
	gemm(Trans::N, Trans::N, scalar_t(1.), _A, R, scalar_t(0.), Sr);
	gemm(Trans::C, Trans::N, scalar_t(1.), _A, R, scalar_t(0.), Sc);
      }
      void operator()(const std::vector<size_t>& I, const std::vector<size_t>& J, DistM_t& B) {
	if (!B.active()) return;
	assert(I.size() == B.rows() && J.size() == B.cols());
	for (std::size_t j=0; j<J.size(); j++)
	  for (std::size_t i=0; i<I.size(); i++) {
	    assert(I[i] >= 0 && I[i] < _A.rows() && J[j] >= 0 && J[j] < _A.cols());
	    B.global(i, j, _Ag(I[i], J[j]));
	  }
	// TODO just to make sure everyone calls this! test!!
	//MPI_Barrier(_comm);
      }
    };

    template<typename scalar_t> class LocalElemMult {
      using DistM_t = DistributedMatrix<scalar_t>;
      using DistMW_t = DistributedMatrixWrapper<scalar_t>;
      using DenseM_t = DenseMatrix<scalar_t>;
      using delem_t = typename std::function
	<void(const std::vector<std::size_t>& I, const std::vector<std::size_t>& J, DistM_t& B)>;
    public:
      LocalElemMult(const delem_t& Aelem, std::pair<std::size_t,std::size_t>& offset, int ctxt)
	: _dAelem(Aelem), _offset(offset), _ctxt_loc(ctxt) {}
      const delem_t& _dAelem;
      const std::pair<std::size_t,std::size_t> _offset;
      int _ctxt_loc;
      void operator()(const std::vector<size_t>& I, const std::vector<size_t>& J, DenseM_t& B) {
	std::vector<std::size_t> gI(I), gJ(J);
	for (auto& i : gI) i += _offset.first;
	for (auto& j : gJ) j += _offset.second;
	DistMW_t dB(_ctxt_loc, 0, 0, I.size(), J.size(), B);
	_dAelem(gI, gJ, dB);
      }
    };

  } // end namespace HSS
} // end namespace strumpack

#endif  // HSS_DISTELEMMULT_HPP
