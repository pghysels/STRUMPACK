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
#include <array>

#include "FrontDPCpp.hpp"

#if defined(STRUMPACK_USE_MPI)
#include "ExtendAdd.hpp"
#include "FrontalMatrixMPI.hpp"
#endif

namespace strumpack {

  template<typename T> struct AssembleData {
    AssembleData(std::size_t d1_, std::size_t d2_,
		 T* F11_, T* F12_, T* F21_, T* F22_,
		 std::size_t n11_, std::size_t n12_, std::size_t n21_,
		 Triplet<T>* e11_, Triplet<T>* e12_, Triplet<T>* e21_)
      : d1(d1_), d2(d2_), F11(F11_), F12(F12_), F21(F21_), F22(F22_),
	n11(n11_), n12(n12_), n21(n21_), e11(e11_), e12(e12_), e21(e21_) {}
    AssembleData(int d1_, int d2_, T* F11_, T* F21_)
      : d1(d1_), d2(d2_), F11(F11_), F21(F21_) {}

    // sizes and pointers for this front
    std::size_t d1 = 0, d2 = 0;
    T *F11 = nullptr, *F12 = nullptr, *F21 = nullptr, *F22 = nullptr;

    // info for extend add
    std::size_t dCB1 = 0, dCB2 = 0;
    T *CB1 = nullptr, *CB2 = nullptr;
    std::size_t *I1 = nullptr, *I2 = nullptr;

    // sparse matrix elements
    std::size_t n11 = 0, n12 = 0, n21 = 0;
    Triplet<T> *e11 = nullptr, *e12 = nullptr, *e21 = nullptr;

    void set_ext_add_left(std::size_t dCB, T* CB, std::size_t* I) {
      dCB1 = dCB;
      CB1 = CB;
      I1 = I;
    }
    void set_ext_add_right(std::size_t dCB, T* CB, std::size_t* I) {
      dCB2 = dCB;
      CB2 = CB;
      I2 = I;
    }
  };

  template<typename T> struct FrontData {
    FrontData() {}
    FrontData(int n1_, int n2_, T* F11_, T* F12_,
	      T* F21_, T* F22_, std::int64_t* piv_)
      : n1(n1_), n2(n2_), F11(F11_), F12(F12_),
	F21(F21_), F22(F22_), piv(piv_) {}
    int n1, n2;
    T *F11, *F12, *F21, *F22;
    std::int64_t* piv;
  };

  uintptr_t round_to_8(uintptr_t p) { return (p + 7) & ~7; }
  uintptr_t round_to_8(void* p) {
    return round_to_8(reinterpret_cast<uintptr_t>(p));
  }

  template<typename scalar_t, typename integer_t> class LevelInfo {
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using FDPC_t = FrontDPCpp<scalar_t,integer_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;

  public:
    LevelInfo() {}

    LevelInfo(const std::vector<F_t*>& fronts,
	      cl::sycl::queue& q) {
      f.reserve(fronts.size());
      for (auto& F : fronts)
        f.push_back(dynamic_cast<FDPC_t*>(F));
      for (auto F : f) {
        auto dsep = F->dim_sep();
        auto dupd = F->dim_upd();
        L_size += dsep*dsep + dsep*dupd;
        U_size += dsep*dupd;
        Schur_size += dupd*dupd;
        piv_size += dsep;
        total_upd_size += dupd;
	F->scratchpad_size_ =
	  std::max(dpcpp::getrf_buffersize<scalar_t>
		   (q, dsep, dsep, dsep),
		   dpcpp::getrs_buffersize<scalar_t>
		   (q, Trans::N, dsep, dupd, dsep, dsep));
	getr_work_size += F->scratchpad_size_;
        if (dsep <= 8)       N8++;
        else if (dsep <= 16) N16++;
        else if (dsep <= 24) N24++;
        else if (dsep <= 32) N32++;
      }
      factor_size = L_size + U_size;
      work_bytes =
        round_to_8(sizeof(scalar_t) * (Schur_size + getr_work_size)) +
        round_to_8(sizeof(std::int64_t) * piv_size) +
	round_to_8(sizeof(FrontData<scalar_t>) * (N8 + N16 + N24 + N32));
    }

    void print_info(int l, int lvls) {
      std::cout << "#  level " << l << " of " << lvls
                << " has " << f.size() << " nodes and "
                << N8 << " <=8, " << N16 << " <=16, "
                << N24 << " <=24, " << N32 << " <=32, needs "
                << factor_size * sizeof(scalar_t) / 1.e6
                << " MB for factors, "
                << Schur_size * sizeof(scalar_t) / 1.e6
                << " MB for Schur complements" << std::endl;
    }

    long long total_flops() {
      long long level_flops = 0;
      for (auto F : f) {
        level_flops += LU_flops(F->F11_) +
          gemm_flops(Trans::N, Trans::N, scalar_t(-1.),
                     F->F21_, F->F12_, scalar_t(1.)) +
          trsm_flops(Side::L, scalar_t(1.), F->F11_, F->F12_) +
          trsm_flops(Side::R, scalar_t(1.), F->F11_, F->F21_);
      }
      return level_flops;
    }

    /*
     * first store L factors, then U factors,
     *  F11, F21, F11, F21, ..., F12, F12, ...
     */
    void set_factor_pointers(scalar_t* factors) {
      for (auto F : f) {
        const int dsep = F->dim_sep();
        const int dupd = F->dim_upd();
        F->F11_ = DenseMW_t(dsep, dsep, factors, dsep); factors += dsep*dsep;
        F->F21_ = DenseMW_t(dupd, dsep, factors, dupd); factors += dupd*dsep;
      }
      for (auto F : f) {
        const int dsep = F->dim_sep();
        const int dupd = F->dim_upd();
        F->F12_ = DenseMW_t(dsep, dupd, factors, dsep); factors += dsep*dupd;
      }
    }

    void set_pivot_pointers(std::int64_t* pmem) {
      for (auto F : f) {
        F->piv_ = pmem;
        pmem += F->dim_sep();
      }
    }

    void set_work_pointers(void* wmem) {
      auto smem = reinterpret_cast<scalar_t*>(wmem);
      for (auto F : f) {
        const int dupd = F->dim_upd();
        if (dupd) {
          F->F22_ = DenseMW_t(dupd, dupd, smem, dupd);
          smem += dupd*dupd;
	}
      }
      for (auto F : f) {
	F->scratchpad_ = smem;
	smem += F->scratchpad_size_;
      }      
      auto imem = reinterpret_cast<std::int64_t*>(round_to_8(smem));
      for (auto F : f) {
        F->piv_ = imem;
        imem += F->dim_sep();
      }
      auto fdat = reinterpret_cast<FrontData<scalar_t>*>
        (round_to_8(imem));
      f8 = fdat;   fdat += N8;
      f16 = fdat;  fdat += N16;
      f24 = fdat;  fdat += N24;
      f32 = fdat;  fdat += N32;
    }

    std::vector<FDPC_t*> f;
    std::size_t L_size = 0, U_size = 0, factor_size = 0,
      Schur_size = 0, piv_size = 0, total_upd_size = 0,
      work_bytes = 0, N8 = 0, N16 = 0, N24 = 0, N32 = 0;
    std::int64_t getr_work_size = 0;
    FrontData<scalar_t> *f8 = nullptr, *f16 = nullptr,
      *f24 = nullptr, *f32 = nullptr;
  };


  template<typename scalar_t,typename integer_t>
  FrontDPCpp<scalar_t,integer_t>::FrontDPCpp
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd)
    : F_t(nullptr, nullptr, sep, sep_begin, sep_end, upd) {}

  template<typename scalar_t,typename integer_t>
  FrontDPCpp<scalar_t,integer_t>::~FrontDPCpp() {
#if defined(STRUMPACK_COUNT_FLOPS)
    const std::size_t dupd = dim_upd();
    const std::size_t dsep = dim_sep();
    STRUMPACK_SUB_MEMORY(dsep*(dsep+2*dupd)*sizeof(scalar_t));
#endif
  }

  template<typename scalar_t,typename integer_t> void
  FrontDPCpp<scalar_t,integer_t>::release_work_memory() {
    F22_.clear();
    host_Schur_.release();
  }

#if defined(STRUMPACK_USE_MPI)
  template<typename scalar_t,typename integer_t> void
  FrontDPCpp<scalar_t,integer_t>::extend_add_copy_to_buffers
  (std::vector<std::vector<scalar_t>>& sbuf,
   const FrontalMatrixMPI<scalar_t,integer_t>* pa) const {
    ExtendAdd<scalar_t,integer_t>::extend_add_seq_copy_to_buffers
      (F22_, sbuf, pa, this);
  }
#endif

  template<typename scalar_t,typename integer_t> void
  FrontDPCpp<scalar_t,integer_t>::extend_add_to_dense
  (DenseM_t& paF11, DenseM_t& paF12, DenseM_t& paF21, DenseM_t& paF22,
   const F_t* p, int task_depth) {
    const std::size_t pdsep = paF11.rows();
    const std::size_t dupd = dim_upd();
    std::size_t upd2sep;
    auto I = this->upd_to_parent(p, upd2sep);
#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop default(shared) grainsize(64)      \
  if(task_depth < params::task_recursion_cutoff_level)
#endif
    for (std::size_t c=0; c<dupd; c++) {
      auto pc = I[c];
      if (pc < pdsep) {
        for (std::size_t r=0; r<upd2sep; r++)
          paF11(I[r],pc) += F22_(r,c);
        for (std::size_t r=upd2sep; r<dupd; r++)
          paF21(I[r]-pdsep,pc) += F22_(r,c);
      } else {
        for (std::size_t r=0; r<upd2sep; r++)
          paF12(I[r],pc-pdsep) += F22_(r, c);
        for (std::size_t r=upd2sep; r<dupd; r++)
          paF22(I[r]-pdsep,pc-pdsep) += F22_(r,c);
      }
    }
    STRUMPACK_FLOPS((is_complex<scalar_t>()?2:1) * dupd * dupd);
    STRUMPACK_FULL_RANK_FLOPS((is_complex<scalar_t>()?2:1) * dupd * dupd);
    release_work_memory();
  }


  // /**
  //  * this doesn't count the memory used for the sparse matrix elements
  //  * needed in the front assembly.
  //  */
  // template<typename scalar_t,typename integer_t>
  // bool sufficient_device_memory
  // (const std::vector<LevelInfo<scalar_t,integer_t>>& ldata) {
  //   std::size_t peak_device_mem = 0;
  //   for (std::size_t l=0; l<ldata.size(); l++) {
  //     auto& L = ldata[l];
  //     // memory needed on this level: factors,
  //     // schur updates, pivot vectors, cuSOLVER work space, ...
  //     std::size_t level_mem = L.factor_size*sizeof(scalar_t) + L.work_bytes;
  //     // the contribution blocks of the previous level are still
  //     // needed for the extend-add
  //     if (l+1 < ldata.size())
  //       level_mem += ldata[l+1].work_bytes;
  //     peak_device_mem = std::max(peak_device_mem, level_mem);
  //   }
  //   // only use 90% of available memory, since we're not counting the
  //   // sparse elements in the peak_device_mem
  //   return peak_device_mem < 0.9 * gpu::available_memory();
  // }

  template<typename T> void
  ea_kernel(int x, int y, int d1, int d2, int dCB,
	    T* F11, T* F12, T* F21, T* F22, T* CB, std::size_t* I) {
    if (x >= dCB || y >= dCB) return;
    auto Ix = I[x], Iy = I[y];
    if (Ix < d1) {
      if (Iy < d1) F11[Iy+Ix*d1] += CB[y+x*dCB];
      else F21[Iy-d1+Ix*d2] += CB[y+x*dCB];
    } else {
      if (Iy < d1) F12[Iy+(Ix-d1)*d1] += CB[y+x*dCB];
      else F22[Iy-d1+(Ix-d1)*d2] += CB[y+x*dCB];
    }
  }

  template<typename T> T rnd(T a, T b) { return ((a + b - 1) / b) * b; }
  
  template <typename T>
  void assemble(cl::sycl::queue& q, std::size_t nf,
		const AssembleData<T>* dat, AssembleData<T>* ddat) {
    // const unsigned int MAX_BLOCKS_Y =
    //   q.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
    // const unsigned int MAX_BLOCKS_Z = MAX_BLOCKS_Y;
    { // front assembly from sparse matrix
      std::size_t nt1 = 128, nt2 = 32;
      std::size_t max1 = nt1, max2 = nt2; 
      for (std::size_t f=0; f<nf; f++) {
	max1 = std::max(max1, dat[f].n11);
	max2 = std::max(max2, std::max(dat[f].n12, dat[f].n21));
      }
      // TODO check if nf is larger than allowed max
      cl::sycl::range<2> global{nf, rnd(max1, nt1)}, local{1, nt1};
      q.parallel_for(cl::sycl::nd_range<2>{global, local},
		     [=](cl::sycl::nd_item<2> it) {
		       auto& F = ddat[it.get_group(0)/*+f*/];
		       auto idx = it.get_global_id(1);
		       if (idx >= F.n11) return;
		       auto& t = F.e11[idx];
		       F.F11[t.r + t.c*F.d1] = t.v;
		     });
      if (max2) {
	cl::sycl::range<2> global{nf, rnd(max2, nt2)}, local{1, nt2};
	q.parallel_for(cl::sycl::nd_range<2>{global, local},
		       [=](cl::sycl::nd_item<2> it) {
			 auto& F = ddat[it.get_group(0)/*+f*/];
			 auto idx = it.get_global_id(1);
			 if (idx < F.n12) {
			   auto& t = F.e12[idx];
			   F.F12[t.r + t.c*F.d1] = t.v;
			 }
			 if (idx < F.n21) {
			   auto& t = F.e21[idx];
			   F.F21[t.r + t.c*F.d2] = t.v;
			 }
		       });
      }
    }
    q.wait_and_throw();
    { // extend-add
      std::size_t nt = 16;
      std::size_t maxCB = nt;
      for (std::size_t f=0; f<nf; f++)
    	maxCB = std::max(maxCB, std::max(dat[f].dCB1, dat[f].dCB2));
      auto gCB = rnd(maxCB, nt);
      cl::sycl::range<3> global{nf, gCB, gCB}, local{1, nt, nt};
      q.parallel_for(cl::sycl::nd_range<3>{global, local},
    		     [=](cl::sycl::nd_item<3> it) {
    		       auto& F = ddat[it.get_group(0)];
    		       if (F.CB1)
    			 ea_kernel(it.get_global_id(1), it.get_global_id(2),
    				   F.d1, F.d2, F.dCB1, F.F11, F.F12, F.F21,
    				   F.F22, F.CB1, F.I1);
    		     });
      q.wait_and_throw();
      q.parallel_for(cl::sycl::nd_range<3>{global, local},
      		     [=](cl::sycl::nd_item<3> it) {
      		       auto& F = ddat[it.get_group(0)];
      		       if (F.CB2)
      			 ea_kernel(it.get_global_id(1), it.get_global_id(2),
      				   F.d1, F.d2, F.dCB2, F.F11, F.F12, F.F21,
      				   F.F22, F.CB2, F.I2);
      		     });
      q.wait_and_throw();
    }
  }


  template<typename scalar_t, typename integer_t> void
  FrontDPCpp<scalar_t,integer_t>::front_assembly
  (cl::sycl::queue& q, const SpMat_t& A, LInfo_t& L) {
    using FDPC_t = FrontDPCpp<scalar_t,integer_t>;
    using Trip_t = Triplet<scalar_t>;
    auto N = L.f.size();
    std::vector<Trip_t> e11, e12, e21;
    std::vector<std::array<std::size_t,3>> ne(N+1);
    std::size_t Isize = 0;
    for (std::size_t n=0; n<N; n++) {
      auto& f = *(L.f[n]);
      ne[n] = std::array<std::size_t,3>{e11.size(), e12.size(), e21.size()};
      A.push_front_elements
        (f.sep_begin_, f.sep_end_, f.upd_, e11, e12, e21);
      if (f.lchild_) Isize += f.lchild_->dim_upd();
      if (f.rchild_) Isize += f.rchild_->dim_upd();
    }
    ne[N] = std::array<std::size_t,3>{e11.size(), e12.size(), e21.size()};
    std::size_t ea_mem_size =
      round_to_8(N*sizeof(AssembleData<scalar_t>)) +
      round_to_8(Isize*sizeof(std::size_t)) +
      round_to_8((e11.size()+e12.size()+e21.size())*sizeof(Trip_t));
    dpcpp::HostMemory<char> hea_mem(ea_mem_size, q);
    dpcpp::DeviceMemory<char> dea_mem(ea_mem_size, q);
    auto hasmbl = hea_mem.as<AssembleData<scalar_t>>();
    auto Iptr = reinterpret_cast<std::size_t*>(round_to_8(hasmbl + N));
    {
      auto helems = reinterpret_cast<Trip_t*>(round_to_8(Iptr + Isize));
      std::copy(e11.begin(), e11.end(), helems);
      std::copy(e12.begin(), e12.end(), helems + e11.size());
      std::copy(e21.begin(), e21.end(), helems + e11.size() + e12.size());
    }
    auto dasmbl = dea_mem.as<AssembleData<scalar_t>>();
    auto dIptr = reinterpret_cast<std::size_t*>(round_to_8(dasmbl + N));
    auto delems = reinterpret_cast<Trip_t*>(round_to_8(dIptr + Isize));
    auto de11 = delems;
    auto de12 = de11 + e11.size();
    auto de21 = de12 + e12.size();
    for (std::size_t n=0; n<N; n++) {
      auto& f = *(L.f[n]);
      hasmbl[n] = AssembleData<scalar_t>
        (f.dim_sep(), f.dim_upd(), f.F11_.data(), f.F12_.data(),
         f.F21_.data(), f.F22_.data(),
         ne[n+1][0]-ne[n][0], ne[n+1][1]-ne[n][1], ne[n+1][2]-ne[n][2],
         de11+ne[n][0], de12+ne[n][1], de21+ne[n][2]);
      if (f.lchild_) {
        auto c = dynamic_cast<FDPC_t*>(f.lchild_.get());
        hasmbl[n].set_ext_add_left(c->dim_upd(), c->F22_.data(), dIptr);
        auto u = c->upd_to_parent(&f);
        std::copy(u.begin(), u.end(), Iptr);
        Iptr += u.size();
        dIptr += u.size();
      }
      if (f.rchild_) {
        auto c = dynamic_cast<FDPC_t*>(f.rchild_.get());
        hasmbl[n].set_ext_add_right(c->dim_upd(), c->F22_.data(), dIptr);
        auto u = c->upd_to_parent(&f);
        std::copy(u.begin(), u.end(), Iptr);
        Iptr += u.size();
        dIptr += u.size();
      }
    }
    dpcpp::memcpy<char>(q, dea_mem, hea_mem, ea_mem_size);
    q.wait();
    assemble(q, N, hasmbl, dasmbl);
  }


  template<typename scalar_t, typename integer_t> void
  FrontDPCpp<scalar_t,integer_t>::factor_small_fronts
  (cl::sycl::queue& q, LInfo_t& L, FrontData<scalar_t>* fdata,
   const Opts_t& opts) {
    if (L.N8 || L.N16 || L.N24 || L.N32) {
      for (std::size_t n=0, n8=0, n16=L.N8,
	     n24=n16+L.N16, n32=n24+L.N24;
           n<L.f.size(); n++) {
        auto& f = *(L.f[n]);
        const auto dsep = f.dim_sep();
        if (dsep <= 32) {
          FrontData<scalar_t>
            t(dsep, f.dim_upd(), f.F11_.data(), f.F12_.data(),
              f.F21_.data(), f.F22_.data(), f.piv_);
          if (dsep <= 8)       fdata[n8++] = t;
          else if (dsep <= 16) fdata[n16++] = t;
          else if (dsep <= 24) fdata[n24++] = t;
          else                 fdata[n32++] = t;
        }
      }
      // gpu::copy_host_to_device(L.f8, fdata, L.N8+L.N16+L.N24+L.N32);
      dpcpp::memcpy(q, L.f8, fdata, (L.N8+L.N16+L.N24+L.N32)).wait();
      std::cout << "TODO factor_block_batch kernel!" << std::endl;
      // auto replace = opts.replace_tiny_pivots();
      // auto thresh = opts.pivot_threshold();
      // gpu::factor_block_batch<scalar_t,8>(L.N8, L.f8, replace, thresh);
      // gpu::factor_block_batch<scalar_t,16>(L.N16, L.f16, replace, thresh);
      // gpu::factor_block_batch<scalar_t,24>(L.N24, L.f24, replace, thresh);
      // gpu::factor_block_batch<scalar_t,32>(L.N32, L.f32, replace, thresh);
    }
  }

  template<typename scalar_t, typename integer_t> void
  FrontDPCpp<scalar_t,integer_t>::factor_large_fronts
  (cl::sycl::queue& q, LInfo_t& L, const Opts_t& opts) {
    for (auto& front : L.f) {
      auto& f = *front;
      if (true) { // (f.dim_sep() > 32) {
	auto e_getrf = dpcpp::getrf
	  (q, f.F11_, f.piv_, f.scratchpad_, f.scratchpad_size_);
	// if (opts.replace_tiny_pivots()) { } // TODO
	if (f.dim_upd()) {
	  auto e_getrs = dpcpp::getrs
	    (q, Trans::N, f.F11_, f.piv_, f.F12_,
	     f.scratchpad_, f.scratchpad_size_, {e_getrf});
	  dpcpp::gemm
	    (q, Trans::N, Trans::N, scalar_t(-1.),
	     f.F21_, f.F12_, scalar_t(1.), f.F22_, {e_getrs});
	}
      }
    }
  }


  template<typename scalar_t,typename integer_t> void
  FrontDPCpp<scalar_t,integer_t>::multifrontal_factorization
  (const SpMat_t& A, const Opts_t& opts,
   int etree_level, int task_depth) {
    // const int max_streams = opts.gpu_streams();

    cl::sycl::queue q(cl::sycl::default_selector{},
		      cl::sycl::property::queue::in_order());
    // cl::sycl::queue q(cl::sycl::host_selector{});
    // cl::sycl::property::queue::in_order());
    if (opts.verbose())
      std::cout << "# SYCL/DPC++ selected device: "
		<< q.get_device().get_info<cl::sycl::info::device::name>()
		<< std::endl;

    const int lvls = this->levels();
    std::vector<LInfo_t> ldata(lvls);
    std::size_t max_small_fronts = 0;
    for (int l=lvls-1; l>=0; l--) {
      std::vector<F_t*> fp;
      this->get_level_fronts(fp, l);
      auto& L = ldata[l];
      L = LInfo_t(fp, q);
      max_small_fronts =
	std::max(max_small_fronts, L.N8+L.N16+L.N24+L.N32);
    }

    // if (!sufficient_device_memory(ldata)) {
    //   split_smaller(A, opts, etree_level, task_depth);
    //   return;
    // }

    dpcpp::HostMemory<FrontData<scalar_t>> fdata(max_small_fronts, q);
    dpcpp::DeviceMemory<char> old_work;
    for (int l=lvls-1; l>=0; l--) {
      TaskTimer tl("");
      tl.start();
      auto& L = ldata[l];
      if (opts.verbose()) L.print_info(l, lvls);
      try {
        dpcpp::DeviceMemory<scalar_t> dev_factors(L.factor_size, q);
        dpcpp::DeviceMemory<char> work_mem(L.work_bytes, q);
	auto e1 = dpcpp::fill
	  (q, dev_factors.template as<scalar_t>(), scalar_t(0.), L.factor_size);
	auto e2 = dpcpp::fill
	  (q, work_mem.template as<scalar_t>(), scalar_t(0.), L.Schur_size);
	L.set_factor_pointers(dev_factors);
        L.set_work_pointers(work_mem);
	// e1.wait();
	// e2.wait();
        front_assembly(q, A, L);
	q.wait();
	old_work = std::move(work_mem);
	// factor_small_fronts
	//   (q, L, fdata.template as<FrontData<scalar_t>>(), opts);
	factor_large_fronts(q, L, opts);
	STRUMPACK_ADD_MEMORY(L.factor_size*sizeof(scalar_t));
        L.f[0]->host_factors_.reset(new scalar_t[L.factor_size]);
        L.f[0]->pivot_mem_.resize(L.piv_size);
	q.wait();
	dpcpp::memcpy
	  (q, L.f[0]->pivot_mem_.data(), L.f[0]->piv_, L.piv_size);
	dpcpp::memcpy<scalar_t>
	  (q, L.f[0]->host_factors_.get(), dev_factors, L.factor_size);
	L.set_factor_pointers(L.f[0]->host_factors_.get());
        L.set_pivot_pointers(L.f[0]->pivot_mem_.data());
	q.wait();
      } catch (const std::bad_alloc& e) {
        std::cerr << "Out of memory" << std::endl;
        abort();
      }
      auto level_flops = L.total_flops();
      STRUMPACK_FULL_RANK_FLOPS(level_flops);
      if (opts.verbose()) {
        auto level_time = tl.elapsed();
        std::cout << "#   GPU Factorization complete, took: "
                  << level_time << " seconds, "
                  << level_flops / 1.e9 << " GFLOPS, "
                  << (float(level_flops) / level_time) / 1.e9
                  << " GFLOP/s" << std::endl;
      }
    }
    const std::size_t dupd = dim_upd();
    if (dupd) { // get the contribution block from the device
      host_Schur_.reset(new scalar_t[dupd*dupd]);
      dpcpp::memcpy(q, host_Schur_.get(), old_work.as<scalar_t>(), dupd*dupd);
      F22_ = DenseMW_t(dupd, dupd, host_Schur_.get(), dupd);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontDPCpp<scalar_t,integer_t>::multifrontal_solve
  (DenseM_t& b, const GPUFactors<scalar_t>* gpu_factors) const {
    FrontalMatrix<scalar_t,integer_t>::multifrontal_solve(b);
  }

  template<typename scalar_t,typename integer_t> void
  FrontDPCpp<scalar_t,integer_t>::forward_multifrontal_solve
  (DenseM_t& b, DenseM_t* work, int etree_level, int task_depth) const {
    DenseMW_t bupd(dim_upd(), b.cols(), work[0], 0, 0);
    bupd.zero();
    if (task_depth == 0) {
      // tasking when calling the children
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
      this->fwd_solve_phase1(b, bupd, work, etree_level, task_depth);
      // no tasking for the root node computations, use system blas threading!
      fwd_solve_phase2(b, bupd, etree_level, params::task_recursion_cutoff_level);
    } else {
      this->fwd_solve_phase1(b, bupd, work, etree_level, task_depth);
      fwd_solve_phase2(b, bupd, etree_level, task_depth);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontDPCpp<scalar_t,integer_t>::fwd_solve_phase2
  (DenseM_t& b, DenseM_t& bupd, int etree_level, int task_depth) const {
    if (dim_sep()) {
      DenseMW_t bloc(dim_sep(), b.cols(), b, this->sep_begin_, 0);

      std::vector<int> piv(piv_, piv_+dim_sep());
      // std::cout << "dpiv (" << dim_sep() << ")= {";
      // for (integer_t i=0; i<dim_sep(); i++)
      // 	std::cout << piv_[i] << " ";
      // std::cout << "}" << std::endl;
      
      // F11_.solve_LU_in_place(bloc, reinterpret_cast<int>(piv.data(), task_depth);
      F11_.solve_LU_in_place(bloc, reinterpret_cast<int*>(piv_), task_depth);
      if (dim_upd()) {
        if (b.cols() == 1)
          gemv(Trans::N, scalar_t(-1.), F21_, bloc,
               scalar_t(1.), bupd, task_depth);
        else
          gemm(Trans::N, Trans::N, scalar_t(-1.), F21_, bloc,
               scalar_t(1.), bupd, task_depth);
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontDPCpp<scalar_t,integer_t>::backward_multifrontal_solve
  (DenseM_t& y, DenseM_t* work, int etree_level, int task_depth) const {
    DenseMW_t yupd(dim_upd(), y.cols(), work[0], 0, 0);
    if (task_depth == 0) {
      // no tasking in blas routines, use system threaded blas instead
      bwd_solve_phase1
        (y, yupd, etree_level, params::task_recursion_cutoff_level);
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
      // tasking when calling children
      this->bwd_solve_phase2(y, yupd, work, etree_level, task_depth);
    } else {
      bwd_solve_phase1(y, yupd, etree_level, task_depth);
      this->bwd_solve_phase2(y, yupd, work, etree_level, task_depth);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontDPCpp<scalar_t,integer_t>::bwd_solve_phase1
  (DenseM_t& y, DenseM_t& yupd, int etree_level, int task_depth) const {
    if (dim_sep()) {
      DenseMW_t yloc(dim_sep(), y.cols(), y, this->sep_begin_, 0);
      if (y.cols() == 1) {
        if (dim_upd())
          gemv(Trans::N, scalar_t(-1.), F12_, yupd,
               scalar_t(1.), yloc, task_depth);
      } else {
        if (dim_upd())
          gemm(Trans::N, Trans::N, scalar_t(-1.), F12_, yupd,
               scalar_t(1.), yloc, task_depth);
      }
    }
  }

  // explicit template instantiations
  template class FrontDPCpp<float,int>;
  template class FrontDPCpp<double,int>;
  template class FrontDPCpp<std::complex<float>,int>;
  template class FrontDPCpp<std::complex<double>,int>;

  template class FrontDPCpp<float,long int>;
  template class FrontDPCpp<double,long int>;
  template class FrontDPCpp<std::complex<float>,long int>;
  template class FrontDPCpp<std::complex<double>,long int>;

  template class FrontDPCpp<float,long long int>;
  template class FrontDPCpp<double,long long int>;
  template class FrontDPCpp<std::complex<float>,long long int>;
  template class FrontDPCpp<std::complex<double>,long long int>;

} // end namespace strumpack
