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
#include <limits>

#include "FrontSYCL.hpp"

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

  constexpr int align_max_struct() {
    auto m = sizeof(std::complex<double>);
    m = std::max(m, sizeof(AssembleData<std::complex<double>>));
    m = std::max(m, sizeof(Triplet<std::complex<double>>));
    int k = 16;
    while (k < int(m)) k *= 2;
    return k;
  }
  std::size_t round_up(std::size_t n) {
    int k = align_max_struct();
    return std::size_t((n + k - 1) / k) * k;
  }
  template<typename T> T* aligned_ptr(void* p) {
    return (T*)(round_up(uintptr_t(p)));
  }

  template<typename scalar_t, typename integer_t> class LevelInfo {
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using FSYCL_t = FrontSYCL<scalar_t,integer_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;

  public:
    LevelInfo() {}

    LevelInfo(const std::vector<F_t*>& fronts,
              cl::sycl::queue& q, const SpMat_t* A=nullptr) {
      f.reserve(fronts.size());
      for (auto& F : fronts)
        f.push_back(dynamic_cast<FSYCL_t*>(F));
#pragma omp parallel for                        \
  reduction(+:L_size,U_size,Schur_size)         \
  reduction(+:piv_size,total_upd_size)
      for (auto F : f) {
        const std::size_t dsep = F->dim_sep();
        const std::size_t dupd = F->dim_upd();
        L_size += dsep*dsep + dsep*dupd;
        U_size += dsep*dupd;
        Schur_size += dupd*dupd;
        piv_size += dsep;
        total_upd_size += dupd;
      }
      if (A) {
        auto N = f.size();
        elems11.resize(N+1);
        elems12.resize(N+1);
        elems21.resize(N+1);
        Isize.resize(N+1);
#pragma omp parallel for
        for (std::size_t i=0; i<N; i++) {
          auto& F = *(f[i]);
          A->count_front_elements
            (F.sep_begin(), F.sep_end(), F.upd(),
             elems11[i+1], elems12[i+1], elems21[i+1]);
          if (F.lchild_) Isize[i+1] += F.lchild_->dim_upd();
          if (F.rchild_) Isize[i+1] += F.rchild_->dim_upd();
        }
        for (std::size_t i=0; i<N; i++) {
          elems11[i+1] += elems11[i];
          elems12[i+1] += elems12[i];
          elems21[i+1] += elems21[i];
          Isize[i+1] += Isize[i];
        }
      }
      factor_size = L_size + U_size;

      factor_bytes = sizeof(scalar_t) * factor_size;
      factor_bytes = round_up(factor_bytes);

      work_bytes = sizeof(scalar_t) * Schur_size;
      work_bytes = round_up(work_bytes);
      work_bytes += sizeof(std::int64_t) * piv_size;
      work_bytes = round_up(work_bytes);

      ea_bytes = sizeof(AssembleData<scalar_t>) * f.size();
      ea_bytes = round_up(ea_bytes);
      ea_bytes += sizeof(std::size_t) * Isize.back();
      ea_bytes = round_up(ea_bytes);
      ea_bytes += sizeof(Triplet<scalar_t>) *
        (elems11.back() + elems12.back() + elems21.back());
      ea_bytes = round_up(ea_bytes);
    }

    void print_info(int l, int lvls) {
      std::cout << "#  level " << l << " of " << lvls
                << " has " << f.size() << " nodes and "
                << factor_bytes / 1.e6
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
        F->F12_ = DenseMW_t(dsep, dupd, factors, dsep); factors += dsep*dupd;
        F->F21_ = DenseMW_t(dupd, dsep, factors, dupd); factors += dupd*dsep;
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
      auto imem = aligned_ptr<std::int64_t>(smem);
      for (auto F : f) {
        F->piv_ = imem;
        imem += F->dim_sep();
      }
    }

    std::vector<FSYCL_t*> f;
    std::size_t L_size = 0, U_size = 0, factor_size = 0,
      Schur_size = 0, piv_size = 0, total_upd_size = 0,
      factor_bytes = 0, work_bytes = 0, ea_bytes = 0;
    std::vector<std::size_t> elems11, elems12, elems21, Isize;
  };


  template<typename scalar_t,typename integer_t>
  FrontSYCL<scalar_t,integer_t>::FrontSYCL
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd)
    : F_t(nullptr, nullptr, sep, sep_begin, sep_end, upd) {}

  template<typename scalar_t,typename integer_t>
  FrontSYCL<scalar_t,integer_t>::~FrontSYCL() {
#if defined(STRUMPACK_COUNT_FLOPS)
    const std::size_t dupd = dim_upd();
    const std::size_t dsep = dim_sep();
    STRUMPACK_SUB_MEMORY(dsep*(dsep+2*dupd)*sizeof(scalar_t));
#endif
  }

  template<typename scalar_t,typename integer_t> void
  FrontSYCL<scalar_t,integer_t>::release_work_memory() {
    F22_.clear();
    host_Schur_.release();
  }

#if defined(STRUMPACK_USE_MPI)
  template<typename scalar_t,typename integer_t> void
  FrontSYCL<scalar_t,integer_t>::extend_add_copy_to_buffers
  (std::vector<std::vector<scalar_t>>& sbuf,
   const FrontalMatrixMPI<scalar_t,integer_t>* pa) const {
    ExtendAdd<scalar_t,integer_t>::extend_add_seq_copy_to_buffers
      (F22_, sbuf, pa, this);
  }
#endif

  template<typename scalar_t,typename integer_t> void
  FrontSYCL<scalar_t,integer_t>::extend_add_to_dense
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

  template<typename scalar_t,typename integer_t>
  std::size_t peak_device_memory
  (const std::vector<LevelInfo<scalar_t,integer_t>>& ldata) {
    std::size_t peak_dmem = 0;
    for (std::size_t l=0; l<ldata.size(); l++) {
      auto& L = ldata[l];
      // memory needed on this level: factors,
      // schur updates, pivot vectors, cuSOLVER work space,
      // assembly data (indices, sparse elements)
      std::size_t level_mem = L.factor_bytes + L.work_bytes + L.ea_bytes;
      // the contribution blocks of the previous level are still
      // needed for the extend-add
      if (l+1 < ldata.size())
        level_mem += ldata[l+1].work_bytes;
      peak_dmem = std::max(peak_dmem, level_mem);
    }
    return peak_dmem;
  }

  template<typename T> struct Assemble {
    AssembleData<T>* dat;
    std::size_t nf;
    Assemble(AssembleData<T>* d, std::size_t N) : dat(d), nf(N) {}
    void operator()(cl::sycl::nd_item<2> it) const {
      std::size_t op = it.get_global_id(0);
      if (op >= nf) return;
      auto& F = dat[op];
      auto idx = it.get_global_id(1);
      if (idx < F.n11) {
        auto& t = F.e11[idx];
        F.F11[t.r + t.c*F.d1] = t.v;
      }
      if (idx < F.n12) {
        auto& t = F.e12[idx];
        F.F12[t.r + t.c*F.d1] = t.v;
      }
      if (idx < F.n21) {
        auto& t = F.e21[idx];
        F.F21[t.r + t.c*F.d2] = t.v;
      }
    }
  };

  template<typename T, unsigned int unroll> struct EA {
    AssembleData<T>* dat;
    bool left;
    std::size_t nf;
    EA(AssembleData<T>* d, std::size_t N, bool l)
      : dat(d), nf(N), left(l) {}
    void operator()(cl::sycl::nd_item<3> it) const {
      int y = it.get_global_id(2),
        x0 = it.get_group(1) * unroll,
        z = it.get_global_id(0);
      if (z >= nf) return;
      auto& f = dat[z];
      auto CB = left ? f.CB1 : f.CB2;
      if (!CB) return;
      auto dCB = left ? f.dCB1 : f.dCB2;
      if (y >= dCB) return;
      auto I = left ? f.I1 : f.I2;
      auto Iy = I[y];
      CB += y + x0*dCB;
      int d1 = f.d1, d2 = f.d2;
      int ld;
      T* F[2];
      if (Iy < d1) {
        ld = d1;
        F[0] = f.F11+Iy;
        F[1] = f.F12+Iy-d1*d1;
      } else {
        ld = d2;
        F[0] = f.F21+Iy-d1;
        F[1] = f.F22+Iy-d1-d1*d2;
      }
#pragma unroll
      for (int i=0; i<unroll; i++) {
        int x = x0 + i;
        if (x >= dCB) break;
        auto Ix = I[x];
        F[Ix >= d1][Ix*ld] += CB[i*dCB];
      }
    }
  };

  template<typename T> T rnd(T a, T b) { return ((a + b - 1) / b) * b; }


  template<typename scalar_t, typename integer_t> void
  FrontSYCL<scalar_t,integer_t>::front_assembly
  (cl::sycl::queue& q, const SpMat_t& A, LInfo_t& L,
   char* hea_mem, char* dea_mem) {
    using FSYCL_t = FrontSYCL<scalar_t,integer_t>;
    using Trip_t = Triplet<scalar_t>;
    auto N = L.f.size();
    auto hasmbl = aligned_ptr<AssembleData<scalar_t>>(hea_mem);
    auto Iptr   = aligned_ptr<std::size_t>(hasmbl + N);
    auto e11    = aligned_ptr<Trip_t>(Iptr + L.Isize.back());
    auto e12    = e11 + L.elems11.back();
    auto e21    = e12 + L.elems12.back();
    auto dasmbl = aligned_ptr<AssembleData<scalar_t>>(dea_mem);
    auto dIptr  = aligned_ptr<std::size_t>(dasmbl + N);
    auto de11   = aligned_ptr<Trip_t>(dIptr + L.Isize.back());
    auto de12   = de11 + L.elems11.back();
    auto de21   = de12 + L.elems12.back();

#pragma omp parallel for
    for (std::size_t n=0; n<N; n++) {
      auto& f = *(L.f[n]);
      A.set_front_elements
        (f.sep_begin_, f.sep_end_, f.upd_,
         e11+L.elems11[n], e12+L.elems12[n], e21+L.elems21[n]);
      hasmbl[n] = AssembleData<scalar_t>
        (f.dim_sep(), f.dim_upd(), f.F11_.data(), f.F12_.data(),
         f.F21_.data(), f.F22_.data(),
         L.elems11[n+1]-L.elems11[n], L.elems12[n+1]-L.elems12[n],
         L.elems21[n+1]-L.elems21[n],
         de11+L.elems11[n], de12+L.elems12[n], de21+L.elems21[n]);
      auto fIptr = Iptr + L.Isize[n];
      auto fdIptr = dIptr + L.Isize[n];
      if (f.lchild_) {
        auto c = dynamic_cast<FSYCL_t*>(f.lchild_.get());
        hasmbl[n].set_ext_add_left(c->dim_upd(), c->F22_.data(), fdIptr);
        c->upd_to_parent(&f, fIptr);
        fIptr += c->dim_upd();
        fdIptr += c->dim_upd();
      }
      if (f.rchild_) {
        auto c = dynamic_cast<FSYCL_t*>(f.rchild_.get());
        hasmbl[n].set_ext_add_right(c->dim_upd(), c->F22_.data(), fdIptr);
        c->upd_to_parent(&f, fIptr);
      }
    }
    dpcpp::memcpy<char>(q, dea_mem, hea_mem, L.ea_bytes);
    q.wait_and_throw();
    { // front assembly from sparse matrix
      std::size_t nnz = 0;
      for (std::size_t f=0; f<N; f++)
        nnz = std::max
          (nnz, std::max(hasmbl[f].n11, std::max(hasmbl[f].n12, hasmbl[f].n21)));
      if (nnz) {
        std::size_t nt = 512, ops = 1;
        while (nt > nnz) {
          nt /= 2;
          ops *= 2;
        }
        assert(rnd(N,ops) * rnd(nnz,nt) < std::numeric_limits<int>::max());
        cl::sycl::range<2> global{rnd(N,ops), rnd(nnz,nt)}, local{ops, nt};
        q.parallel_for(cl::sycl::nd_range<2>{global, local},
                       Assemble<scalar_t>(dasmbl, N));
      }
    }
    q.wait_and_throw();
    { // extend-add
      std::size_t gCB = 0;
      for (std::size_t f=0; f<N; f++)
        gCB = std::max(gCB, std::max(hasmbl[f].dCB1, hasmbl[f].dCB2));
      if (gCB) {
        std::size_t nt = 256, ops = 1;
        const unsigned int unroll = 16;
        while (nt > gCB) {
          nt /= 2;
          ops *= 2;
        }
        std::size_t gx = (gCB + unroll - 1) / unroll;
        gCB = rnd(gCB, nt);
        assert(gCB * gx * rnd(N,ops) < std::numeric_limits<int>::max());
        cl::sycl::range<3> global{rnd(N, ops), gx, gCB}, local{ops, 1, nt};
        q.parallel_for(cl::sycl::nd_range<3>{global, local},
                       EA<scalar_t,unroll>(dasmbl, N, true));
        q.wait_and_throw();
        q.parallel_for(cl::sycl::nd_range<3>{global, local},
                       EA<scalar_t,unroll>(dasmbl, N, false));
        q.wait_and_throw();
      }
    }
  }


  template<typename scalar_t, typename integer_t>
  struct BatchMetaData {
    using LInfo_t = LevelInfo<scalar_t,integer_t>;
    BatchMetaData() {}
    BatchMetaData(const std::vector<LInfo_t>& L,
                  cl::sycl::queue& q) {
      std::size_t nb = 0;
      for (auto& l : L) nb = std::max(nb, l.f.size());
      std::size_t bytes = nb * 2 * sizeof(std::int64_t);
      bytes = round_up(bytes);
      bytes += nb * 2 * sizeof(std::int64_t); // ds, du
      bytes = round_up(bytes);
      bytes += nb * 5 * sizeof(void*);        // F11, F12, F21, F22, piv
      bytes = round_up(bytes);
      bytes += nb * 2 * sizeof(scalar_t);     // alpha, beta
      bytes = round_up(bytes);
      bytes += nb * sizeof(std::int64_t);     // group_sizes
      bytes = round_up(bytes);
      bytes += nb * sizeof(oneapi::mkl::transpose); // op
      bytes = round_up(bytes);

      hmem_ = dpcpp::HostMemory<char>(bytes, q);
      ds = hmem_.as<std::int64_t>();
      du = ds + nb;
      F11 = aligned_ptr<scalar_t*>(du + nb);
      F12 = F11 + nb;
      F21 = F12 + nb;
      F22 = F21 + nb;
      piv = aligned_ptr<std::int64_t*>(F22 + nb);
      alpha = aligned_ptr<scalar_t>(piv + nb);
      beta = alpha + nb;
      group_sizes = aligned_ptr<std::int64_t>(beta + nb);
      op = aligned_ptr<oneapi::mkl::transpose>(group_sizes + nb);
      dpcpp::fill(q, alpha, scalar_t(-1.), nb);
      dpcpp::fill(q, beta, scalar_t(1.), nb);
      dpcpp::fill(q, group_sizes, std::int64_t(1), nb);
      dpcpp::fill(q, op, oneapi::mkl::transpose::N, nb);
      for (auto& l : L) {
        set_level(q, l, false);
        lwork = std::max
          (lwork,
           oneapi::mkl::lapack::getrf_batch_scratchpad_size<scalar_t>
           (q, ds, ds, ds, l.f.size(), group_sizes));
        lwork = std::max
          (lwork,
           oneapi::mkl::lapack::getrs_batch_scratchpad_size<scalar_t>
           (q, op, ds, du, ds, ds, l.f.size(), group_sizes));
      }
      scratchpad = dpcpp::DeviceMemory<scalar_t>(lwork, q);
    }
    std::size_t set_level(cl::sycl::queue& q, const LInfo_t& L,
                          bool Schur, std::size_t B=0) {
      std::size_t i = 0;
      for (auto& f : L.f) {
        if (Schur && (f->dim_sep() == 0 || f->dim_upd() == 0))
          continue;
        if (f->dim_sep() <= B)
          continue;
        ds[i] = f->dim_sep();
        du[i] = f->dim_upd();
        F11[i] = f->F11_.data();  F12[i] = f->F12_.data();
        F21[i] = f->F21_.data();  F22[i] = f->F22_.data();
        piv[i] = f->piv_;
        i++;
      }
      return i;
    }
    std::size_t set_level_small(cl::sycl::queue& q, const LInfo_t& L,
                                std::size_t Bmin, std::size_t Bmax) {
      std::size_t i = 0;
      for (auto& f : L.f) {
        if (f->dim_sep() <= Bmin || f->dim_sep() > Bmax)
          continue;
        ds[i] = f->dim_sep();
        du[i] = f->dim_upd();
        F11[i] = f->F11_.data();  F12[i] = f->F12_.data();
        F21[i] = f->F21_.data();  F22[i] = f->F22_.data();
        piv[i] = f->piv_;
        i++;
      }
      return i;
    }
    std::int64_t lwork = 0, *ds = nullptr, *du = nullptr,
      **piv = nullptr, *group_sizes = nullptr;
    scalar_t *alpha = nullptr, *beta = nullptr,
      **F11 = nullptr, **F12 = nullptr,
      **F21 = nullptr, **F22 = nullptr;
    oneapi::mkl::transpose* op = nullptr;
    dpcpp::DeviceMemory<scalar_t> scratchpad;
  private:
    dpcpp::HostMemory<char> hmem_;
  };


  template<std::size_t B, typename scalar_t, typename integer_t>
  struct PartialFactor {
    std::int64_t *ds_, *du_, **piv_;
    scalar_t **F11_, **F12_, **F21_, **F22_;
    PartialFactor(std::int64_t *ds, std::int64_t *du, std::int64_t **piv,
                  scalar_t **F11, scalar_t **F12, scalar_t **F21, scalar_t **F22)
      : ds_(ds), du_(du), piv_(piv),
        F11_(F11), F12_(F12), F21_(F21), F22_(F22) {}
    void operator()(cl::sycl::nd_item<3> it) const {
      auto front = it.get_group(0);
      int j = it.get_global_id(1), i = it.get_global_id(2);
      int n = ds_[front], n2 = du_[front];
      auto A11 = F11_[front];
      auto piv = piv_[front];
      for (int k=0; k<n; k++) {
        // TODO make p and Amax global?
        auto p = k;
        auto Amax = std::abs(A11[k+k*n]);
        for (int l=k+1; l<n; l++) {
          auto tmp = std::abs(A11[l+k*n]);
          if (tmp > Amax) {
            Amax = tmp;
            p = l;
          }
        }
        if (i == 0 && j == 0)
          piv[k] = p + 1;
        it.barrier();
        if (Amax == scalar_t(0.)) {
          // TODO
          // if (info == 0)
          //   info = k;
        } else {
          // swap row k with the pivot row
          if (j < n && i == k && p != k) {
            auto tmp = A11[k+j*n];
            A11[k+j*n] = A11[p+j*n];
            A11[p+j*n] = tmp;
          }
        }
        it.barrier();
        // divide by the pivot element
        if (j == k && i > k && i < n)
          A11[i+j*n] /= A11[k+k*n];
        it.barrier();
        // Schur update
        if (j > k && i > k && j < n && i < n)
          A11[i+j*n] -= A11[i+k*n] * A11[k+j*n];
        it.barrier();
      }
      auto A12 = F12_[front];
      for (int cb=0; cb<n2; cb+=B) {
        int c = cb + j;
        bool col = c < n2;
        // L trsm (unit diag)
        for (int k=0; k<n; k++) {
          if (i > k && i < n && col)
            A12[i+c*n] -= A11[i+k*n] * A12[k+c*n];
          it.barrier();
        }
        // U trsm
        for (int k=n-1; k>=0; k--) {
          if (i == k && col)
            A12[k+c*n] /= A11[k+k*n];
          it.barrier();
          if (i < k && col)
            A12[i+c*n] -= A11[i+k*n] * A12[k+c*n];
          it.barrier();
        }
      }
      // Schur GEMM
      auto A22 = F22_[front], A21 = F21_[front];
      for (int c=j; c<n2; c+=B)
        for (int r=i; r<n2; r+=B)
          for (int k=0; k<n; k++)
            A22[r+c*n2] -= A21[r+k*n2] * A12[k+c*n];
    }
  };


  template<std::size_t B, typename scalar_t, typename integer_t>
  void partial_factor_small(cl::sycl::queue& q, std::size_t nb,
                            BatchMetaData<scalar_t,integer_t>& batch) {
    if (!nb) return;
    cl::sycl::range<3> global{nb, B, B}, local{1, B, B};
    q.parallel_for(cl::sycl::nd_range<3>{global, local},
                   PartialFactor<B, scalar_t, integer_t>
                   (batch.ds, batch.du, batch.piv,
                    batch.F11, batch.F12, batch.F21, batch.F22));
  }

  template<typename scalar_t, typename integer_t> void
  FrontSYCL<scalar_t,integer_t>::factor_batch
  (cl::sycl::queue& q, const LInfo_t& L, Batch_t& batch,
   const Opts_t& opts) {
#if 1
    auto nb = batch.set_level(q, L, false);
    oneapi::mkl::lapack::getrf_batch
      (q, batch.ds, batch.ds, batch.F11, batch.ds, batch.piv,
       nb, batch.group_sizes, batch.scratchpad.get(), batch.lwork).wait();
    nb = batch.set_level(q, L, true);
    oneapi::mkl::lapack::getrs_batch
      (q, batch.op, batch.ds, batch.du, batch.F11, batch.ds,
       batch.piv, batch.F12, batch.ds,
       nb, batch.group_sizes, batch.scratchpad.get(), batch.lwork).wait();
    oneapi::mkl::blas::column_major::gemm_batch
      (q, batch.op, batch.op, batch.du, batch.du, batch.ds,
       batch.alpha, const_cast<const scalar_t**>(batch.F21), batch.du,
       const_cast<const scalar_t**>(batch.F12), batch.ds,
       batch.beta, batch.F22, batch.du, nb, batch.group_sizes).wait();
#else
    auto Bmax = 16;
    auto nb = batch.set_level_small(q, L, 0, 8);
    partial_factor_small<8, scalar_t, integer_t>(q, nb, batch);
    q.wait();
    nb = batch.set_level_small(q, L, 8, 16);
    partial_factor_small<16, scalar_t, integer_t>(q, nb, batch);
    q.wait();

    nb = batch.set_level(q, L, false, Bmax);
    oneapi::mkl::lapack::getrf_batch
      (q, batch.ds, batch.ds, batch.F11, batch.ds, batch.piv,
       nb, batch.group_sizes, batch.scratchpad.get(), batch.lwork).wait();
    nb = batch.set_level(q, L, true, Bmax);
    oneapi::mkl::lapack::getrs_batch
      (q, batch.op, batch.ds, batch.du, batch.F11, batch.ds,
       batch.piv, batch.F12, batch.ds,
       nb, batch.group_sizes, batch.scratchpad.get(), batch.lwork).wait();
    oneapi::mkl::blas::column_major::gemm_batch
      (q, batch.op, batch.op, batch.du, batch.du, batch.ds,
       batch.alpha, const_cast<const scalar_t**>(batch.F21), batch.du,
       const_cast<const scalar_t**>(batch.F12), batch.ds,
       batch.beta, batch.F22, batch.du, nb, batch.group_sizes).wait();
#endif
    // #else
    // auto nb = batch.set_level(q, L, false);
    // for (auto& f : L.f)
    //   dpcpp::getrf(q, f->F11_, f->piv_, batch.scratchpad.get(),
    //                batch.lwork).wait();
    // nb = batch.set_level(q, L, true);
    // for (auto& f : L.f)
    //   dpcpp::getrs(q, Trans::N, f->F11_, f->piv_, f->F12_,
    //                     batch.scratchpad.get(), batch.lwork).wait();
    // for (auto& f : L.f)
    //   dpcpp::gemm(q, Trans::N, Trans::N, scalar_t(-1.),
    //                    f->F21_, f->F12_, scalar_t(1.), f->F22_).wait();
    // #endif
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  FrontSYCL<scalar_t,integer_t>::split_smaller
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
    if (opts.verbose())
      std::cout << "# Factorization does not fit in GPU memory, "
        "splitting in smaller traversals." << std::endl;
    ReturnCode err_code = ReturnCode::SUCCESS;
    if (lchild_) {
      auto el = lchild_->multifrontal_factorization
        (A, opts, etree_level+1, task_depth);
      if (el != ReturnCode::SUCCESS) err_code = el;
    }
    if (rchild_) {
      auto er = rchild_->multifrontal_factorization
        (A, opts, etree_level+1, task_depth);
      if (er != ReturnCode::SUCCESS) err_code = er;
    }

    const std::size_t dupd = dim_upd(), dsep = dim_sep();
    STRUMPACK_ADD_MEMORY(dsep*(dsep+2*dupd)*sizeof(scalar_t));
    STRUMPACK_ADD_MEMORY(dupd*dupd*sizeof(scalar_t));
    host_factors_.reset(new scalar_t[dsep*(dsep+2*dupd)]);
    host_Schur_.reset(new scalar_t[dupd*dupd]);
    {
      auto fmem = host_factors_.get();
      F11_ = DenseMW_t(dsep, dsep, fmem, dsep); fmem += dsep*dsep;
      F12_ = DenseMW_t(dsep, dupd, fmem, dsep); fmem += dsep*dupd;
      F21_ = DenseMW_t(dupd, dsep, fmem, dupd);
    }
    F22_ = DenseMW_t(dupd, dupd, host_Schur_.get(), dupd);
    F11_.zero(); F12_.zero();
    F21_.zero(); F22_.zero();
    A.extract_front
      (F11_, F12_, F21_, this->sep_begin_, this->sep_end_,
       this->upd_, task_depth);
    if (lchild_) {
#pragma omp parallel
#pragma omp single
      lchild_->extend_add_to_dense(F11_, F12_, F21_, F22_, this, 0);
    }
    if (rchild_) {
#pragma omp parallel
#pragma omp single
      rchild_->extend_add_to_dense(F11_, F12_, F21_, F22_, this, 0);
    }
    TaskTimer tl("");
    tl.start();
    if (dsep) {
      cl::sycl::queue q; //(cl::sycl::default_selector{});
      auto scratchpad_size = std::max
        (dpcpp::getrf_buffersize<scalar_t>(q, dsep, dsep, dsep),
         dpcpp::getrs_buffersize<scalar_t>
         (q, Trans::N, dsep, dupd, dsep, dsep));
      dpcpp::DeviceMemory<scalar_t> dm11(dsep*dsep + scratchpad_size, q);
      auto scratchpad = dm11 + dsep*dsep;
      dpcpp::DeviceMemory<std::int64_t> dpiv(dsep, q);
      DenseMW_t dF11(dsep, dsep, dm11, dsep);
      dpcpp::memcpy(q, dF11, F11_).wait();
      dpcpp::getrf(q, dF11, dpiv, scratchpad, scratchpad_size).wait();
      // TODO check info code
      pivot_mem_.resize(dsep);
      piv_ = pivot_mem_.data();
      dpcpp::memcpy(q, F11_, dF11);
      dpcpp::memcpy(q, piv_, dpiv.as<std::int64_t>(), dsep);
      q.wait();
      if (opts.replace_tiny_pivots()) {
        // TODO do this on the device!
        auto thresh = opts.pivot_threshold();
        for (std::size_t i=0; i<F11_.rows(); i++)
          if (std::abs(F11_(i,i)) < thresh)
            F11_(i,i) = (std::real(F11_(i,i)) < 0) ? -thresh : thresh;
      }
      if (dupd) {
        dpcpp::DeviceMemory<scalar_t> dm12(dsep*dupd, q);
        DenseMW_t dF12(dsep, dupd, dm12, dsep);
        dpcpp::memcpy(q, dF12, F12_).wait();
        dpcpp::getrs(q, Trans::N, dF11, dpiv, dF12,
                     scratchpad, scratchpad_size).wait();
        dpcpp::memcpy(q, F12_, dF12);
        dm11.release();
        q.wait();
        dpcpp::DeviceMemory<scalar_t> dm2122((dsep+dupd)*dupd, q);
        DenseMW_t dF21(dupd, dsep, dm2122, dupd),
          dF22(dupd, dupd, dm2122+(dsep*dupd), dupd);
        dpcpp::memcpy(q, dF21, F21_);
        dpcpp::memcpy(q, dF22.data(), host_Schur_.get(), dupd*dupd);
        q.wait();
        dpcpp::gemm
          (q, Trans::N, Trans::N, scalar_t(-1.),
           dF21, dF12, scalar_t(1.), dF22).wait();
        dpcpp::memcpy(q, host_Schur_.get(), dF22.data(), dupd*dupd).wait();
      }
    }
    // count flops
    auto level_flops = LU_flops(F11_) +
      gemm_flops(Trans::N, Trans::N, scalar_t(-1.), F21_, F12_, scalar_t(1.)) +
      trsm_flops(Side::L, scalar_t(1.), F11_, F12_) +
      trsm_flops(Side::R, scalar_t(1.), F11_, F21_);
    STRUMPACK_FULL_RANK_FLOPS(level_flops);
    if (opts.verbose()) {
      auto level_time = tl.elapsed();
      std::cout << "#   GPU Factorization complete, took: "
                << level_time << " seconds, "
                << level_flops / 1.e9 << " GFLOPS, "
                << (float(level_flops) / level_time) / 1.e9
                << " GFLOP/s" << std::endl;
    }
    return err_code;
  }


  template<auto query, typename T>
  void do_query(const T& obj_to_query, const std::string& name, int indent=4) {
    std::cout << std::string(indent, ' ') << name << " is '"
              << obj_to_query.template get_info<query>() << "'\n";
  }


  template<typename scalar_t,typename integer_t> ReturnCode
  FrontSYCL<scalar_t,integer_t>::multifrontal_factorization
  (const SpMat_t& A, const Opts_t& opts,
   int etree_level, int task_depth) {
    ReturnCode err_code = ReturnCode::SUCCESS;
    cl::sycl::queue q; //(cl::sycl::default_selector{});
    // cl::sycl::queue q(cl::sycl::cpu_selector{});
    if (opts.verbose())
      std::cout << "# SYCL/DPC++ selected device: "
                << q.get_device().get_info<cl::sycl::info::device::name>()
                << std::endl;

    // // Loop through the available platforms
    // for (auto const& this_platform : cl::sycl::platform::get_platforms() ) {
    //   std::cout << "Found Platform:\n";
    //   do_query<cl::sycl::info::platform::name>
    //          (this_platform, "info::platform::name");
    //   do_query<cl::sycl::info::platform::vendor>
    //          (this_platform, "info::platform::vendor");
    //   do_query<cl::sycl::info::platform::version>
    //          (this_platform, "info::platform::version");
    //   do_query<cl::sycl::info::platform::profile>
    //          (this_platform, "info::platform::profile");
    //   // Loop through the devices available in this plaform
    //   for (auto &dev : this_platform.get_devices() ) {
    //          std::cout << " Device: "
    //                    << dev.get_info<cl::sycl::info::device::name>() << "\n";
    //          std::cout << "is_host(): "
    //                    << (dev.is_host() ? "Yes" : "No") << "\n";
    //          std::cout << "is_cpu(): "
    //                    << (dev.is_cpu() ? "Yes" : "No") << "\n";
    //          std::cout << "is_gpu(): "
    //                    << (dev.is_gpu() ? "Yes" : "No") << "\n";
    //          std::cout << "is_accelerator(): "
    //                    << (dev.is_accelerator() ? "Yes" : "No") << "\n";
    //          do_query<cl::sycl::info::device::vendor>(dev, "info::device::vendor");
    //          do_query<cl::sycl::info::device::driver_version>
    //            (dev, "info::device::driver_version");
    //          do_query<cl::sycl::info::device::max_work_item_dimensions>
    //            (dev, "info::device::max_work_item_dimensions");
    //          do_query<cl::sycl::info::device::max_work_group_size>
    //            (dev, "info::device::max_work_group_size");
    //          do_query<cl::sycl::info::device::mem_base_addr_align>
    //            (dev, "info::device::mem_base_addr_align");
    //          do_query<cl::sycl::info::device::partition_max_sub_devices>
    //            (dev, "info::device::partition_max_sub_devices");
    //          // do_query<cl::sycl::info::device::max_work_item_sizes>
    //          //   (dev, "info::device::max_work_item_sizes");
    //          std::cout << "    info::device::max_work_item_sizes" << " is '"
    //                    << dev.get_info<cl::sycl::info::device::max_work_item_sizes>()[0] << "'\n";
    //          std::cout << "    info::device::max_work_item_sizes" << " is '"
    //                    << dev.get_info<cl::sycl::info::device::max_work_item_sizes>()[1] << "'\n";
    //          std::cout << "    info::device::max_work_item_sizes" << " is '"
    //                    << dev.get_info<cl::sycl::info::device::max_work_item_sizes>()[2] << "'\n";
    //   }
    // }

    const int lvls = this->levels();
    std::vector<LInfo_t> ldata(lvls);
    for (int l=lvls-1; l>=0; l--) {
      std::vector<F_t*> fp;
      this->get_level_fronts(fp, l);
      ldata[l] = LInfo_t(fp, q, &A);
    }

    auto peak_dmem = peak_device_memory(ldata);
    dpcpp::DeviceMemory<char> all_dmem;
    BatchMetaData<scalar_t,integer_t> batch;
    try {
      all_dmem = dpcpp::DeviceMemory<char>(peak_dmem, q, false);
      batch = BatchMetaData<scalar_t,integer_t>(ldata, q);
    } catch (std::exception& e) {
      return split_smaller(A, opts, etree_level, task_depth);
    }

    std::size_t peak_hea_mem = 0;
    for (int l=lvls-1; l>=0; l--)
      peak_hea_mem = std::max(peak_hea_mem, ldata[l].ea_bytes);
    dpcpp::HostMemory<char> hea_mem(peak_hea_mem, q);
    char* old_work = nullptr;
    for (int l=lvls-1; l>=0; l--) {
      TaskTimer tl("");
      tl.start();
      auto& L = ldata[l];
      if (opts.verbose()) L.print_info(l, lvls);
      try {
        char *work_mem = nullptr, *dea_mem = nullptr;
        scalar_t* dev_factors = nullptr;
        if (l % 2) {
          work_mem = all_dmem;
          dea_mem = work_mem + L.work_bytes;
          dev_factors = aligned_ptr<scalar_t>(dea_mem + L.ea_bytes);
        } else {
          work_mem = all_dmem + peak_dmem - L.work_bytes;
          dea_mem = work_mem - L.ea_bytes;
          dev_factors = aligned_ptr<scalar_t>(dea_mem - L.factor_bytes);
        }
        dpcpp::fill(q, dev_factors, scalar_t(0.), L.factor_size);
        dpcpp::fill(q, reinterpret_cast<scalar_t*>(work_mem),
                    scalar_t(0.), L.Schur_size);
        L.set_factor_pointers(dev_factors);
        L.set_work_pointers(work_mem);
        front_assembly(q, A, L, hea_mem, dea_mem);
        old_work = work_mem;
        //factor_large_fronts(q, L, opts);
        factor_batch(q, L, batch, opts);
        STRUMPACK_ADD_MEMORY(L.factor_bytes);
        L.f[0]->host_factors_.reset(new scalar_t[L.factor_size]);
        L.f[0]->pivot_mem_.resize(L.piv_size);
        q.wait_and_throw();
        dpcpp::memcpy
          (q, L.f[0]->pivot_mem_.data(), L.f[0]->piv_, L.piv_size);
        dpcpp::memcpy<scalar_t>
          (q, L.f[0]->host_factors_.get(), dev_factors, L.factor_size);
        L.set_factor_pointers(L.f[0]->host_factors_.get());
        L.set_pivot_pointers(L.f[0]->pivot_mem_.data());
        q.wait_and_throw();
      } catch (const std::bad_alloc& e) {
        std::cerr << "Out of memory" << std::endl;
        abort();
      }
      auto level_flops = L.total_flops();
      STRUMPACK_FLOPS(level_flops);
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
      dpcpp::memcpy(q, host_Schur_.get(),
                    reinterpret_cast<scalar_t*>(old_work), dupd*dupd);
      q.wait_and_throw();
      F22_ = DenseMW_t(dupd, dupd, host_Schur_.get(), dupd);
    }
    return err_code;
  }

  // template<typename scalar_t,typename integer_t> void
  // FrontSYCL<scalar_t,integer_t>::multifrontal_solve
  // (DenseM_t& b, const GPUFactors<scalar_t>* gpu_factors) const {
  //   FrontalMatrix<scalar_t,integer_t>::multifrontal_solve(b);
  // }

  template<typename scalar_t,typename integer_t> void
  FrontSYCL<scalar_t,integer_t>::forward_multifrontal_solve
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
  FrontSYCL<scalar_t,integer_t>::fwd_solve_phase2
  (DenseM_t& b, DenseM_t& bupd, int etree_level, int task_depth) const {
    if (dim_sep()) {
      DenseMW_t bloc(dim_sep(), b.cols(), b, this->sep_begin_, 0);
      std::vector<int> p(piv_, piv_+dim_sep());
      F11_.solve_LU_in_place(bloc, p.data(), task_depth);
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
  FrontSYCL<scalar_t,integer_t>::backward_multifrontal_solve
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
  FrontSYCL<scalar_t,integer_t>::bwd_solve_phase1
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
  template class FrontSYCL<float,int>;
  template class FrontSYCL<double,int>;
  template class FrontSYCL<std::complex<float>,int>;
  template class FrontSYCL<std::complex<double>,int>;

  template class FrontSYCL<float,long int>;
  template class FrontSYCL<double,long int>;
  template class FrontSYCL<std::complex<float>,long int>;
  template class FrontSYCL<std::complex<double>,long int>;

  template class FrontSYCL<float,long long int>;
  template class FrontSYCL<double,long long int>;
  template class FrontSYCL<std::complex<float>,long long int>;
  template class FrontSYCL<std::complex<double>,long long int>;

} // end namespace strumpack
