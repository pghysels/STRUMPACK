#ifndef HSS_MATRIX_EXTRACT_HPP
#define HSS_MATRIX_EXTRACT_HPP

namespace strumpack {
  namespace HSS {

    template<typename scalar_t> scalar_t
    HSSMatrix<scalar_t>::get(std::size_t i, std::size_t j) const {
      if (this->leaf()) return D_(i, j);
      DenseM_t e(this->cols(), 1);
      e.zero();
      e(j,0) = scalar_t(1.);
      return apply(e)(i,0);
    }

    template<typename scalar_t> DenseMatrix<scalar_t>
    HSSMatrix<scalar_t>::extract
    (const std::vector<std::size_t>& I,
     const std::vector<std::size_t>& J) const {
      DenseM_t B(I.size(), J.size());
      B.zero();
      extract_add(I, J, B);
      return B;
    }

    template<typename scalar_t> void HSSMatrix<scalar_t>::extract_add
    (const std::vector<std::size_t>& I,
     const std::vector<std::size_t>& J, DenseM_t& B) const {
      WorkExtract<scalar_t> w;
      w.J = J;
      w.I = I;
      w.ycols.reserve(J.size());
      for (std::size_t c=0; c<J.size(); c++) w.ycols.push_back(c);

#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
      extract_fwd(w, false, this->openmp_task_depth_);

      w.rl2g.reserve(I.size());
      for (std::size_t r=0; r<I.size(); r++) w.rl2g.push_back(r);
      w.cl2g.reserve(J.size());
      for (std::size_t c=0; c<J.size(); c++) w.cl2g.push_back(c);

      // TODO is this necessary???
      w.z = DenseM_t(U_.cols(), w.ycols.size());
      w.z.zero();

#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
      extract_bwd(B, w, this->openmp_task_depth_);
    }

    template<typename scalar_t> void HSSMatrix<scalar_t>::extract_fwd
    (WorkExtract<scalar_t>& w, bool odiag, int depth) const {
      if (w.J.empty()) return;
      if (this->leaf()) {
        if (odiag) w.y = V_.extract_rows(w.J).transpose();
        else w.ycols.clear();
      } else {
        w.split_extraction_sets(child(0)->dims());
        for (std::size_t c=0; c<w.J.size(); c++) {
          if (w.J[c] < child(0)->cols())
            w.c[0].ycols.push_back(w.ycols[c]);
          else w.c[1].ycols.push_back(w.ycols[c]);
        }
        bool tasked = depth < params::task_recursion_cutoff_level;
        if (tasked) {
#pragma omp task default(shared)                                        \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
          child(0)->extract_fwd
            (w.c[0], odiag || !w.c[1].I.empty(), depth+1);
#pragma omp task default(shared)                                        \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
          child(1)->extract_fwd
            (w.c[1], odiag || !w.c[0].I.empty(), depth+1);
#pragma omp taskwait
        } else {
          child(0)->extract_fwd
            (w.c[0], odiag || !w.c[1].I.empty(), depth+1);
          child(1)->extract_fwd
            (w.c[1], odiag || !w.c[0].I.empty(), depth+1);
        }
        w.ycols.clear();
        if (!odiag) return;
        auto ncols = w.c[0].ycols.size() + w.c[1].ycols.size();
        w.ycols.resize(ncols);
        std::copy(w.c[0].ycols.begin(), w.c[0].ycols.end(), w.ycols.begin());
        std::copy(w.c[1].ycols.begin(), w.c[1].ycols.end(),
                  w.ycols.begin()+w.c[0].ycols.size());
        if (V_.cols()) {
          DenseM_t y01(V_.rows(), ncols);
          y01.zero();
          copy(w.c[0].y, y01, 0, 0);
          copy(w.c[1].y, y01, child(0)->V_rank(), w.c[0].y.cols());
          // TODO get Vdense, then do two separate gemms to reduce flops!!!
          w.y = V_.applyC(y01, depth);
          STRUMPACK_EXTRACTION_FLOPS(V_.applyC_flops(y01.cols()));
        } else w.y = DenseM_t(0, w.J.size());
      }
    }

    template<typename scalar_t> void HSSMatrix<scalar_t>::extract_bwd
    (std::vector<Triplet<scalar_t>>& triplets,
     WorkExtract<scalar_t>& w, int depth) const {
      if (w.I.empty()) return;
      if (this->leaf()) {
        std::vector<Triplet<scalar_t>> lt;
        if (w.z.cols() && U_.cols())
          lt.reserve(w.I.size()*(w.J.size()+ w.z.cols()));
        else lt.reserve(w.I.size()*w.J.size());
        assert(w.cl2g.size() >= w.J.size());
        assert(w.rl2g.size() >= w.I.size());
        for (std::size_t c=0; c<w.J.size(); c++)
          for (std::size_t r=0; r<w.I.size(); r++)
            lt.emplace_back(w.rl2g[r], w.cl2g[c], D_(w.I[r],w.J[c]));
        if (w.z.cols() && U_.cols()) {
          DenseM_t tmp(w.I.size(), w.z.cols());
          gemm(Trans::N, Trans::N, scalar_t(1), U_.extract_rows(w.I),
               w.z, scalar_t(0.), tmp, depth);
          STRUMPACK_EXTRACTION_FLOPS
            (gemm_flops(Trans::N, Trans::N, scalar_t(1), U_.extract_rows(w.I),
                        w.z, scalar_t(0.)));
          assert(w.rl2g.size() >= w.I.size());
          for (std::size_t c=0; c<w.z.cols(); c++)
            for (std::size_t r=0; r<w.I.size(); r++)
              lt.emplace_back(w.rl2g[r], w.zcols[c], tmp(r,c));
        }
#pragma omp critical(extract_bwd)
        {
          triplets.reserve(triplets.size() + lt.size());
          triplets.insert(triplets.end(), lt.begin(), lt.end());
        }
      } else {
        extract_bwd_internal(w, depth);
        bool tasked = depth < params::task_recursion_cutoff_level;
        if (tasked) {
#pragma omp task default(shared)                                        \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
          child(0)->extract_bwd(triplets, w.c[0], depth+1);
#pragma omp task default(shared)                                        \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
          child(1)->extract_bwd(triplets, w.c[1], depth+1);
#pragma omp taskwait
        } else {
          child(0)->extract_bwd(triplets, w.c[0], depth+1);
          child(1)->extract_bwd(triplets, w.c[1], depth+1);
        }
      }
    }

    template<typename scalar_t> void HSSMatrix<scalar_t>::extract_bwd
    (DenseMatrix<scalar_t>& B, WorkExtract<scalar_t>& w, int depth) const {
      if (w.I.empty()) return;
      if (this->leaf()) {
        for (std::size_t c=0; c<w.J.size(); c++)
          for (std::size_t r=0; r<w.I.size(); r++)
            B(w.rl2g[r],w.cl2g[c]) += D_(w.I[r],w.J[c]);
        STRUMPACK_EXTRACTION_FLOPS(w.J.size()*w.I.size());
        if (w.z.cols() && U_.cols()) {
          DenseM_t tmp(w.I.size(), w.z.cols());
          gemm(Trans::N, Trans::N, scalar_t(1), U_.extract_rows(w.I),
               w.z, scalar_t(0.), tmp, depth);
          for (std::size_t c=0; c<w.z.cols(); c++)
            for (std::size_t r=0; r<w.I.size(); r++)
              B(w.rl2g[r],w.zcols[c]) += tmp(r,c);
          STRUMPACK_EXTRACTION_FLOPS
            (gemm_flops(Trans::N, Trans::N, scalar_t(1), U_.extract_rows(w.I),
                        w.z, scalar_t(0.)) + w.z.cols()*w.I.size());
        }
      } else {
        extract_bwd_internal(w, depth);
        bool tasked = depth < params::task_recursion_cutoff_level;
        if (tasked) {
#pragma omp task default(shared)                                        \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
          child(0)->extract_bwd(B, w.c[0], depth+1);
#pragma omp task default(shared)                                        \
  final(depth >= params::task_recursion_cutoff_level-1) mergeable
          child(1)->extract_bwd(B, w.c[1], depth+1);
#pragma omp taskwait
        } else {
          child(0)->extract_bwd(B, w.c[0], depth+1);
          child(1)->extract_bwd(B, w.c[1], depth+1);
        }
      }
    }

    template<typename scalar_t> void HSSMatrix<scalar_t>::extract_bwd_internal
    (WorkExtract<scalar_t>& w, int depth) const {
      w.split_extraction_sets(child(0)->dims());
      w.c[0].rl2g.reserve(w.c[0].I.size());
      w.c[1].rl2g.reserve(w.c[1].I.size());
      for (std::size_t r=0; r<w.I.size(); r++) {
        if (w.I[r] < child(0)->rows()) w.c[0].rl2g.push_back(w.rl2g[r]);
        else w.c[1].rl2g.push_back(w.rl2g[r]);
      }
      w.c[0].cl2g.reserve(w.c[0].J.size());
      w.c[1].cl2g.reserve(w.c[1].J.size());
      for (std::size_t c=0; c<w.J.size(); c++) {
        if (w.J[c] < child(0)->cols()) w.c[0].cl2g.push_back(w.cl2g[c]);
        else w.c[1].cl2g.push_back(w.cl2g[c]);
      }
      auto U = U_.dense();
      if (!w.c[0].I.empty()) {
        auto z0cols = w.c[1].y.cols() + w.z.cols();
        auto z0rows = B01_.rows();
        w.c[0].z = DenseM_t(z0rows, z0cols);
        if (!w.c[1].ycols.empty()) {
          DenseMW_t z00(z0rows, w.c[1].y.cols(), w.c[0].z, 0, 0);
          gemm(Trans::N, Trans::N, scalar_t(1.), B01_,
               w.c[1].y, scalar_t(0.), z00, depth);
          STRUMPACK_EXTRACTION_FLOPS
            (gemm_flops(Trans::N, Trans::N, scalar_t(1.), B01_,
                        w.c[1].y, scalar_t(0.)));
        }
        DenseMW_t z01(z0rows, w.z.cols(), w.c[0].z, 0, w.c[1].y.cols());
        if (U.cols()) {
          DenseMW_t U0(z0rows, U.cols(), U, 0, 0);
          gemm(Trans::N, Trans::N, scalar_t(1.), U0, w.z,
               scalar_t(0.), z01, depth);
          STRUMPACK_EXTRACTION_FLOPS
            (gemm_flops(Trans::N, Trans::N, scalar_t(1.), U0, w.z,
                        scalar_t(0.)));
        } else z01.zero();
        w.c[0].zcols.reserve(z0cols);
        for (auto c : w.c[1].ycols) w.c[0].zcols.push_back(c);
        for (auto c : w.zcols) w.c[0].zcols.push_back(c);
      }
      if (!w.c[1].I.empty()) {
        auto z1cols = w.c[0].y.cols() + w.z.cols();
        auto z1rows = B10_.rows();
        w.c[1].z = DenseM_t(z1rows, z1cols);
        if (!w.c[0].ycols.empty()) {
          DenseMW_t z10(z1rows, w.c[0].y.cols(), w.c[1].z, 0, 0);
          gemm(Trans::N, Trans::N, scalar_t(1.), B10_,
               w.c[0].y, scalar_t(0.), z10, depth);
          STRUMPACK_EXTRACTION_FLOPS
            (gemm_flops(Trans::N, Trans::N, scalar_t(1.), B10_,
                        w.c[0].y, scalar_t(0.)));
        }
        DenseMW_t z11(z1rows, w.z.cols(), w.c[1].z, 0, w.c[0].y.cols());
        if (U.cols()) {
          DenseMW_t U1(z1rows, U.cols(), U, child(0)->U_rank(), 0);
          gemm(Trans::N, Trans::N, scalar_t(1.), U1, w.z,
               scalar_t(0.), z11, depth);
          STRUMPACK_EXTRACTION_FLOPS
            (gemm_flops(Trans::N, Trans::N, scalar_t(1.), U1, w.z,
                        scalar_t(0.)));
        } else z11.zero();
        w.c[1].zcols.reserve(z1cols);
        for (auto c : w.c[0].ycols) w.c[1].zcols.push_back(c);
        for (auto c : w.zcols) w.c[1].zcols.push_back(c);
      }
    }

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_MATRIX_EXTRACT_HPP
