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
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reverse.h>
#include <thrust/sequence.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/transform_scan.h>
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/scatter.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/unique.h>
#include <thrust/count.h>
#include <thrust/transform.h>
#include <thrust/binary_search.h>
#include <thrust/host_vector.h>

// Execution policy that does not synchronize the device after each
// algorithm. All work is on the default stream, so kernels still run
// in order; the host only waits when it reads back a result.
#if defined(__HIP_PLATFORM_AMD__)
#include <thrust/system/hip/execution_policy.h>
#define STRUMPACK_ND_NOSYNC thrust::hip::par_nosync
#define STRUMPACK_ND_DEVICE_SYNC() (void)hipDeviceSynchronize()
#define STRUMPACK_ND_MALLOC(p, n) (hipMalloc(p, n) == hipSuccess)
#define STRUMPACK_ND_FREE(p) (void)hipFree(p)
#else
#include <thrust/system/cuda/execution_policy.h>
#define STRUMPACK_ND_NOSYNC thrust::cuda::par_nosync
#define STRUMPACK_ND_DEVICE_SYNC() (void)cudaDeviceSynchronize()
#define STRUMPACK_ND_MALLOC(p, n) (cudaMalloc(p, n) == cudaSuccess)
#define STRUMPACK_ND_FREE(p) (void)cudaFree(p)
#endif

#include <map>
#include <new>
#include <limits>
#include <type_traits>
#include <stdexcept>
#include <vector>
#include <queue>
#include <functional>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <chrono>

#include <assert.h>

#include "sparse/SeparatorTree.hpp"

namespace strumpack {
  namespace ordering {

    // For subgraphs with fewer than MD_SWITCH nodes, use a minimum
    // degree ordering. This is also the number of threads per block
    // in the minimum_degree kernel. Should be a multiple of
    // 8*sizeof(unsigned int) == 32, since the adjacency is stored as
    // a bitset of unsigned ints (this is unrelated to the warp size,
    // so it works for both 32-wide CUDA warps and 64-wide AMD
    // wavefronts).
    const int MD_SWITCH = 128;

    // "no value yet" for the per component argmin (see
    // argmin_score_ftor)
    constexpr int ND_NONE = 0x7fffffff;

    /**
     * Caching allocator for the temporary storage of the thrust
     * algorithms (sorts, scans, copy_if, reductions) in the ND loop.
     * Freed blocks are kept and reused for later requests of at most
     * the same size, instead of a hipMalloc/hipFree pair per call
     * (hipFree also synchronizes the device). All ND work is on the
     * default stream, so a block handed to a later algorithm is only
     * written after the previous user has finished with it. All
     * blocks are released in the destructor.
     */
    struct nd_cache_alloc {
      using value_type = char;
      std::multimap<std::ptrdiff_t, char*> free_;  // size -> block
      std::map<char*, std::ptrdiff_t> used_;       // block -> size
      nd_cache_alloc() = default;
      nd_cache_alloc(const nd_cache_alloc&) = delete;
      nd_cache_alloc& operator=(const nd_cache_alloc&) = delete;
      ~nd_cache_alloc() {
        for (auto& b : free_) STRUMPACK_ND_FREE(b.second);
        for (auto& b : used_) STRUMPACK_ND_FREE(b.first);
      }
      char* allocate(std::ptrdiff_t n) {
        auto it = free_.lower_bound(n);
        if (it != free_.end()) {
          auto r = *it;
          free_.erase(it);
          used_.emplace(r.second, r.first);
          return r.second;
        }
        void* q = nullptr;
        if (!STRUMPACK_ND_MALLOC(&q, std::max<std::ptrdiff_t>(n, 1))) {
          // release the cache and try once more
          for (auto& b : free_) STRUMPACK_ND_FREE(b.second);
          free_.clear();
          if (!STRUMPACK_ND_MALLOC(&q, std::max<std::ptrdiff_t>(n, 1)))
            throw std::bad_alloc();
        }
        used_.emplace(static_cast<char*>(q), n);
        return static_cast<char*>(q);
      }
      void deallocate(char* q, std::size_t) {
        auto it = used_.find(q);
        if (it == used_.end()) return;
        free_.emplace(it->second, q);
        used_.erase(it);
      }
    };

    /** integer setting from the environment, read once per call site */
    inline int nd_env_int(const char* name, int def) {
      auto s = std::getenv(name);
      return s ? std::atoi(s) : def;
    }

    /**
     * Connected components by union-find (ECL-CC style), in one pass
     * over the edges: parent must be initialized to the identity.
     * Roots are always linked to the smaller root, so parent[x] <= x
     * and the root of every tree is its smallest node. After
     * uf_root_ftor, parent[i] (the label) of a live node is the
     * smallest index in its component, the same result as label
     * propagation; eliminated nodes keep parent[i] = i. Reads of
     * parent may be stale (other threads link concurrently), but a
     * stale value is still an ancestor, and the links themselves are
     * atomicCAS, so the result does not depend on the scheduling.
     */
    // parent is read and written through a volatile pointer: other
    // threads change it concurrently (atomicCAS), and plain loads may
    // be cached (in registers, or in the non-coherent L1 on AMD)
    __device__ inline int uf_find(int* parent_, int x) {
      volatile int* parent = parent_;
      int curr = parent[x];
      if (curr != x) {
        int next, prev = x;
        while (curr > (next = parent[curr])) {
          parent[prev] = next; // path splitting, next is an ancestor
          prev = curr;
          curr = next;
        }
      }
      return curr;
    }
    template<typename integer> struct uf_hook_ftor {
      const integer *ptr, *ind;
      const bool* elim;
      int* parent;
      __device__ void operator()(int i) const {
        if (elim[i]) return;
        int ri = uf_find(parent, i);
        for (auto k=ptr[i], khi=ptr[i+1]; k<khi; k++) {
          int j = ind[k];
          if (j == i || elim[j]) continue;
          int rj = uf_find(parent, j);
          while (ri != rj) {
            if (ri < rj) {
              int o = atomicCAS(parent + rj, rj, ri);
              if (o == rj) break;
              rj = o;
            } else {
              int o = atomicCAS(parent + ri, ri, rj);
              if (o == ri) { ri = rj; break; }
              ri = o;
            }
          }
        }
      }
    };
    // final labels: walk to the root without path compression, and
    // only write parent[i] (with compression, another thread could
    // overwrite parent[i] with a non-root ancestor after it was set
    // to the root). Every tree is complete after uf_hook_ftor, so
    // the walks see no concurrent links.
    struct uf_root_ftor {
      const bool* elim;
      int* parent;
      __device__ void operator()(int i) const {
        if (elim[i]) return;
        volatile int* par = parent;
        int r = par[i], next;
        while (r > (next = par[r])) r = next;
        par[i] = r;
      }
    };

    /**
     * Frontier queue BFS for all components simultaneously. Each node
     * enters the queue q once, the layers are consecutive: layer it
     * is q[qb[it], qb[it] + qs[it]). bfs_q_init_ftor (one thread per
     * component) sets label[root] = cs[c] and puts the live roots in
     * layer 0 (qs[0], qs[1] and qb[0] must be zero). Kernel it labels
     * the unvisited (label == unv) neighbors of layer it with label
     * + 1 and appends them to layer it+1, so the labels are the same
     * as those of a sweep over all nodes per layer; only the order in
     * the queue depends on the scheduling. It also sets qb[it+1] and
     * clears the counter qs[it+2] of the layer after next, and
     * maintains levels[c], the number of layers of component c (must
     * be initialized to 1): every thread that creates layer it+1 in
     * component c writes the same value it+2.
     * check: skip roots that are not a node of component c (coarse
     * graphs, components without coarse nodes).
     */
    struct bfs_q_init_ftor {
      const int *roots, *cs, *comp;
      const bool* elim;
      int m;
      bool check;
      int *label, *q, *qs;
      __device__ void operator()(int c) const {
        int r = roots[c];
        if (check && (r < 0 || r >= m || comp[r] != c)) return;
        label[r] = cs[c];
        if (elim[r]) return;
        q[atomicAdd(qs, 1)] = r;
      }
    };
    template<typename integer> __global__ void
    bfs_q_layer_kernel(int it, int unv, const integer* ptr,
                       const integer* ind, const bool* elim, int* label,
                       const int* comp, int* levels, int* q, int* qs,
                       int* qb) {
      // new nodes are first collected per block in shared memory,
      // then appended with one global atomic per block
      constexpr int SQ = 1024;
      __shared__ int sq[SQ];
      __shared__ int scnt, sbase;
      const int b = qb[it], s = qs[it];
      if (blockIdx.x == 0 && threadIdx.x == 0) {
        qb[it+1] = b + s;
        qs[it+2] = 0;
      }
      if (s == 0) return;
      const int nb = b + s;
      for (int k0=blockIdx.x*blockDim.x; k0<s; k0+=gridDim.x*blockDim.x) {
        if (threadIdx.x == 0) scnt = 0;
        __syncthreads();
        const int k = k0 + threadIdx.x;
        if (k < s) {
          const int i = q[b+k];
          const int li = label[i];
          bool run = false;
          for (auto e=ptr[i], ehi=ptr[i+1]; e<ehi; e++) {
            const int j = ind[e];
            // (a plain read of label[j] before the CAS, to skip the
            // atomic for labeled neighbors, was ~10% slower on Serena
            // on MI-class GPUs: same labels, extra memory round trip)
            if (elim[j]) continue;
            if (atomicCAS(label+j, unv, li+1) == unv) {
              run = true;
              int ps = atomicAdd(&scnt, 1);
              if (ps < SQ) sq[ps] = j;
              else q[nb + atomicAdd(qs+it+1, 1)] = j;
            }
          }
          if (run) {
            const int c = comp[i];
            if (levels[c] < it + 2) levels[c] = it + 2;
          }
        }
        __syncthreads();
        const int cnt = min(scnt, SQ);
        if (threadIdx.x == 0 && cnt) sbase = atomicAdd(qs+it+1, cnt);
        __syncthreads();
        for (int t=threadIdx.x; t<cnt; t+=blockDim.x)
          q[nb+sbase+t] = sq[t];
        __syncthreads();
      }
    }

    /**
     * Root of a connected component after union-find: a live
     * node that kept its own index as label (the smallest index in
     * its component). The roots are compacted in increasing order
     * (thrust::copy_if is stable), so component c is the component
     * with the c-th smallest minimum node index, independent of
     * thread scheduling.
     */
    struct comp_root_pred {
      const bool* elim;
      const int* label;
      __device__ bool operator()(int i) const
      { return !elim[i] && label[i] == i; }
    };

    /**
     * Component sizes (comp_size must be zeroed), eliminated nodes
     * get comp = nr_comps. The sizes are first accumulated per thread
     * block in shared memory: early in the nested dissection there
     * are only a few large components, and one global atomic per
     * node on the same counter is very slow.
     */
    __global__ void count_comps_kernel(int n, const bool* eliminated,
                                       int nr_comps, int* comp,
                                       int* comp_size) {
      __shared__ int sc, scnt;
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (threadIdx.x == 0) { sc = -1; scnt = 0; }
      __syncthreads();
      int c = -1;
      if (i < n) {
        if (!eliminated[i]) c = comp[i];
        else comp[i] = nr_comps;
      }
      if (c >= 0) atomicCAS(&sc, -1, c);
      __syncthreads();
      if (c >= 0) {
        if (c == sc) atomicAdd(&scnt, 1);
        else atomicAdd(comp_size + c, 1);
      }
      __syncthreads();
      if (threadIdx.x == 0 && scnt) atomicAdd(comp_size + sc, scnt);
    }

    struct random_roots_ftor {
      int seed, *root, *start, *map;
      bool mix; // STRUMPACK_ND_ROOT_MIX: different seed per component
      random_roots_ftor(int seed_, int* root_, int* start_, int* map_,
                        bool mix_=false)
        : seed(seed_), root(root_), start(start_), map(map_), mix(mix_) {}
      __device__ void operator()(int i) {
        thrust::minstd_rand gen
          (mix ? unsigned(seed) * 2654435761u + unsigned(i) * 40503u + 1u
           : unsigned(seed));
        if (mix) gen.discard(4);
        thrust::uniform_int_distribution<int>
          dist(start[i], start[i+1]-1);
        root[i] = map[dist(gen)];
      }
    };

    /**
     * Deterministic per component argmin, used to select BFS roots.
     * Every candidate node i has a score S(i) in [0, ND_NONE) (-1:
     * not a candidate); the selected node of component c is the
     * candidate with the smallest score, the smallest index among
     * ties. Two passes with 32-bit atomics: argmin_score_ftor
     * computes the minimum score per component (best, initialized to
     * ND_NONE), argmin_node_ftor the smallest node attaining it (node,
     * initialized to ND_NONE), then argmin_root_ftor writes it to
     * root[c]; components without a candidate keep their previous
     * root. A single pass with a packed 64-bit (score, node)
     * atomicMin was ~10% slower overall on Transport: early in the
     * dissection all nodes of a huge component hit the same counter.
     * The plain read before the atomicMin lets most threads skip it.
     */
    template<typename S> struct argmin_score_ftor {
      S s;
      const int* comp;
      int* best;
      __device__ void operator()(int i) const {
        int v = s(i);
        if (v < 0) return;
        int* b = best + comp[i];
        if (v < *b) atomicMin(b, v);
      }
    };
    template<typename S> struct argmin_node_ftor {
      S s;
      const int *comp, *best;
      int* node;
      __device__ void operator()(int i) const {
        int v = s(i);
        if (v < 0 || v != best[comp[i]]) return;
        atomicMin(node + comp[i], i);
      }
    };
    struct argmin_root_ftor {
      const int* node;
      int* root;
      __device__ void operator()(int c) const {
        int r = node[c];
        if (r != ND_NONE) root[c] = r;
      }
    };
    /**
     * Score for a new BFS root (pseudo-peripheral search): nodes in
     * the last layer of their component, by (live) degree.
     */
    template<typename integer> struct peripheral_score {
      const integer *ptr, *ind;
      const bool* elim;
      const int *label, *comp, *levels, *comp_start;
      __device__ int operator()(int i) const {
        if (elim[i]) return -1;
        int c = comp[i];
        if (label[i] != comp_start[c] + levels[c] - 1) return -1;
        int deg = 0;
        for (auto k=ptr[i], khi=ptr[i+1]; k<khi; k++)
          if (!elim[ind[k]]) deg++;
        return deg;
      }
    };

    /**
     * Store layer cardinalities for the component starting at node i
     * (== label[i]) at layer_size[i], and at layer_size[i+1] for the
     * next layer, etc. Count nr of nodes in each separator, where a
     * separator is all nodes in a layer that are also connected to
     * the next layer, store result in sep_size[l] for level l.
     */
    template<typename integer> struct cardinalities_ftor {
      integer *ptr, *ind;
      bool *eliminated;
      int *label, *comp, *levels, *comp_start, *layer_size, *sep_size;
      // node weights (coarse graphs, STRUMPACK_ND_ML=2), nullptr: 1
      const int* vw = nullptr;
      cardinalities_ftor(integer* ptr_, integer* ind_,
                         bool* eliminated_, int* label_, int* comp_,
                         int* levels_, int* comp_start_, int* layer_size_,
                         int* sep_size_)
        : ptr(ptr_), ind(ind_), eliminated(eliminated_), label(label_),
          comp(comp_), levels(levels_), comp_start(comp_start_),
          layer_size(layer_size_), sep_size(sep_size_) {}
      // contribution of node i: returns its layer l (-1: none), its
      // weight wi, and whether it counts for sep_size[l] (dn)
      __device__ int eval(int i, int& wi, bool& dn) const {
        dn = false;
        wi = 0;
        if (eliminated[i]) return -1;
        auto c = comp[i];
        auto nlvls = levels[c];
        if (nlvls < 3) // whole component eliminated
          return -1;
        auto l0 = comp_start[c];
        int l = label[i];
        wi = vw ? vw[i] : 1;
        if (l == l0 || l == l0+nlvls-1) // ignore first/last
          return l;
        for (auto k=ptr[i], khi=ptr[i+1]; k<khi; k++) {
          auto j = ind[k];
          if (eliminated[j]) continue;
          if (label[j] == l + 1) { dn = true; break; }
        }
        return l;
      }
    };
    /**
     * Launch with blocks of at least 2*NS threads. The counts are
     * accumulated per block in shared memory for up to NS layers
     * (slot l % NS, first come), the other layers go directly to
     * global memory: consecutive nodes are often in the same few
     * layers, and per node global atomics on the same counters are
     * slow. Integer sums, so the result is the same.
     */
    template<typename integer> __global__ void
    cardinalities_kernel(int m, cardinalities_ftor<integer> f) {
      constexpr int NS = 8;
      __shared__ int skey[NS], scnt[2*NS];
      if (threadIdx.x < NS) skey[threadIdx.x] = -1;
      if (threadIdx.x < 2*NS) scnt[threadIdx.x] = 0;
      __syncthreads();
      const int i = blockIdx.x * blockDim.x + threadIdx.x;
      int wi = 0;
      bool dn = false;
      const int l = i < m ? f.eval(i, wi, dn) : -1;
      if (l >= 0) {
        const int h = l & (NS-1);
        int k = skey[h];
        if (k == -1) {
          k = atomicCAS(&skey[h], -1, l);
          if (k == -1) k = l;
        }
        if (k == l) {
          atomicAdd(&scnt[h], wi);
          if (dn) atomicAdd(&scnt[NS+h], wi);
        } else {
          atomicAdd(f.layer_size+l, wi);
          if (dn) atomicAdd(f.sep_size+l, wi);
        }
      }
      __syncthreads();
      if (threadIdx.x < NS) {
        const int t = threadIdx.x, k = skey[t];
        if (k >= 0) {
          if (scnt[t]) atomicAdd(f.layer_size+k, scnt[t]);
          if (scnt[NS+t]) atomicAdd(f.sep_size+k, scnt[NS+t]);
        }
      }
    }

    /**
     * For each component, choose the layer l giving the smallest
     * normalized separator |S| (1/|A| + 1/|B|), with S the part of
     * layer l connected to layer l+1, A = layers <= l minus S, and
     * B = layers > l. sep_side[c] is set to 0 (mark_sep_ftor, the
     * vertex cover and FM also handle side 1: S connected to l-1).
     */
    // launched over the nodes (one of which, the root, does the work
    // per component), or with per_comp over the components
    template<typename integer> struct minimize_sep_ftor {
      integer *ptr, *ind;
      bool *eliminated;
      int *label, *comp, *levels, *comp_start, *layer_size,
        *sep_size, *comp_root, *comp_root_opt, *sep_level, *sep_side;
      float *min_norm_sep;
      float bal; // balance constraint, see below
      // STRUMPACK_ND_TWOEND: kind of level structure (0: BFS from
      // comp_root, 2: two-end from comp_root and root2), recorded in
      // opt_mode (and opt_root2) for the best candidate
      int *root2, *opt_root2, *opt_mode, mode;
      // per_comp: the argument is a component c, the work is done for
      // its root comp_root[c] (nnodes: number of nodes, to skip
      // components without a valid root), same result as the launch
      // over all nodes
      bool per_comp = false;
      int nnodes = 0;
      minimize_sep_ftor(integer* ptr_, integer* ind_,
                        bool* eliminated_, int* label_, int* comp_,
                        int* levels_, int* comp_start_,
                        int* layer_size_, int* sep_size_,
                        int* comp_root_, int* comp_root_opt_,
                        int* sep_level_, int* sep_side_,
                        float* min_norm_sep_, float bal_,
                        int* root2_=nullptr, int* opt_root2_=nullptr,
                        int* opt_mode_=nullptr, int mode_=0)
        : ptr(ptr_), ind(ind_),
          eliminated(eliminated_), label(label_), comp(comp_),
          levels(levels_), comp_start(comp_start_),
          layer_size(layer_size_), sep_size(sep_size_),
          comp_root(comp_root_), comp_root_opt(comp_root_opt_),
          sep_level(sep_level_), sep_side(sep_side_),
          min_norm_sep(min_norm_sep_), bal(bal_), root2(root2_),
          opt_root2(opt_root2_), opt_mode(opt_mode_), mode(mode_) {}
      __device__ void operator()(int i) {
        if (per_comp) {
          const int cc = i;
          i = comp_root[cc];
          if (i < 0 || i >= nnodes || comp[i] != cc) return;
        }
        if (eliminated[i]) return;
        auto c = comp[i];
        // one thread per component: the BFS root (with a two-end
        // level structure, layer 0 can have many nodes)
        if (i != comp_root[c])
          return;
        auto nlvls = levels[c];
        if (nlvls < 3) {
          // a short two-end structure (possible for the off-axis
          // pairs, STRUMPACK_ND_TWOEND=3) should not overwrite the
          // choice made for a longer single-root structure. For the
          // pair (r1, r2) this never happens: it has as many levels
          // as the BFS from r1.
          if (mode == 2) return;
          comp_root_opt[c] = comp_root[c];
          if (opt_mode) opt_mode[c] = 0;
          return;
        }
        auto c0 = comp_start[c];
        auto cA = layer_size[c0];
        // on a coarse graph (STRUMPACK_ND_ML=2) the weighted layer
        // sizes of a component add up to the fine component size
        auto cB = comp_start[c+1] - c0 - cA;
        // best candidate overall (minl), and best candidate that
        // satisfies the balance constraint min(|A|,|B|) >= bal
        // (|A|+|B|) (minlb). With bal == 0 every candidate passes.
        float minNS = 0, minNSb = 0;
        int minl = -1, minlb = -1;
        for (auto l=1; l<nlvls-1; l++) {
          auto cL = layer_size[c0+l];
          cB -= cL;
          int S = sep_size[c0+l], A = cA + cL - S, B = cB;
          cA += cL;
          if (A <= 0 || B <= 0) {
            // degenerate, only accept as last resort (first layer)
            if (minl < 0) { minl = l; minNS = 3.0e37f; }
            continue;
          }
          float NS = S * (1.f / A + 1.f / B);
          if (minl < 0 || NS < minNS) { minNS = NS; minl = l; }
          if (bal > 0 && float(min(A, B)) < bal * float(A + B))
            continue;
          if (minlb < 0 || NS < minNSb) { minNSb = NS; minlb = l; }
        }
        if (minlb >= 0) {
          minNS = minNSb; minl = minlb;
        } else if (bal > 0) {
          // no balanced candidate, penalize so that a balanced
          // separator from another BFS restart is preferred
          minNS += 1.0e30f;
        }
        if (minl >= 0 && minNS < min_norm_sep[c]) {
          sep_level[c] = c0 + minl;
          sep_side[c] = 0;
          comp_root_opt[c] = comp_root[c];
          min_norm_sep[c] = minNS;
          if (opt_mode) opt_mode[c] = mode;
          if (root2) opt_root2[c] = root2[c];
        }
      }
    };

    /**
     * Two-end level structure (STRUMPACK_ND_TWOEND=1). With d1, d2
     * the BFS distances from two roots r1, r2 of a component and
     * D = d1(r2), the key (d1 - d2 + D) / 2 lies in [0, D] (triangle
     * inequality) and differs by at most 1 between neighbors, so it
     * is a valid level structure. Its layers lie near the bisector
     * of r1 and r2 (flat cross-sections), instead of spheres around
     * a single root. On input dist1 holds the labels of the BFS from
     * r1, label those of the BFS from r2 = root2[c]. If mode is not
     * null, only components with mode[c] == 2 are combined, the
     * others get the labels and levels of the BFS from r1 back.
     */
    struct twoend_combine_ftor {
      const bool* elim;
      const int *comp, *comp_start, *dist1, *root2, *lev1, *mode;
      int *label, *levels;
      __device__ void operator()(int i) const {
        if (elim[i]) return;
        int c = comp[i], c0 = comp_start[c];
        if (mode && mode[c] != 2) {
          label[i] = dist1[i];
          if (i == root2[c]) levels[c] = lev1[c];
          return;
        }
        int D = dist1[root2[c]] - c0;
        int d1 = dist1[i] - c0, d2 = label[i] - c0;
        label[i] = c0 + (d1 - d2 + D) / 2;
        if (i == root2[c]) levels[c] = D + 1;
      }
    };

    /**
     * Off-axis far node (STRUMPACK_ND_TWOEND=3): per component, the
     * node maximizing min(d1, d2), with d1, d2 the BFS distances from
     * r1 and r2 (labels dist1, dist2), smallest index among ties
     * (score for argmin_score_ftor, every live node is a candidate).
     */
    struct offaxis_score {
      const bool* elim;
      const int *comp, *comp_start, *dist1, *dist2;
      __device__ int operator()(int i) const {
        if (elim[i]) return -1;
        int c = comp[i], c0 = comp_start[c];
        int m = min(dist1[i] - c0, dist2[i] - c0);
        return (ND_NONE - 1) - m;
      }
    };

    /**
     * Separator selection with an explicit flag per node (refined by
     * the vertex cover and FM if enabled). flag[i]
     * = 1 if node i is eliminated in this step: either part of the
     * separator chosen by minimize_sep_ftor, or in a component with
     * fewer than 3 levels (eliminated as a whole).
     */
    template<typename integer> struct mark_sep_ftor {
      integer *ptr, *ind;
      bool *eliminated;
      int *label, *comp, *levels, *sep_level, *sep_side, *flag;
      mark_sep_ftor(integer* ptr_, integer* ind_, bool* eliminated_,
                    int* label_, int* comp_, int* levels_,
                    int* sep_level_, int* sep_side_, int* flag_)
        : ptr(ptr_), ind(ind_), eliminated(eliminated_), label(label_),
          comp(comp_), levels(levels_), sep_level(sep_level_),
          sep_side(sep_side_), flag(flag_) {}
      __device__ void operator()(int i) {
        flag[i] = 0;
        if (eliminated[i]) return;
        auto c = comp[i];
        if (levels[c] < 3) { flag[i] = 1; return; }
        auto li = label[i];
        if (li == sep_level[c]) {
          auto lt = sep_side[c] ? li - 1 : li + 1;
          for (auto k=ptr[i], khi=ptr[i+1]; k<khi; k++) {
            auto j = ind[k];
            if (eliminated[j]) continue;
            if (label[j] == lt) { flag[i] = 1; return; }
          }
        }
      }
    };
    struct count_flag_ftor {
      int *flag, *comp, *cnt;
      count_flag_ftor(int* flag_, int* comp_, int* cnt_)
        : flag(flag_), comp(comp_), cnt(cnt_) {}
      __device__ void operator()(int i) {
        if (flag[i]) atomicAdd(cnt+comp[i], 1);
      }
    };
    struct nonzero_pred {
      __device__ bool operator()(int f) const { return f != 0; }
    };
    /**
     * Eliminate the nodes of list (sorted by component, and by index
     * within a component), in that order.
     */
    template<typename integer> struct elim_list_ftor {
      const int* list;
      bool* elim;
      integer* perm;
      __device__ void operator()(int k) const {
        int i = list[k];
        perm[k] = i;
        elim[i] = true;
      }
    };

    /**
     * Nodes of the components with at most nmax nodes, grouped by
     * component: l2g[cs[c], cs[c+1]) gets the nodes of component c in
     * arbitrary order (cnt must be zeroed); minimum_degree sorts them.
     */
    struct md_gather_ftor {
      const bool* elim;
      const int *comp, *cs;
      int nmax;
      int *cnt, *l2g;
      __device__ void operator()(int i) const {
        if (elim[i]) return;
        const int c = comp[i], a = cs[c];
        if (cs[c+1] - a > nmax) return;
        l2g[a + atomicAdd(cnt + c, 1)] = i;
      }
    };

    /**
     * Exact fill (deficiency) of eliminating local node v with
     * adjacency row Dv (interior bits in words [0, H), boundary bits
     * in [H, B)): the number of missing edges between two interior
     * neighbors of v, plus between an interior and a boundary
     * neighbor. Missing edges between two boundary neighbors are not
     * counted: those nodes belong to the ancestor separators, whose
     * fronts are dense anyway. Rows do not contain their own bit.
     */
    template<int B, int BP> __device__ inline int
    md_fill(const unsigned int (*D)[BP], const unsigned int* Dv) {
      constexpr int BI = sizeof(unsigned int)*8, H = B / 2;
      int ii = 0, ib = 0;
      for (int w=0; w<H; w++) {
        unsigned int x = Dv[w];
        while (x) {
          const int a = w * BI + __ffs(x) - 1;
          x &= x - 1;
          // a itself is counted as missing in its own word
          ii--;
#pragma unroll
          for (int u=0; u<H; u++) ii += __popc(Dv[u] & ~D[a][u]);
#pragma unroll
          for (int u=H; u<B; u++) ib += __popc(Dv[u] & ~D[a][u]);
        }
      }
      return ii / 2 + ib; // interior pairs were counted twice
    }

    /**
     * Run minimum degree (or minimum fill, see mode) on graphs with up
     * to NT nodes.  The ordering is written to perm+perm_off in
     * reverse order.  This is called for different blocks c =
     * blockIdx.x, blocks should have NT threads.
     * l2g[comp_start[c],comp_star[c+1]) should have all indices of
     * comp c, in any order (see md_gather_ftor). g2l is used as work
     * space.
     * mode (STRUMPACK_ND_MD_FILL): 0: minimum degree, 1: minimum
     * degree, ties broken by the smallest fill (md_fill), 2: minimum
     * fill, ties broken by the smallest degree. Remaining ties go to
     * the earliest node in a BFS (Cuthill-McKee like) order of the
     * component, computed in the kernel: pseudo-peripheral search
     * starting from the smallest index node (next root: minimum
     * degree node in the last level, smallest index among ties,
     * until the number of levels stops growing), then nodes by
     * (level, index).
     * The pivot is chosen with one atomicMin on a unique 64-bit key,
     * so the result does not depend on thread scheduling.
     */
    template<int NT, typename integer> __global__
    void minimum_degree(integer* ptr, integer* ind, bool* eliminated,
                        int* g2l, int* l2g, int* comp_start,
                        integer* perm, int* perm_off, int nmax,
                        int mode) {
      const int c = blockIdx.x;
      const int c0 = comp_start[c];
      const int n = comp_start[c+1] - c0;
      if (n > NT || n > nmax) return;
      perm += perm_off[c];
      const int tid = threadIdx.x;
      constexpr int BI = sizeof(unsigned int)*8; // bits per block
      // x2 for the boundary nodes
      constexpr int B = 2 * NT / BI;        // blocks per row
      // rows padded by one word: thread tid accesses row tid, with
      // stride B the threads would hit the same shared memory banks
      __shared__ unsigned int D[NT][B+1];
      using ull = unsigned long long;
      // pivot keys (and minimum degrees for mode 1), triple buffered,
      // see the elimination loop
      __shared__ ull skey[3];
      __shared__ unsigned int sdeg[3];
      __shared__ int hfull;
      // hash table of boundary (already eliminated) neighbors, slot
      // index = boundary bit position, -1 = empty
      __shared__ int htab[NT];
      // BFS level of each local node, and local node at each position
      // of the tie breaking order
      __shared__ int lev[NT], pinv[NT];
      bool elim = tid >= n;
      // local numbering by increasing global index (l2g is filled
      // with atomics, so its order depends on the scheduling)
      if (!elim) htab[tid] = l2g[c0+tid];
      __syncthreads();
      if (!elim) {
        const int myg = htab[tid];
        int rnk = 0;
        for (int j=0; j<n; j++) rnk += htab[j] < myg;
        lev[rnk] = myg;
      }
      __syncthreads();
      integer gi = 0;
      if (!elim) {
        gi = lev[tid];
        g2l[gi] = tid;
      }
      htab[tid] = -1;
      __syncthreads();
      if (!elim) {
#pragma unroll
        for (int j=0; j<B; j++)
          D[tid][j] = 0;
        for (auto k=ptr[gi], khi=ptr[gi+1]; k<khi; k++) {
          auto j = ind[k];
          if (!eliminated[j]) {
            auto lj = g2l[j];
            D[tid][lj / BI] |= (1u << (lj % BI));
          }
        }
        // no self loop, also if the graph has a diagonal entry
        D[tid][tid / BI] &= ~(1u << (tid % BI));
      }
      // also capture the boundary, see: "The Minimum Degree Ordering
      // with Constraints", Joseph W. H. Liu.
      // https://doi.org/10.1137/0910069 or section 2.8 in "Improving
      // the Run Time and Quality of Nested Dissection Ordering",
      // Bruce Hendrickson and Edward Rothberg.
      // https://doi.org/10.1137/S1064827596300656
      // Each boundary (eliminated) neighbor gets a slot in the shared
      // hash table; the slot index is its bit in the boundary part of
      // D. At most NT boundary nodes are tracked. The threads insert
      // in parallel (atomicCAS, linear probing): the slot of a node
      // then depends on the scheduling, but the ordering does not,
      // since degrees and fills only count bits. If the table
      // overflows, which nodes are dropped would depend on the
      // scheduling, so then the table is rebuilt with the threads
      // inserting one after the other. (Neighbors in other live
      // components do not exist, so eliminated[] of a neighbor is not
      // changed by the other blocks.)
      if (tid == 0) hfull = 0;
      __syncthreads();
      if (!elim) {
        for (auto k=ptr[gi], khi=ptr[gi+1]; k<khi; k++) {
          int key = int(ind[k]);
          if (!eliminated[key]) continue;
          unsigned h = (unsigned(key) * 2654435761u) % NT;
          bool ok = false;
          for (int probe=0; probe<NT; probe++) {
            int old = atomicCAS(htab + h, -1, key);
            if (old == -1 || old == key) {
              D[tid][(NT + h) / BI] |= (1u << (h % BI));
              ok = true;
              break;
            }
            h = (h + 1) % NT;
          }
          if (!ok) hfull = 1;
        }
      }
      __syncthreads();
      if (hfull) {
        htab[tid] = -1;
        if (!elim)
          for (int j=B/2; j<B; j++) D[tid][j] = 0;
        __syncthreads();
        for (int t=0; t<n; t++) {
          if (tid == t) {
            for (auto k=ptr[gi], khi=ptr[gi+1]; k<khi; k++) {
              int key = int(ind[k]);
              if (!eliminated[key]) continue;
              unsigned h = (unsigned(key) * 2654435761u) % NT;
              for (int probe=0; probe<NT; probe++) {
                int old = htab[h];
                if (old == -1) htab[h] = key;
                if (old == -1 || old == key) {
                  D[tid][(NT + h) / BI] |= (1u << (h % BI));
                  break;
                }
                h = (h + 1) % NT;
              }
            }
          }
          __syncthreads();
        }
      }

      // Tie breaking order: pseudo-peripheral BFS on the interior part
      // of the bitset rows (words [0, H)), all decisions are uniform
      // over the block (__syncthreads_or, shared root).
      constexpr int H = B / 2;
      __shared__ unsigned int bvis[H], bfr[H], bnx[H];
      __shared__ int sroot, sbest;
      unsigned int ideg = 0;
      if (!elim) {
#pragma unroll
        for (int w=0; w<H; w++) ideg += __popc(D[tid][w]);
      }
      if (tid == 0) sroot = 0;
      __syncthreads();
      for (int old=1; ; ) {
        const int r = sroot;
        if (tid < H)
          bvis[tid] = bfr[tid] = (r / BI == tid) ? (1u << (r % BI)) : 0u;
        if (!elim) lev[tid] = (tid == r) ? 0 : -1;
        int L = 0;
        __syncthreads();
        while (true) {
          if (tid < H) bnx[tid] = 0;
          __syncthreads();
          if (!elim && ((bfr[tid / BI] >> (tid % BI)) & 1u)) {
#pragma unroll
            for (int w=0; w<H; w++)
              if (D[tid][w]) atomicOr(bnx + w, D[tid][w]);
          }
          __syncthreads();
          bool any = false;
          if (tid < H) {
            const unsigned int x = bnx[tid] & ~bvis[tid];
            bvis[tid] |= x;
            bfr[tid] = x;
            any = x != 0;
          }
          if (!__syncthreads_or(any)) break;
          L++;
          if (!elim && ((bfr[tid / BI] >> (tid % BI)) & 1u)) lev[tid] = L;
        }
        if (L + 1 <= old) break;
        old = L + 1;
        if (tid == 0) sbest = 0x7fffffff;
        __syncthreads();
        if (!elim && lev[tid] == L)
          atomicMin(&sbest, int(ideg << 8) | tid);
        __syncthreads();
        if (tid == 0) sroot = sbest & 0xff;
        __syncthreads();
      }
      // position of each node in the (level, index) order
      int prio = tid;
      if (!elim) {
        const int lv = lev[tid];
        int r = 0;
        for (int j=0; j<n; j++) {
          const int lj = lev[j];
          r += (lj < lv) || (lj == lv && j < tid);
        }
        prio = r;
        pinv[r] = tid;
      }
      __syncthreads();

      // Elimination loop. Step i uses the key buffers b = i % 3, and
      // resets those of step i+1 before its first barrier: their last
      // readers were in step i-2, which every thread finished before
      // the barrier of step i-1. For minimum degree a single barrier
      // per step suffices: a thread only reads its own row before the
      // barrier, and after it only row min_i, which is not written in
      // step i (only rows of non pivots are updated, each by its own
      // thread), and whose last update (step i-1) precedes the
      // barrier. md_fill reads other rows before the pivot barrier,
      // so modes 1 and 2 first wait for the updates of step i-1.
      if (tid < 3) { skey[tid] = ~0ull; sdeg[tid] = ~0u; }
      __syncthreads();
      for (int i=0; i<n; i++) {
        const int b = i % 3;
        if (tid == 0) {
          skey[(i+1) % 3] = ~0ull;
          sdeg[(i+1) % 3] = ~0u;
        }
        if (mode != 0) __syncthreads();
        unsigned int deg = 0;
        if (!elim) {
#pragma unroll
          for (int j=0; j<B; j++) deg += __popc(D[tid][j]);
        }
        if (mode == 1) {
          if (!elim) atomicMin(sdeg + b, deg);
          __syncthreads();
        }
        if (!elim) {
          ull key;
          bool cand = true;
          if (mode == 0)
            key = (ull(deg) << 32) | ull(prio);
          else if (mode == 1) {
            cand = deg == sdeg[b];
            key = cand ? (ull(md_fill<B,B+1>(D, D[tid])) << 32) | ull(prio)
              : 0;
          } else
            key = (ull(md_fill<B,B+1>(D, D[tid])) << 32) |
              (ull(deg) << 16) | ull(prio);
          if (cand) atomicMin(skey + b, key);
        }
        __syncthreads();
        const int min_i = pinv[int(skey[b] & 0xffffull)];
        if (!elim) {
          if (tid == min_i) {
            elim = true;
            // order is reversed
            perm[n-1-i] = gi;
            eliminated[gi] = true;
          } else {
            // Schur update / create a clique (only row min_i is read,
            // and its thread does not write)
            if (D[tid][min_i / BI] & (1u << (min_i % BI)))
#pragma unroll
              for (int j=0; j<B; j++)
                D[tid][j] |= D[min_i][j];
            // clear column min_i, and the own bit that came in with
            // row min_i (it would inflate the degree by one)
            D[tid][min_i / BI] &= ~(1u << (min_i % BI));
            D[tid][tid / BI] &= ~(1u << (tid % BI));
          }
        }
      }
    }

    /**
     * Record the nested dissection tree structure. Each component c
     * that is processed in an ND step becomes one tree node, with id
     * off + c. Nodes eliminated in this step (as separator, as a
     * minimum degree leaf, or as a small component) get tnode set to
     * that tree node. Nodes that are still alive get tparent set to
     * it, so the components formed in the next step know their
     * parent separator.
     */
    struct tag_tree_ftor {
      int nr_comps, off, *comp, *tnode, *tparent;
      bool *eliminated;
      tag_tree_ftor(int nr_comps_, int off_, int* comp_, bool* eliminated_,
                    int* tnode_, int* tparent_)
        : nr_comps(nr_comps_), off(off_), comp(comp_), tnode(tnode_),
          tparent(tparent_), eliminated(eliminated_) {}
      __device__ void operator()(int i) {
        auto c = comp[i];
        // nodes eliminated in an earlier step have comp == nr_comps
        if (c < 0 || c >= nr_comps) return;
        if (eliminated[i]) {
          if (tnode[i] < 0) tnode[i] = off + c;
        } else tparent[i] = off + c;
      }
    };

    struct tuple_less_equal_ftor {
      __device__ __forceinline__
      bool operator()(const thrust::tuple<int,int>& t)
      { return thrust::get<0>(t) <= thrust::get<1>(t); }
    };
    struct larger_to_zero_ftor {
      int v;
      larger_to_zero_ftor(int v_) : v(v_) {}
      __device__ __forceinline__ int operator()(int a)
      { return a > v ? 0 : a; }
    };

    template<typename P, typename T>
    bool less_equal_n(const P& pol, int n, const T& a, const T& b) {
      return thrust::all_of
        (pol,
         thrust::make_zip_iterator(thrust::make_tuple(a.begin(), b.begin())),
         thrust::make_zip_iterator(thrust::make_tuple(a.begin()+n, b.begin()+n)),
         tuple_less_equal_ftor());
    }


    /**
     * Diagnostics, enabled with STRUMPACK_ND_STATS=1. Prints one line
     * per nested dissection step (all components are split
     * simultaneously in one step), and for the first few steps the
     * size and balance of the largest separator.
     */
    inline bool nd_stats_enabled() {
      static const bool on = [] {
        auto s = std::getenv("STRUMPACK_ND_STATS");
        return s && std::atoi(s) != 0;
      }();
      return on;
    }

    // sep_count[c] is the number of nodes eliminated from component
    // c in this step (separator, or whole component if < 3 levels)
    inline void print_nd_stats
    (int step, int n, int nr_comps, int nr_elim, int nr_elim_MD,
     const thrust::device_vector<int>& sep_count,
     const thrust::device_vector<int>& comp_start,
     const thrust::device_vector<int>& levels,
     const thrust::device_vector<int>& sep_level,
     const thrust::device_vector<int>& sep_side,
     const thrust::device_vector<int>& label,
     const thrust::device_vector<int>& comp,
     const thrust::device_vector<bool>& eliminated) {
      thrust::host_vector<int> hs(nr_comps), hcs(nr_comps+1),
        hl(nr_comps), hsl(nr_comps);
      thrust::copy_n(sep_count.begin(), nr_comps, hs.begin());
      thrust::copy_n(comp_start.begin(), nr_comps+1, hcs.begin());
      thrust::copy_n(levels.begin(), nr_comps, hl.begin());
      thrust::copy_n(sep_level.begin(), nr_comps, hsl.begin());
      long long nsep = 0, nsmall = 0;
      int nseps = 0, cmax = -1;
      for (int c=0; c<nr_comps; c++) {
        if (hcs[c+1] == hcs[c]) continue;
        if (hl[c] < 3) { nsmall += hs[c]; continue; }
        nseps++;
        nsep += hs[c];
        if (cmax == -1 || hs[c] > hs[cmax]) cmax = c;
      }
      std::cout << "# ND step " << std::setw(2) << step
                << ": comps=" << nr_comps
                << " elim_before=" << nr_elim
                << " MD=" << nr_elim_MD
                << " seps=" << nseps << " sep_nodes=" << nsep
                << " small_comp_nodes=" << nsmall;
      if (cmax >= 0) {
        int csize = hcs[cmax+1] - hcs[cmax];
        std::cout << " max_sep=" << hs[cmax] << " (comp size " << csize
                  << ", " << hl[cmax] << " lvls)";
        if (step < 6) {
          // balance of the largest separator, see minimize_sep_ftor
          thrust::host_vector<int> hlab(label), hcomp(comp);
          thrust::host_vector<bool> helim(eliminated);
          int side = sep_side[cmax];
          long long A = 0, L = 0, B = 0;
          for (int i=0; i<n; i++) {
            if (helim[i] || hcomp[i] != cmax) continue;
            if (hlab[i] < hsl[cmax]) A++;
            else if (hlab[i] == hsl[cmax]) L++;
            else B++;
          }
          if (side) B += L - hs[cmax];
          else A += L - hs[cmax];
          std::cout << " |A|=" << A << " |S|=" << hs[cmax]
                    << " |B|=" << B << " sep_layer="
                    << hsl[cmax] - hcs[cmax]
                    << (side ? " (up)" : " (down)");
        }
      }
      std::cout << std::endl;
    }

    /**
     * Separator refinement on the GPU (STRUMPACK_ND_VC=thresh,
     * thresh > 0, default 1). For every component
     * with a separator of at least thresh nodes, the separator lies
     * in layer lo or lo+1 of the level structure, and any vertex
     * cover of the bipartite graph between layers lo (U) and lo+1 (V)
     * is a valid separator. Steps, all components at once:
     *  - greedy parallel matching: unmatched U nodes propose to their
     *    lowest index unmatched V neighbor, V nodes accept the lowest
     *    index proposal (deterministic), repeated a number of rounds,
     *  - Z = nodes reachable from unmatched U nodes by alternating
     *    paths (level synchronous BFS),
     *  - cover = (U \ Z) + (V intersect Z). This covers every U-V edge
     *    for any matching; for a maximum matching it is minimum.
     *    The new separator is only used if it is smaller than the
     *    original one.
     */
    template<typename integer> struct vc_ctx {
      const integer *ptr, *ind;
      const bool* elim;
      const int *comp, *levels, *sep_level, *sep_side, *label;
      int *flag, *mate, *best, *z, *oldcnt, *newcnt;
      bool* running;
      int n, thresh;
      // work arrays for the augmenting path phases, best is reused
      // as the BFS parent (U node) of each V node, rend[root] is the
      // free V node ending the path of a tree (n: none yet), prop[j]
      // the smallest U node proposing to claim V node j
      int *uroot = nullptr, *ulev = nullptr, *rend = nullptr,
        *prop = nullptr;
      // 0: not involved, 1: in U (layer lo), 2: in V (layer lo+1)
      __device__ int raw_role(int i) const {
        if (elim[i]) return 0;
        int c = comp[i];
        if (levels[c] < 3) return 0;
        int lo = sep_side[c] ? sep_level[c] - 1 : sep_level[c];
        int l = label[i];
        return (l == lo) ? 1 : ((l == lo + 1) ? 2 : 0);
      }
      __device__ int role(int i) const {
        int r = raw_role(i);
        return (r && oldcnt[comp[i]] < thresh) ? 0 : r;
      }
    };
    template<typename integer> struct vc_init_ftor {
      vc_ctx<integer> x;
      __device__ void operator()(int i) const {
        x.mate[i] = -1; x.best[i] = x.n; x.z[i] = 0;
        if (x.flag[i] && x.raw_role(i))
          atomicAdd(x.oldcnt + x.comp[i], 1);
      }
    };
    template<typename integer> struct vc_propose_ftor {
      vc_ctx<integer> x;
      __device__ void operator()(int i) const {
        if (x.mate[i] >= 0 || x.role(i) != 1) return;
        int jmin = x.n;
        for (auto k=x.ptr[i], khi=x.ptr[i+1]; k<khi; k++) {
          int j = x.ind[k];
          if (j < jmin && x.mate[j] < 0 && x.role(j) == 2) jmin = j;
        }
        if (jmin < x.n) atomicMin(x.best + jmin, i);
      }
    };
    template<typename integer> struct vc_accept_ftor {
      vc_ctx<integer> x;
      __device__ void operator()(int j) const {
        int u = x.best[j];
        if (u >= x.n) return;
        x.best[j] = x.n;
        x.mate[j] = u;
        x.mate[u] = j;
        *x.running = true;
      }
    };
    /**
     * Augmenting path phase: alternating BFS forest grown from all
     * free U nodes simultaneously. Per layer, the U nodes of the
     * layer propose to claim their unclaimed V neighbors (atomicMin,
     * vc_aug_expand_ftor), then each proposed V node is claimed by
     * its smallest proposer (vc_aug_claim_ftor), so the result does
     * not depend on thread scheduling, and every V node belongs to
     * at most one tree (the trees are vertex disjoint). When a tree
     * reaches a free V node, the smallest such node ends the path of
     * the tree's root, which is augmented afterwards by one thread.
     * If no augmenting path is found in a phase, the matching is
     * maximum (the forest covers everything reachable from free U
     * nodes by alternating paths).
     * (Frontier queues for these layers, as for the BFS, were tried:
     * same matching, but ~40% slower than the sweeps over all nodes
     * on Transport, also with per block staging of the appends.)
     */
    template<typename integer> struct vc_aug_init_ftor {
      vc_ctx<integer> x;
      __device__ void operator()(int i) const {
        bool root = x.mate[i] < 0 && x.role(i) == 1;
        x.best[i] = -1;
        x.ulev[i] = root ? 0 : -1;
        x.uroot[i] = root ? i : -1;
        x.rend[i] = x.n;
        x.prop[i] = x.n;
      }
    };
    template<typename integer> struct vc_aug_expand_ftor {
      vc_ctx<integer> x;
      int L;
      __device__ void operator()(int i) const {
        if (x.ulev[i] != L) return;
        if (x.rend[x.uroot[i]] < x.n) return; // tree is done
        for (auto k=x.ptr[i], khi=x.ptr[i+1]; k<khi; k++) {
          int j = x.ind[k];
          if (j == x.mate[i] || x.best[j] != -1 || x.role(j) != 2)
            continue;
          atomicMin(x.prop + j, i);
        }
      }
    };
    template<typename integer> struct vc_aug_claim_ftor {
      vc_ctx<integer> x;
      int L;
      __device__ void operator()(int j) const {
        int u = x.prop[j];
        if (u >= x.n) return;
        x.prop[j] = x.n;
        x.best[j] = u;
        *x.running = true;
        int root = x.uroot[u], m = x.mate[j];
        if (m < 0) atomicMin(x.rend + root, j);
        else { x.uroot[m] = root; x.ulev[m] = L + 1; }
      }
    };
    template<typename integer> struct vc_aug_apply_ftor {
      vc_ctx<integer> x;
      __device__ void operator()(int i) const {
        if (x.ulev[i] != 0) return;
        int v = x.rend[i];
        if (v >= x.n) return;
        while (true) {
          int u = x.best[v], nv = x.mate[u];
          x.mate[u] = v;
          x.mate[v] = u;
          if (u == i) break;
          v = nv;
        }
        *x.running = true;
      }
    };
    // Z: the free U nodes (level 0 of the queue q, see
    // bfs_q_layer_kernel for the layout) and everything reachable from
    // them by alternating paths
    template<typename integer> struct vc_zinit_ftor {
      vc_ctx<integer> x;
      int *q, *qs;
      __device__ void operator()(int i) const {
        if (x.mate[i] < 0 && x.role(i) == 1) {
          x.z[i] = 1;
          q[atomicAdd(qs, 1)] = i;
        }
      }
    };
    // one layer of the alternating BFS: from a U node of Z to its V
    // neighbors other than its mate, from a V node of Z to its mate.
    // Z is the closure, so the result is the same as for repeated
    // sweeps over all nodes.
    template<typename integer> __global__ void
    vc_z_layer_kernel(vc_ctx<integer> x, int it, int* q, int* qs,
                      int* qb) {
      const int b = qb[it], s = qs[it];
      if (blockIdx.x == 0 && threadIdx.x == 0) {
        qb[it+1] = b + s;
        qs[it+2] = 0;
      }
      const int nb = b + s;
      for (int k=blockIdx.x*blockDim.x+threadIdx.x; k<s;
           k+=gridDim.x*blockDim.x) {
        const int i = q[b+k];
        if (x.role(i) == 1) {
          const int mi = x.mate[i];
          for (auto e=x.ptr[i], ehi=x.ptr[i+1]; e<ehi; e++) {
            const int j = x.ind[e];
            if (j == mi || x.z[j] || x.role(j) != 2) continue;
            if (atomicCAS(x.z + j, 0, 1) == 0)
              q[nb + atomicAdd(qs + it + 1, 1)] = j;
          }
        } else {
          const int m = x.mate[i];
          if (m >= 0 && !x.z[m] && x.role(m) == 1 &&
              atomicCAS(x.z + m, 0, 1) == 0)
            q[nb + atomicAdd(qs + it + 1, 1)] = m;
        }
      }
    }
    template<typename integer> struct vc_count_new_ftor {
      vc_ctx<integer> x;
      __device__ void operator()(int i) const {
        int r = x.role(i);
        if (!r) return;
        if ((r == 1) ? !x.z[i] : x.z[i])
          atomicAdd(x.newcnt + x.comp[i], 1);
      }
    };
    template<typename integer> struct vc_apply_ftor {
      vc_ctx<integer> x;
      __device__ void operator()(int i) const {
        int r = x.role(i);
        if (!r) return;
        int c = x.comp[i];
        if (x.newcnt[c] < x.oldcnt[c])
          x.flag[i] = (r == 1) ? !x.z[i] : x.z[i];
      }
    };

    /**
     * FM style separator refinement on the GPU (STRUMPACK_ND_FM=passes,
     * default 100, 0 = off), applied after the vertex cover refinement, to
     * all components at once. Every node of a component (with >= 3
     * levels) gets a part: S (sepflag), A (BFS layer <= lo) or B
     * (layer > lo); BFS layers only touch adjacent layers, so this is
     * a valid separator. In each pass:
     *  - propose: a separator node v can move to side X, then its
     *    neighbors in the other side are pulled into S. The gain is
     *    1 - (number pulled in). Moves are proposed if the gain is
     *    positive, or zero and the move improves the balance, and if
     *    min(|A|,|B|) does not drop below STRUMPACK_ND_FM_BAL
     *    (|A|+|B|) (default 0.3), unless it does not get worse,
     *  - select: two adjacent separator nodes moving to opposite
     *    sides would create an A-B edge; then only the one with the
     *    larger (gain, -index) moves (deterministic),
     *  - apply the moves.
     * The separator stays valid and never grows. The counts per
     * component are computed once (fm_count_kernel) and then updated
     * incrementally by fm_apply_ftor. fm_apply_ftor sets *moved = 1
     * if any node moved in the pass (one flag per pass, so that
     * several passes can be launched before the host checks for
     * convergence, STRUMPACK_ND_FM_BATCH).
     */
    template<typename integer> struct fm_ctx {
      const integer *ptr, *ind;
      const bool* elim;
      const int *comp, *levels, *sep_level, *sep_side, *label;
      int *flag, *part, *side, *gain, *mv, *cnt;
      int* moved;
      float bal;
      // hill climbing (coarse levels, STRUMPACK_ND_ML_NEG > 0): gain >=
      // -neg are allowed, nodes that moved are locked until the end
      // of the round (lock != nullptr), zero gain moves need not
      // improve the balance
      int* lock = nullptr;
      int neg = 0;
      // node weights (multilevel refinement on coarse graphs);
      // nullptr: every node has weight 1. Counts, gains and the
      // balance check are then sums of weights.
      const int* vw = nullptr;
      __device__ int w(int i) const { return vw ? vw[i] : 1; }
      __device__ bool ok(int nA, int nB, int a, int b) const {
        if (a <= 0 || b <= 0) return false;
        int omin = nA < nB ? nA : nB, nmin = a < b ? a : b;
        return nmin >= omin || float(nmin) >= bal * float(a + b);
      }
    };
    template<typename integer> struct fm_init_ftor {
      fm_ctx<integer> x;
      __device__ void operator()(int i) const {
        int r = -1;
        if (!x.elim[i]) {
          int c = x.comp[i];
          if (x.levels[c] >= 3) {
            int lo = x.sep_side[c] ? x.sep_level[c] - 1 : x.sep_level[c];
            r = x.flag[i] ? 2 : (x.label[i] <= lo ? 0 : 1);
          }
        }
        x.part[i] = r;
      }
    };
    /**
     * Counts of A, B, S per component (cnt must be zeroed). As in
     * count_comps_kernel, the counts of the component of the first
     * participating node in a block are accumulated in shared
     * memory: early in the dissection there are only a few
     * components, and global atomics on the same 3 counters from
     * every node are very slow.
     */
    __global__ void fm_count_kernel(int n, const int* part, const int* comp,
                                    int* cnt, const int* vw = nullptr) {
      __shared__ int sc, scnt[3];
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (threadIdx.x == 0) { sc = -1; scnt[0] = scnt[1] = scnt[2] = 0; }
      __syncthreads();
      int r = -1, c = -1;
      if (i < n) { r = part[i]; if (r >= 0) c = comp[i]; }
      if (r >= 0) atomicCAS(&sc, -1, c);
      __syncthreads();
      if (r >= 0) {
        int wi = vw ? vw[i] : 1;
        if (c == sc) atomicAdd(scnt + r, wi);
        else atomicAdd(cnt + 3*c + r, wi);
      }
      __syncthreads();
      if (threadIdx.x < 3 && sc >= 0 && scnt[threadIdx.x])
        atomicAdd(cnt + 3*sc + threadIdx.x, scnt[threadIdx.x]);
    }
    template<typename integer> struct fm_propose_ftor {
      fm_ctx<integer> x;
      __device__ void operator()(int v) const {
        x.side[v] = -1; x.mv[v] = 0;
        if (x.part[v] != 2) return;
        if (x.lock && x.lock[v]) return;
        int c = x.comp[v], nA = x.cnt[3*c], nB = x.cnt[3*c+1];
        // kA, kB: weight of the neighbors in A, B (pulled into S
        // when v moves to the other side)
        int kA = 0, kB = 0;
        for (auto k=x.ptr[v], khi=x.ptr[v+1]; k<khi; k++) {
          int u = x.ind[k], r = x.part[u];
          if (r == 0) kA += x.w(u);
          else if (r == 1) kB += x.w(u);
        }
        const int wv = x.w(v);
        int gA = wv - kB, gB = wv - kA;
        bool okA = x.ok(nA, nB, nA + wv, nB - kB),
          okB = x.ok(nA, nB, nA - kA, nB + wv);
        int X = -1, g = 0;
        if (okA && (!okB || gA > gB || (gA == gB && nA <= nB))) {
          X = 0; g = gA;
        } else if (okB) { X = 1; g = gB; }
        if (X < 0 || g < -x.neg) return;
        if (g == 0 && !x.lock) { // only if the balance improves
          int a = X == 0 ? nA + wv : nA - kA, b = X == 0 ? nB - kB : nB + wv;
          int omin = nA < nB ? nA : nB, nmin = a < b ? a : b;
          if (nmin <= omin) return;
        }
        x.side[v] = X;
        x.gain[v] = g;
      }
    };
    template<typename integer> struct fm_select_ftor {
      fm_ctx<integer> x;
      __device__ void operator()(int v) const {
        int X = x.side[v];
        if (X < 0) return;
        int g = x.gain[v];
        for (auto k=x.ptr[v], khi=x.ptr[v+1]; k<khi; k++) {
          int u = x.ind[k];
          if (x.part[u] != 2) continue;
          int Y = x.side[u];
          // opposite sides conflict; a negative gain move also
          // yields to any better neighboring move, so that the
          // separator does not grow everywhere at once
          if (Y < 0 || (Y == X && g >= 0)) continue;
          int gu = x.gain[u];
          if (gu > g || (gu == g && u < v)) return;
        }
        x.mv[v] = 1;
      }
    };
    template<typename integer> struct fm_apply_ftor {
      fm_ctx<integer> x;
      __device__ void operator()(int v) const {
        if (!x.mv[v]) return;
        int X = x.side[v], Y = 1 - X;
        x.part[v] = X;
        if (x.lock) x.lock[v] = 1;
        // neighbors in the other side join the separator. Only nodes
        // in S move, and adjacent S nodes never move to opposite
        // sides (fm_select_ftor), so a node in side Y can only be
        // changed to S, and the CAS counts each pulled node once.
        // pulled: weight of the nodes pulled into S
        int pulled = 0;
        for (auto k=x.ptr[v], khi=x.ptr[v+1]; k<khi; k++) {
          int u = x.ind[k];
          if (x.part[u] == Y && atomicCAS(x.part + u, Y, 2) == Y)
            pulled += x.w(u);
        }
        const int wv = x.w(v);
        int* cc = x.cnt + 3*x.comp[v];
        atomicAdd(cc + X, wv);
        if (pulled != wv) atomicAdd(cc + 2, pulled - wv);
        if (pulled) atomicAdd(cc + Y, -pulled);
        *x.moved = 1;
      }
    };
    /**
     * Hill climbing FM: per component, remember the best state seen
     * (smallest separator, ties broken by larger min(|A|,|B|)).
     * fm_best_comp_ftor runs per component on the counts, and marks
     * improved components; fm_copy_best_ftor copies their labels;
     * fm_restore_ftor restores the best labels and clears the locks.
     */
    struct fm_best_comp_ftor {
      const int* cnt;
      int *bestS, *bestMin, *improved;
      bool* flag;
      __device__ void operator()(int c) const {
        int S = cnt[3*c+2], m = min(cnt[3*c], cnt[3*c+1]);
        int imp = S < bestS[c] || (S == bestS[c] && m > bestMin[c]);
        improved[c] = imp;
        if (imp) { bestS[c] = S; bestMin[c] = m; *flag = true; }
      }
    };
    struct fm_copy_best_ftor {
      const int *part, *comp, *improved;
      int* best;
      __device__ void operator()(int i) const {
        int r = part[i];
        if (r >= 0 && improved[comp[i]]) best[i] = r;
      }
    };
    struct fm_restore_ftor {
      int* part;
      const int* best;
      int* lock;
      __device__ void operator()(int i) const {
        if (part[i] >= 0) part[i] = best[i];
        lock[i] = 0;
      }
    };
    template<typename integer> struct fm_final_ftor {
      fm_ctx<integer> x;
      __device__ void operator()(int i) const {
        int r = x.part[i];
        if (r >= 0) x.flag[i] = (r == 2) ? 1 : 0;
      }
    };

    /**
     * Multilevel separators (STRUMPACK_ND_ML=2), for all components
     * with at least STRUMPACK_ND_ML_MIN nodes at once:
     *  - coarsening (ml_build): handshake heavy edge matching (each
     *    node picks its heaviest unmatched neighbor, ties broken by a
     *    symmetric hash, pairs that pick each other are matched),
     *    with a cap on the node weight. The coarse graph is built by
     *    sorting edge keys, as in nd_compress_graph, with summed edge
     *    weights.
     *  - separator search on the coarsest graph (ml_search), then
     *    refinement (ml_down): weighted FM on each coarse level and
     *    projection to the next finer level (one gather). A coarse
     *    move shifts a whole cluster, so the separator can move many
     *    hops per pass. A projected separator is valid: coarse nodes
     *    in A and B are not adjacent, so none of their members are.
     */
    template<typename integer> struct ml_level {
      int m = 0;                              // number of nodes
      thrust::device_vector<integer> ptr, ind;
      thrust::device_vector<int> ew,          // edge weights
        vw, comp, part,                       // per node
        cmap;     // node of the finer level -> node of this level
    };
    __device__ inline unsigned ml_hash(int a, int b) {
      unsigned lo = unsigned(a < b ? a : b), hi = unsigned(a < b ? b : a);
      unsigned x = lo * 2654435761u ^ (hi + 0x9e3779b9u) * 2246822519u;
      x ^= x >> 15; x *= 2654435761u; x ^= x >> 13;
      return x;
    }
    // level 0: node i takes part if it has a part (component with
    // >= 3 levels) and its component has at least mmin nodes
    struct ml_act0_ftor {
      const int *part, *comp, *cs;
      int mmin;
      int* act;
      __device__ void operator()(int i) const {
        int r = part[i];
        act[i] = (r >= 0 && cs[comp[i]+1] - cs[comp[i]] >= mmin) ? 1 : 0;
      }
    };
    // coarse levels: keep matching while the component is large
    struct ml_actc_ftor {
      const int *comp, *csz;
      int cmin;
      int* act;
      __device__ void operator()(int i) const {
        act[i] = csz[comp[i]] > cmin ? 1 : 0;
      }
    };
    template<typename integer> struct ml_propose_ftor {
      const integer *ptr, *ind;
      const int *ew, *vw, *part, *act, *mate;
      int* choice;
      int cap;
      __device__ void operator()(int i) const {
        choice[i] = -1;
        if (!act[i] || mate[i] >= 0) return;
        // part == nullptr: no restriction (STRUMPACK_ND_ML=2)
        const int wi = vw ? vw[i] : 1, pi = part ? part[i] : 0;
        int best = -1, bw = -1;
        unsigned bh = 0;
        for (auto k=ptr[i], khi=ptr[i+1]; k<khi; k++) {
          int j = ind[k];
          if (j == i || !act[j] || mate[j] >= 0 || (part && part[j] != pi))
            continue;
          if (wi + (vw ? vw[j] : 1) > cap) continue;
          int w = ew ? ew[k] : 1;
          unsigned h = ml_hash(i, j);
          if (w > bw || (w == bw && h > bh)) { best = j; bw = w; bh = h; }
        }
        choice[i] = best;
      }
    };
    struct ml_match_ftor {
      const int* choice;
      int* mate;
      __device__ void operator()(int i) const {
        int j = choice[i];
        if (j >= 0 && mate[i] < 0 && choice[j] == i) mate[i] = j;
      }
    };
    // one coarse node per matched pair (the lower index) or single
    struct ml_lead_ftor {
      const int *keep, *mate;
      int* lead;
      __device__ void operator()(int i) const {
        lead[i] = ((!keep || keep[i]) && (mate[i] < 0 || i < mate[i]))
          ? 1 : 0;
      }
    };
    struct ml_cmap_ftor {
      const int *keep, *mate, *cid;
      int* cmap;
      __device__ void operator()(int i) const {
        if (keep && !keep[i]) { cmap[i] = -1; return; }
        int j = mate[i];
        cmap[i] = cid[(j >= 0 && j < i) ? j : i];
      }
    };
    struct ml_cattr_ftor {
      const int *lead, *mate, *cid, *vw, *comp, *part;
      int *cvw, *ccomp, *cpart;
      __device__ void operator()(int i) const {
        if (!lead[i]) return;
        int c = cid[i], j = mate[i];
        cvw[c] = (vw ? vw[i] : 1) + (j >= 0 ? (vw ? vw[j] : 1) : 0);
        ccomp[c] = comp[i];
        cpart[c] = part ? part[i] : 0;
      }
    };
    // edge keys (coarse row << 32 | coarse col), ~0 for edges that
    // are dropped (to a node that does not take part, or inside a
    // coarse node); one key per edge of the finer level
    template<typename integer> struct ml_ekey_ftor {
      const integer *ptr, *ind;
      const int *ew, *cmap;
      unsigned long long* key;
      int* val;
      __device__ void operator()(int i) const {
        int ci = cmap[i];
        for (auto k=ptr[i], khi=ptr[i+1]; k<khi; k++) {
          int cj = ci < 0 ? -1 : cmap[ind[k]];
          key[k] = (ci < 0 || cj < 0 || cj == ci) ? ~0ull :
            ((static_cast<unsigned long long>(ci) << 32) |
             static_cast<unsigned long long>(unsigned(cj)));
          val[k] = ew ? ew[k] : 1;
        }
      }
    };
    template<typename integer> struct ml_low32_ftor {
      __host__ __device__ integer operator()(unsigned long long k) const
      { return integer(k & 0xffffffffull); }
    };
    template<typename integer> struct ml_high32_ftor {
      __host__ __device__ integer operator()(unsigned long long k) const
      { return integer(k >> 32); }
    };
    struct ml_proj_ftor {
      const int *cmap, *cpart;
      int* part;
      __device__ void operator()(int i) const {
        int c = cmap[i];
        if (c >= 0) part[i] = cpart[c];
      }
    };
    // number of nodes per component (block aggregated, see
    // count_comps_kernel)
    __global__ void ml_count_kernel(int m, const int* comp, int* cnt) {
      __shared__ int sc, scnt;
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (threadIdx.x == 0) { sc = -1; scnt = 0; }
      __syncthreads();
      int c = i < m ? comp[i] : -1;
      if (c >= 0) atomicCAS(&sc, -1, c);
      __syncthreads();
      if (c >= 0) {
        if (c == sc) atomicAdd(&scnt, 1);
        else atomicAdd(cnt + c, 1);
      }
      __syncthreads();
      if (threadIdx.x == 0 && scnt) atomicAdd(cnt + sc, scnt);
    }

    /**
     * STRUMPACK_ND_ML=2: separator search on the coarsest graph. The
     * coarse nodes keep the component number of the graph being
     * ordered, and the BFS labels use the same offsets comp_start[c]
     * (a coarse component is never larger than the fine one), so the
     * layer and separator count arrays of size n can be reused, and
     * n is the "unvisited" label.
     */
    struct ml_minroot_ftor {
      const int* comp;
      int* root;
      __device__ void operator()(int i) const { atomicMin(root + comp[i], i); }
    };
    // random restart root per component (components without coarse
    // nodes are skipped)
    struct ml_rand_root_ftor {
      unsigned seed;
      const int *cst, *map;
      int* root;
      __device__ void operator()(int c) const {
        int a = cst[c], s = cst[c+1] - a;
        if (s <= 0) return;
        thrust::minstd_rand g(seed * 2654435761u + unsigned(c) * 40503u + 1u);
        g.discard(4);
        thrust::uniform_int_distribution<int> d(0, s-1);
        root[c] = map[a + d(g)];
      }
    };
    // parts from the chosen layer: 2 = S, 0 = A, 1 = B, as in
    // mark_sep_ftor / fm_init_ftor; -1 for components with < 3 levels
    template<typename integer> struct ml_cpart_ftor {
      const integer *ptr, *ind;
      const int *comp, *levels, *label, *seplev, *sepside;
      int* part;
      __device__ void operator()(int i) const {
        int c = comp[i];
        if (levels[c] < 3) { part[i] = -1; return; }
        int sl = seplev[c], l = label[i], up = sepside[c];
        if (l == sl) {
          int lt = up ? l - 1 : l + 1;
          for (auto k=ptr[i], khi=ptr[i+1]; k<khi; k++)
            if (label[ind[k]] == lt) { part[i] = 2; return; }
        }
        part[i] = (l <= (up ? sl - 1 : sl)) ? 0 : 1;
      }
    };
    // per component: use the multilevel separator (counts c2) if its
    // normalized separator S (1/A + 1/B) is smaller than that of the
    // BFS separator (counts c1)
    struct ml_pick_ftor {
      const int *c1, *c2;
      int* pick;
      __device__ void operator()(int c) const {
        int A1 = c1[3*c], B1 = c1[3*c+1], S1 = c1[3*c+2];
        int A2 = c2[3*c], B2 = c2[3*c+1], S2 = c2[3*c+2];
        int r = 0;
        if (A2 > 0 && B2 > 0) {
          if (A1 <= 0 || B1 <= 0) r = 1;
          else r = double(S2) * (1.0 / A2 + 1.0 / B2) <
                 double(S1) * (1.0 / A1 + 1.0 / B1);
        }
        pick[c] = r;
      }
    };
    struct ml_apply_pick_ftor {
      const int *comp, *pick, *src;
      int* dst;
      __device__ void operator()(int i) const {
        int r = src[i];
        if (r >= 0 && pick[comp[i]]) dst[i] = r;
      }
    };

    /**
     * Compute the nested dissection ordering perm (new -> old) on the
     * device. Also returns the ND tree structure: tnode[i] is the
     * tree node in which graph node i was eliminated, and
     * tree_parent[t] is the parent of tree node t (-1 for a root).
     * Tree node ids increase with the ND step, so children always
     * have larger ids than their parent.
     */
    template<typename integer> void
    nd_bfs_device(int n, thrust::device_vector<integer>& ptr,
                  thrust::device_vector<integer>& ind,
                  thrust::device_vector<integer>& perm,
                  thrust::device_vector<int>& tnode,
                  std::vector<int>& tree_parent) {
      int nr_elim = 0;        // nr of already eliminated nodes
      thrust::device_vector<int>
        label(n),             // node label: union-find parent when
                              // finding connected components, BFS
                              // label in the level sets
        comp(n),              // number of the component for each node
        degree(n),            // node degrees, used to find a new
                              // peripheral node, which is a node in
                              // the last level with the smallest
                              // degree. Also used to store the
                              // caridnalities of the levels
        levels(n),            // number of levels in the level sets
                              // (one value per component)
        old_levels(n),        // previous number of leves in the level
                              // sets (one value per component)
        comp_start(n+1),      // start index of each component (one
                              // value per component), this is an
                              // exclusive scan of the sizes of the
                              // components
        comp_root(n),         // start points for the level set BFS
                              // for each component, also used to
                              // store root of nodes
        comp_root_opt(n),     // start points for the level set BFS
                              // for each component, also used to
                              // store root of nodes
        sep_size(n),          // size of the separator corresponding
                              // to each level in the level set, where
                              // the separator is the subset of nodes
                              // in the level that are connected to
                              // the next level
        sep_level(n);         // the optimal level from the level sets
                              // to use as a separator, for each
                              // component
      thrust::device_vector<float>
        min_norm_sep(n);      // the minimal normalized separator
                              // value, for each component
      thrust::device_vector<bool>
        eliminated(n, false), // bool flags for each node indicating
                              // if the node has been eliminated yet
        running(1);           // bool flag to check whether the BFS
                              // has completed
      thrust::device_vector<int>
        tparent(n, -1),       // tree node of the separator above each
                              // live node (see tag_tree_ftor)
        elist(n), ekey(n);    // nodes eliminated in an ND step, and
                              // their component (sort key)
      // per component best score and node, for the deterministic
      // argmin selection of BFS roots (see argmin_score_ftor)
      thrust::device_vector<int> pbest(n+1), pnode(n+1);
      thrust::device_vector<int>
        sep_side(n, 0);       // side of the chosen separator layer
                              // (always 0, see minimize_sep_ftor)
      // STRUMPACK_ND_REPS: number of BFS restarts (pseudo-peripheral
      // start, then random starts) when searching for a separator,
      // default 4 (was 2)
      static const int nreps =
        std::max(1, nd_env_int("STRUMPACK_ND_REPS", 4));
      // STRUMPACK_ND_BAL: balance constraint, e.g. 0.2 rejects
      // separators with min(|A|,|B|) < 0.2 (|A|+|B|) when possible
      static const float sep_bal = [] {
        auto s = std::getenv("STRUMPACK_ND_BAL");
        return s ? float(std::atof(s)) : 0.f;
      }();
      // STRUMPACK_ND_VC=t: refine separators with at least t nodes to
      // a (minimum) vertex cover between the two adjacent layers,
      // default 1 (all separators), 0 disables the refinement.
      static const int vc_thresh = nd_env_int("STRUMPACK_ND_VC", 1);
      static const int vc_max_rounds =
        std::max(1, nd_env_int("STRUMPACK_ND_VC_ROUNDS", 64));
      // STRUMPACK_ND_VC_PHASES: max number of augmenting path phases
      // after the greedy matching (0: greedy matching only). On
      // Geo_1438, 10 phases recover ~96% of the separator reduction
      // of an exact maximum matching; more phases mostly cost time.
      static const int vc_max_phases =
        std::max(0, nd_env_int("STRUMPACK_ND_VC_PHASES", 10));
      thrust::device_vector<int> sepflag(n),
        vc_mate(vc_thresh > 0 ? n : 0),
        vc_best(vc_thresh > 0 ? n : 0),
        vc_z(vc_thresh > 0 ? n : 0),
        vc_cnt(vc_thresh > 0 ? 2*n : 0),
        vc_aux(vc_thresh > 0 && vc_max_phases > 0 ? 4*n : 0);
      // STRUMPACK_ND_FM=passes: FM style separator refinement after
      // the vertex cover (see fm_ctx), default 100, 0 disables it.
      // Passes are cheap (no recount, no host sync per pass), and
      // FM stops early when nothing moves; 100 instead of 10 passes
      // costs < 0.15 s and gives -5% fill on Geo_1438 (ND_TODO.md).
      // STRUMPACK_ND_FM_BAL: balance limit for the FM moves.
      static const int fm_passes =
        std::max(0, nd_env_int("STRUMPACK_ND_FM", 100));
      static const float fm_bal = [] {
        auto s = std::getenv("STRUMPACK_ND_FM_BAL");
        return s ? float(std::atof(s)) : 0.3f;
      }();
      // part, side, gain, mv (n each), counts (3 per component)
      thrust::device_vector<int>
        fm_buf(fm_passes > 0 && vc_thresh > 0 ? 7*n : 0);
      // STRUMPACK_ND_FM_BATCH: number of FM passes launched before
      // the host checks whether the last one moved anything (default
      // 8). Passes after convergence do not change anything.
      static const int fm_batch =
        std::max(1, nd_env_int("STRUMPACK_ND_FM_BATCH", 8));
      // one "moved" flag per FM pass
      thrust::device_vector<int>
        fm_pflag(fm_passes > 0 && vc_thresh > 0 ? fm_passes : 0);
      // STRUMPACK_ND_ML=2 (default 0; any value > 0 means 2):
      // multilevel separators (see ml_level), for
      // components with >= STRUMPACK_ND_ML_MIN nodes (default 5000).
      // STRUMPACK_ND_ML_COARSE: stop coarsening a component at this
      // many nodes (default 100). STRUMPACK_ND_ML_LEVELS: max coarse
      // levels (default 16; with 10 the coarsest graph still had
      // ~2.5k nodes, 16 gave ~-2% fill). STRUMPACK_ND_ML_ROUNDS: handshake
      // matching rounds per level (default 4). STRUMPACK_ND_ML_FM:
      // FM passes per coarse level (default 30).
      static const int ml_mode = nd_env_int("STRUMPACK_ND_ML", 0);
      static const int ml_min =
        std::max(1, nd_env_int("STRUMPACK_ND_ML_MIN", 5000));
      static const int ml_coarse =
        std::max(1, nd_env_int("STRUMPACK_ND_ML_COARSE", 100));
      static const int ml_levels =
        std::max(1, nd_env_int("STRUMPACK_ND_ML_LEVELS", 16));
      static const int ml_rounds =
        std::max(1, nd_env_int("STRUMPACK_ND_ML_ROUNDS", 4));
      const bool ml_on = ml_mode > 0 && fm_passes > 0 && vc_thresh > 0;
      const int ml_fm = std::min
        (std::max(1, nd_env_int("STRUMPACK_ND_ML_FM", 30)), fm_passes);
      // a level that shrinks the graph by less than this is dropped,
      // and ends the coarsening
      const double ml_shrink = 0.9;
      // matching work arrays, allocated once (coarse levels are
      // smaller than n), and the hierarchy (reused at every ND step)
      thrust::device_vector<int> ml_act(ml_on ? n : 0),
        ml_mate(ml_on ? n : 0), ml_choice(ml_on ? n : 0),
        ml_lead(ml_on ? n : 0), ml_cid(ml_on ? n : 0),
        ml_csz(ml_on ? n : 0), ml_val, ml_uval;
      thrust::device_vector<unsigned long long> ml_key, ml_ukey;
      std::vector<ml_level<integer>> mlv(ml_on ? ml_levels : 0);
      double t_ml = 0, t_ml_build = 0, t_ml_search = 0, t_ml_down = 0;
      long long ml_saved = 0;
      // STRUMPACK_ND_ML=2: hierarchy without part restriction, the
      // separator is computed on the coarsest graph (see ml_search),
      // projected and refined back, and used per component if better
      // than the BFS separator. STRUMPACK_ND_ML_BAL: balance
      // constraint on the coarse graph (default 0),
      // STRUMPACK_ND_ML_REPS: BFS restarts on the coarse graph
      // (default 4, same fill as 8 and cheaper).
      static const float ml_bal = [] {
        auto s = std::getenv("STRUMPACK_ND_ML_BAL");
        return s ? float(std::atof(s)) : 0.f;
      }();
      static const int ml_reps =
        std::max(1, nd_env_int("STRUMPACK_ND_ML_REPS", 4));
      // STRUMPACK_ND_ML_NEG=g (default 1, 0 = off): hill climbing FM
      // on the coarse levels (see fm_best_comp_ftor), after the plain
      // coarse FM passes: moves with gain >= -g, rounds end after
      // STRUMPACK_ND_ML_CLIMB (default 3) passes without improvement,
      // every component is restored to its best state. With
      // ML_LEVELS=16: -3% fill on Serena and Geo_1438 (ND_TODO.md).
      // These ML_* defaults only matter with STRUMPACK_ND_ML=2.
      static const int ml_neg =
        std::max(0, nd_env_int("STRUMPACK_ND_ML_NEG", 1));
      static const int ml_climb =
        std::max(1, nd_env_int("STRUMPACK_ND_ML_CLIMB", 3));
      // best part, lock (n each), bestS, bestMin, improved (per comp)
      thrust::device_vector<int> ml_hc(ml_on && ml_neg > 0 ? 5*n : 0);
      thrust::device_vector<bool> ml_hc_flag(1);
      const int n2 = ml_on ? n : 0;
      thrust::device_vector<int> cl_label(n2), cl_levels(n2), cl_old(n2),
        cl_root(n2), cl_root_opt(n2), cl_seplev(n2), cl_sepside(n2),
        cl_layer(n2), cl_sepsz(n2), cl_d1(n2), cl_d2(n2),
        cl_d3(n2), cl_lev1(n2), cl_lev2(n2), cl_r2(n2), cl_r3(n2),
        cl_optr2(n2), cl_optmode(n2), cl_cst(n2 ? n+1 : 0), cl_map(n2),
        cl_key(n2), ml_fpart(n2), ml_cnt;
      thrust::device_vector<float> cl_ns(n2);
      thrust::device_vector<bool> cl_elim(n2, false);
      // STRUMPACK_ND_TWOEND (default 3, 0 disables): after each
      // restart's pseudo-peripheral search, also score two-end
      // level structures (see twoend_combine_ftor).
      //  1: the pair (r1, r2), r1 the pseudo-peripheral node, r2 a
      //     min degree node in the last layer of the BFS from r1
      //  3: as 1, plus the pairs (r1, r3) and (r2, r3), r3 = node
      //     maximizing min(d1, d2) (off the r1-r2 axis). Default:
      //     -4% fill on Serena, -2% on Geo_1438, ~+0.3 s ordering
      //     time on the scalar test problems (see ND_TODO.md).
      //  (mode 2, r3 = far end of the BFS from r2, was removed:
      //   r3 == r1 in ~96% of the cases on Serena)
      // STRUMPACK_ND_ROOT_MIX (default 1, 0 disables): different
      // random restart root seed per component.
      static const int te_mode = nd_env_int("STRUMPACK_ND_TWOEND", 3);
      static const bool two_end = te_mode != 0;
      static const bool root_mix =
        nd_env_int("STRUMPACK_ND_ROOT_MIX", 1) != 0;
      thrust::device_vector<int>
        te_dist1(two_end ? n : 0), te_lev1(two_end ? n : 0),
        te_root2(two_end ? n : 0), te_opt_root2(two_end ? n : 0),
        te_opt_mode(two_end ? n : 0),
        te_dist2(te_mode == 3 ? n : 0), te_lev2(te_mode == 3 ? n : 0),
        te_root3(te_mode == 3 ? n : 0), te_dist3(te_mode == 3 ? n : 0);
      long long vc_saved = 0, fm_saved = 0, fm_moves = 0;
      double t_vc = 0, t_md = 0;
      using clk = std::chrono::steady_clock;
      auto secs = [](clk::time_point a) {
        return std::chrono::duration<double>(clk::now() - a).count();
      };
      // STRUMPACK_ND_SEED: seed for the random BFS restart roots
      // (STRUMPACK_ND_REPS > 1). Default: the default minstd_rand
      // seed, i.e. the same roots sequence in every run.
      static const int nd_seed = nd_env_int("STRUMPACK_ND_SEED", 0);
      thrust::minstd_rand rng;
      if (nd_seed > 0) rng.seed(nd_seed);
      auto p = [](auto& v) {
        return thrust::raw_pointer_cast(v.data());
      };
      // execution policy for all thrust calls in the ND loop: no
      // device synchronization, cached temporary storage
      nd_cache_alloc alloc;
      const auto pol = STRUMPACK_ND_NOSYNC(alloc);
      auto run = [&](const auto& f) {
        thrust::for_each_n(pol, thrust::counting_iterator<int>(0), n, f);
      };
      // STRUMPACK_ND_BFS_BATCH: number of BFS layers (or matching /
      // alternating BFS rounds in the vertex cover) launched before
      // checking on the host whether the search has finished. Extra
      // launches after convergence do not change anything.
      static const int bfs_batch =
        std::max(1, nd_env_int("STRUMPACK_ND_BFS_BATCH", 8));
      // STRUMPACK_ND_MD_MAX: components with at most this many nodes
      // (of the graph being ordered, i.e. groups when the graph is
      // compressed) are ordered with minimum degree. At most
      // MD_SWITCH (the minimum_degree kernel block size).
      static const int md_max = std::min
        (MD_SWITCH, std::max(0, nd_env_int("STRUMPACK_ND_MD_MAX", MD_SWITCH)));
      // STRUMPACK_ND_MD_FILL: pivot rule of the minimum_degree
      // kernel, 0: minimum degree, 1: minimum degree with minimum
      // fill tie breaking, 2 (default): minimum fill (see
      // minimum_degree). Minimum fill gave less fill on every tested
      // problem (-0.2% to -1.6%), at the same ordering and factor
      // time (ND_TODO.md, "Minimum-degree leaves").
      static const int md_fill_mode = std::min
        (2, std::max(0, nd_env_int("STRUMPACK_ND_MD_FILL", 2)));
      auto clear_running = [&]() {
        thrust::fill_n(pol, running.begin(), 1, false);
      };
      const bool stats = nd_stats_enabled();
      // STRUMPACK_ND_STATS: time per phase of the ND loop. The device
      // is synchronized at every phase boundary, so only with stats.
      enum { PH_CC, PH_MD, PH_SEARCH, PH_REBUILD, PH_VC, PH_FM, PH_ELIM,
             PH_N };
      const char* ph_name[PH_N] = {"components", "min-degree",
        "separator search", "rebuild", "vertex cover", "FM (+ML)",
        "elimination"};
      double t_ph[PH_N] = {0};
      auto tph0 = clk::now();
      auto phase_end = [&](int ph) {
        if (!stats) return;
        STRUMPACK_ND_DEVICE_SYNC();
        auto t = clk::now();
        t_ph[ph] += std::chrono::duration<double>(t - tph0).count();
        tph0 = t;
      };
      int nd_step = 0;
      // STRUMPACK_ND_CAND=k: candidate diagnostics (host side, slow)
      // for the first k ND steps, for the largest component: every
      // BFS layer cut considered (S, |A|, |B|, balance, normalized
      // separator, boundary nodes = nodes with an eliminated
      // neighbor), and the chosen cut after BFS / VC / FM.
      static const int cand_steps =
        std::max(0, nd_env_int("STRUMPACK_ND_CAND", 0));
      std::vector<integer> h_ptr, h_ind;
      std::vector<int> h_lab, h_comp, h_cs, h_part;
      std::vector<char> h_elim, h_bnd;
      int cand_c = -1;
      auto cand_load = [&](int nr_comps) {
        if (h_ptr.empty()) {
          h_ptr.resize(ptr.size()); h_ind.resize(ind.size());
          thrust::copy(ptr.begin(), ptr.end(), h_ptr.begin());
          thrust::copy(ind.begin(), ind.end(), h_ind.begin());
          h_lab.resize(n); h_comp.resize(n); h_part.resize(n);
          h_elim.resize(n); h_bnd.resize(n);
        }
        thrust::host_vector<bool> he(eliminated);
        for (int i=0; i<n; i++) h_elim[i] = he[i];
        thrust::copy(comp.begin(), comp.end(), h_comp.begin());
        h_cs.resize(nr_comps+1);
        thrust::copy_n(comp_start.begin(), nr_comps+1, h_cs.begin());
        long long nb = 0;
        for (int i=0; i<n; i++) {
          h_bnd[i] = 0;
          if (h_elim[i]) continue;
          for (auto k=h_ptr[i]; k<h_ptr[i+1]; k++)
            if (h_elim[h_ind[k]]) { h_bnd[i] = 1; nb++; break; }
        }
        cand_c = 0;
        for (int c=1; c<nr_comps; c++)
          if (h_cs[c+1]-h_cs[c] > h_cs[cand_c+1]-h_cs[cand_c]) cand_c = c;
        std::cout << "# CANDSTEP step=" << nd_step << " comps=" << nr_comps
                  << " largest=" << h_cs[cand_c+1]-h_cs[cand_c]
                  << " live_bnd_total=" << nb << std::endl;
      };
      auto cand_print = [&](int rep, int it) {
        const int c = cand_c, c0 = h_cs[c], N = h_cs[c+1] - c0;
        const int L = levels[c];
        if (L < 3) return;
        thrust::copy(label.begin(), label.end(), h_lab.begin());
        std::vector<int> ls(L), ss(L);
        std::vector<long long> bl(L, 0);
        thrust::copy_n(degree.begin()+c0, L, ls.begin());
        thrust::copy_n(sep_size.begin()+c0, L, ss.begin());
        long long Bt = 0;
        for (int i=0; i<n; i++)
          if (!h_elim[i] && h_comp[i] == c && h_bnd[i]) {
            bl[h_lab[i]-c0]++; Bt++;
          }
        std::cout << "# CAND step=" << nd_step << " rep=" << rep
                  << " it=" << it << " N=" << N << " L=" << L
                  << " bnd=" << Bt << std::endl
                  << "#   l S A B bal NS bndL bndA bndB" << std::endl;
        long long cA = ls[0], cB = N - ls[0], cb = bl[0];
        int bl1 = -1, bl2 = -1, bl3 = -1;
        float ns1 = 0, ns2 = 0; long long s3 = 0;
        for (int l=1; l<L-1; l++) {
          long long cL = ls[l]; cB -= cL; cb += bl[l];
          long long S = ss[l], A = cA + cL - S, B = cB;
          cA += cL;
          if (A <= 0 || B <= 0) continue;
          float ns = S * (1.f/A + 1.f/B);
          double bal = double(std::min(A, B)) / double(A + B);
          std::cout << "#   " << l << " " << S << " " << A << " " << B
                    << " " << std::setprecision(3) << bal << " " << ns
                    << " " << bl[l] << " " << cb << " " << Bt - cb
                    << std::endl;
          if (bl1 < 0 || ns < ns1) { bl1 = l; ns1 = ns; }
          if (bal >= 0.2 && (bl2 < 0 || ns < ns2)) { bl2 = l; ns2 = ns; }
          if (bal >= 0.2 && (bl3 < 0 || S < s3)) { bl3 = l; s3 = S; }
        }
        std::cout << "# CANDSUM step=" << nd_step << " rep=" << rep
                  << " it=" << it << " N=" << N << " L=" << L
                  << " best_l=" << bl1 << " best_NS=" << ns1
                  << " best_bal02_l=" << bl2 << " NS=" << ns2
                  << " minS_bal02_l=" << bl3 << " S=" << s3 << std::endl;
      };
      // cut of the largest component: from flags (S) and BFS layers
      // (A: layer <= lo), or from the FM parts (0=A, 1=B, 2=S)
      auto cand_cut = [&](const char* tag,
                          const thrust::device_vector<int>& v,
                          bool is_part) {
        const int c = cand_c;
        if (levels[c] < 3) return;
        thrust::copy(label.begin(), label.end(), h_lab.begin());
        thrust::copy_n(v.begin(), n, h_part.begin());
        const int lo = sep_side[c] ? sep_level[c] - 1 : sep_level[c];
        long long cnt[3] = {0,0,0}, bnd[3] = {0,0,0};
        for (int i=0; i<n; i++) {
          if (h_elim[i] || h_comp[i] != c) continue;
          int r = is_part ? h_part[i] :
            (h_part[i] ? 2 : (h_lab[i] <= lo ? 0 : 1));
          if (r < 0) continue;
          cnt[r]++; if (h_bnd[i]) bnd[r]++;
        }
        long long A = cnt[0], B = cnt[1], S = cnt[2];
        std::cout << "# CANDCUT step=" << nd_step << " " << tag
                  << " mode=" << (two_end ? int(te_opt_mode[c]) : 0)
                  << " layer=" << sep_level[c] - h_cs[c]
                  << " S=" << S << " A=" << A << " B=" << B
                  << " bal=" << std::setprecision(3)
                  << double(std::min(A,B)) / double(std::max(1LL,A+B))
                  << " NS=" << (A>0 && B>0 ? S*(1.0/A+1.0/B) : -1.0)
                  << " bndA=" << bnd[0] << " bndS=" << bnd[2]
                  << " bndB=" << bnd[1] << std::endl;
      };
      auto run_m = [&](int m, const auto& f) {
        thrust::for_each_n(pol, thrust::counting_iterator<int>(0), m, f);
      };
      // root[c] = argmin over the m nodes (component cmp[i]) of the
      // score s, see argmin_score_ftor
      auto argmin_root = [&](int m, const int* cmp, int nr_comps,
                             const auto& s,
                             thrust::device_vector<int>& root) {
        using S = std::decay_t<decltype(s)>;
        thrust::fill_n(pol, pbest.begin(), nr_comps, ND_NONE);
        thrust::fill_n(pol, pnode.begin(), nr_comps, ND_NONE);
        run_m(m, argmin_score_ftor<S>{s, cmp, p(pbest)});
        run_m(m, argmin_node_ftor<S>{s, cmp, p(pbest), p(pnode)});
        run_m(nr_comps, argmin_root_ftor{p(pnode), p(root)});
      };
      // frontier queue BFS (see bfs_q_layer_kernel): counters qs,
      // layer offsets qb, queue q
      const int qlen = n + bfs_batch + 4;
      thrust::device_vector<int> bfs_qbuf(2*qlen + n);
      // BFS on a graph with m nodes from roots[c], c < nr; label must
      // be filled with n (unvisited) by the caller, levels is set
      auto bfs_run = [&](int m, const integer* gp, const integer* gi,
                         const bool* el, int* lab, const int* cmp,
                         const int* cs, int* lev, const int* roots,
                         int nr, bool check) {
        int *qs = p(bfs_qbuf), *qb = qs + qlen, *q = qb + qlen;
        thrust::fill_n(pol, bfs_qbuf.begin(), 2, 0);
        thrust::fill_n(pol, bfs_qbuf.begin()+qlen, 1, 0);
        thrust::fill_n(pol, thrust::device_pointer_cast(lev), nr, 1);
        run_m(nr, bfs_q_init_ftor{roots, cs, cmp, el, m, check, lab, q, qs});
        const int G = std::max(1, std::min((m+255)/256, 2048));
        int it = 0;
        do {
          for (int b=0; b<bfs_batch; b++, it++)
            bfs_q_layer_kernel<integer><<<G, 256>>>
              (it, n, gp, gi, el, lab, cmp, lev, q, qs, qb);
        } while (int(bfs_qbuf[it]) > 0);
      };
      auto cardinalities = [&](int m, const cardinalities_ftor<integer>& cf) {
        if (m > 0)
          cardinalities_kernel<integer><<<(m+255)/256, 256>>>(m, cf);
      };
      // separate random stream, so ML does not change the restarts of
      // the BFS separator search
      thrust::minstd_rand ml_rng(nd_seed > 0 ? unsigned(nd_seed) + 7777u
                                 : 7777u);
      // Coarse hierarchy of the components taking part (see
      // ml_act0_ftor and ml_level). Returns the number
      // of coarse levels built; the sizes are stored in ml_sizes.
      std::vector<int> ml_sizes;
      auto ml_build = [&](int nr_comps) -> int {
        int* fb = p(fm_buf);
        ml_sizes.clear();
        run(ml_act0_ftor{fb, p(comp), p(comp_start), ml_min, p(ml_act)});
        const int W = thrust::reduce(pol, ml_act.begin(), ml_act.end());
        if (W == 0) return 0;
        ml_sizes.push_back(W);
        // current (finer) level
        const integer *gptr = p(ptr), *gind = p(ind);
        const int *gew = nullptr, *gvw = nullptr, *gcomp = p(comp),
          *gpart = nullptr, *gkeep = p(ml_act);
        int m = n, mkept = W;
        long long E = ptr[n];
        int L = 0;
        while (L < ml_levels) {
          thrust::fill_n(pol, ml_mate.begin(), m, -1);
          const int cap = std::max(2, int(3.0 * W / mkept));
          for (int r=0; r<ml_rounds; r++) {
            run_m(m, ml_propose_ftor<integer>
                  {gptr, gind, gew, gvw, gpart, p(ml_act), p(ml_mate),
                   p(ml_choice), cap});
            run_m(m, ml_match_ftor{p(ml_choice), p(ml_mate)});
          }
          run_m(m, ml_lead_ftor{gkeep, p(ml_mate), p(ml_lead)});
          thrust::exclusive_scan
            (pol, ml_lead.begin(), ml_lead.begin()+m, ml_cid.begin());
          const int mc = int(ml_cid[m-1]) + int(ml_lead[m-1]);
          if (mc > ml_shrink * mkept) break;
          auto& lv = mlv[L];
          lv.m = mc;
          lv.cmap.resize(m);
          lv.vw.resize(mc); lv.comp.resize(mc); lv.part.resize(mc);
          run_m(m, ml_cmap_ftor{gkeep, p(ml_mate), p(ml_cid), p(lv.cmap)});
          run_m(m, ml_cattr_ftor{p(ml_lead), p(ml_mate), p(ml_cid), gvw,
                                 gcomp, gpart, p(lv.vw), p(lv.comp),
                                 p(lv.part)});
          // coarse graph: sort the edge keys, sum duplicate edges
          if ((long long)ml_key.size() < E) {
            ml_key.resize(E); ml_val.resize(E);
            ml_ukey.resize(E); ml_uval.resize(E);
          }
          run_m(m, ml_ekey_ftor<integer>
                {gptr, gind, gew, p(lv.cmap), p(ml_key), p(ml_val)});
          thrust::sort_by_key
            (pol, ml_key.begin(), ml_key.begin()+E, ml_val.begin());
          auto ends = thrust::reduce_by_key
            (pol, ml_key.begin(), ml_key.begin()+E, ml_val.begin(),
             ml_ukey.begin(), ml_uval.begin());
          const long long nu = ends.first - ml_ukey.begin();
          const long long nv = thrust::lower_bound
            (pol, ml_ukey.begin(), ml_ukey.begin()+nu, ~0ull) - ml_ukey.begin();
          lv.ind.resize(nv); lv.ew.resize(nv); lv.ptr.resize(mc+1);
          thrust::transform(pol, ml_ukey.begin(), ml_ukey.begin()+nv,
                            lv.ind.begin(), ml_low32_ftor<integer>());
          thrust::copy_n(pol, ml_uval.begin(), nv, lv.ew.begin());
          auto rows = thrust::make_transform_iterator
            (ml_ukey.begin(), ml_high32_ftor<integer>());
          thrust::lower_bound(pol, rows, rows + nv,
                              thrust::counting_iterator<integer>(0),
                              thrust::counting_iterator<integer>(mc+1),
                              lv.ptr.begin());
          L++;
          ml_sizes.push_back(mc);
          gptr = p(lv.ptr); gind = p(lv.ind); gew = p(lv.ew);
          gvw = p(lv.vw); gcomp = p(lv.comp);
          gkeep = nullptr;
          m = mc; mkept = mc; E = nv;
          // keep matching only in components that are still large
          thrust::fill_n(pol, ml_csz.begin(), nr_comps, 0);
          ml_count_kernel<<<(mc+255)/256, 256>>>(mc, gcomp, p(ml_csz));
          run_m(mc, ml_actc_ftor{gcomp, p(ml_csz), ml_coarse, p(ml_act)});
        }
        return L;
      };
      // refine from the coarsest level L-1 down, weighted FM on each
      // coarse level, the result goes to dest (parts of the graph
      // being ordered). The scratch arrays and counts of fm_buf are
      // reused, the fine counts are recomputed by the caller.
      auto ml_down = [&](int L, int nr_comps, int* dest) {
        int* fb = p(fm_buf);
        for (int l=L-1; l>=0; l--) {
          auto& lv = mlv[l];
          fm_ctx<integer> f{};
          f.ptr = p(lv.ptr); f.ind = p(lv.ind);
          f.comp = p(lv.comp); f.part = p(lv.part); f.vw = p(lv.vw);
          f.side = fb + n; f.gain = fb + 2*n; f.mv = fb + 3*n;
          f.cnt = fb + 4*n; f.bal = fm_bal;
          thrust::fill_n(pol, fm_buf.begin()+4*n,
                         3*nr_comps, 0);
          fm_count_kernel<<<(lv.m+255)/256, 256>>>
            (lv.m, f.part, f.comp, f.cnt, f.vw);
          thrust::fill_n(pol, fm_pflag.begin(), ml_fm, 0);
          for (int ps=0; ps<ml_fm; ) {
            const int last = std::min(ml_fm, ps + fm_batch);
            for (; ps<last; ps++) {
              f.moved = p(fm_pflag) + ps;
              run_m(lv.m, fm_propose_ftor<integer>{f});
              run_m(lv.m, fm_select_ftor<integer>{f});
              run_m(lv.m, fm_apply_ftor<integer>{f});
            }
            if (!fm_pflag[last-1]) break;
          }
          if (ml_neg > 0) {
            // hill climbing on this coarse level, as on the fine graph
            const int m = lv.m;
            int* hb = p(ml_hc);
            fm_ctx<integer> h = f;
            h.lock = hb + n; h.neg = ml_neg;
            auto recount = [&]() {
              thrust::fill_n(pol, fm_buf.begin()+4*n,
                             3*nr_comps, 0);
              fm_count_kernel<<<(m+255)/256, 256>>>
                (m, h.part, h.comp, h.cnt, h.vw);
            };
            auto upd = [&]() -> bool {
              thrust::fill_n(pol, ml_hc_flag.begin(), 1,
                             false);
              thrust::for_each_n
                (pol, thrust::counting_iterator<int>(0),
                 nr_comps, fm_best_comp_ftor
                 {h.cnt, hb + 2*n, hb + 3*n, hb + 4*n, p(ml_hc_flag)});
              run_m(m, fm_copy_best_ftor{h.part, h.comp, hb + 4*n, hb});
              return ml_hc_flag[0];
            };
            auto restore = [&]() {
              run_m(m, fm_restore_ftor{h.part, hb, hb + n});
            };
            recount();
            thrust::fill_n(pol, ml_hc.begin()+n, m, 0);
            thrust::fill_n(pol, ml_hc.begin()+2*n,
                           nr_comps, n+1);
            thrust::fill_n(pol, ml_hc.begin()+3*n,
                           nr_comps, -1);
            upd();
            thrust::fill_n(pol, fm_pflag.begin(), ml_fm, 0);
            int since = 0;
            bool round_imp = false;
            for (int ps=0; ps<ml_fm; ps++) {
              if (ps > 0) {
                bool imp = upd();
                since = imp ? 0 : since + 1;
                round_imp = round_imp || imp;
                if (since >= ml_climb) {
                  restore();
                  if (!round_imp) break;
                  since = 0; round_imp = false;
                  recount();
                }
              }
              h.moved = p(fm_pflag) + ps;
              run_m(m, fm_propose_ftor<integer>{h});
              run_m(m, fm_select_ftor<integer>{h});
              run_m(m, fm_apply_ftor<integer>{h});
              if (!fm_pflag[ps]) {
                restore();
                if (!round_imp) break;
                since = 0; round_imp = false;
                recount();
              }
            }
            recount();
            upd();
            restore();
          }
          int* fpart = l ? p(mlv[l-1].part) : dest;
          const int mf = l ? mlv[l-1].m : n;
          run_m(mf, ml_proj_ftor{p(lv.cmap), p(lv.part), fpart});
        }
      };
      // STRUMPACK_ND_ML=2: separator search on coarse level lv, for
      // all its components at once: pseudo-peripheral BFS with
      // ml_reps random restarts and the two-end structures of the
      // graph being ordered (TWOEND=3: also the off-axis pairs),
      // weighted layer counts. Result in lv.part.
      auto ml_search = [&](ml_level<integer>& lv, int nr_comps) {
        const int m = lv.m;
        integer *cp = p(lv.ptr), *ci = p(lv.ind);
        int *cc = p(lv.comp), *cvw = p(lv.vw), *cs = p(comp_start);
        bool* ce = p(cl_elim);
        auto bfs = [&](thrust::device_vector<int>& roots) {
          thrust::fill_n(pol, cl_label.begin(), m, n);
          bfs_run(m, cp, ci, ce, p(cl_label), cc, cs, p(cl_levels),
                  p(roots), nr_comps, true);
        };
        auto score = [&](thrust::device_vector<int>& ra, int* rb,
                         int mode) {
          thrust::fill_n(pol, cl_layer.begin(), n, 0);
          thrust::fill_n(pol, cl_sepsz.begin(), n, 0);
          cardinalities_ftor<integer> cf
            (cp, ci, ce, p(cl_label), cc, p(cl_levels), cs, p(cl_layer),
             p(cl_sepsz));
          cf.vw = cvw;
          cardinalities(m, cf);
          minimize_sep_ftor<integer> ms
            (cp, ci, ce, p(cl_label), cc, p(cl_levels), cs,
             p(cl_layer), p(cl_sepsz), p(ra), p(cl_root_opt),
             p(cl_seplev), p(cl_sepside), p(cl_ns), ml_bal, rb,
             rb ? p(cl_optr2) : nullptr, p(cl_optmode), mode);
          ms.per_comp = true; ms.nnodes = m;
          run_m(nr_comps, ms);
        };
        auto far_end = [&](thrust::device_vector<int>& r) {
          argmin_root(m, cc, nr_comps, peripheral_score<integer>
                      {cp, ci, ce, p(cl_label), cc, p(cl_levels), cs}, r);
        };
        auto combine = [&](thrust::device_vector<int>& d1,
                           thrust::device_vector<int>& r2,
                           thrust::device_vector<int>& lev1,
                           const int* mode) {
          run_m(m, twoend_combine_ftor
                {ce, cc, cs, p(d1), p(r2), p(lev1), mode, p(cl_label),
                 p(cl_levels)});
        };
        // initial roots (lowest coarse node per component), coarse
        // component offsets and nodes sorted by component (for the
        // random restarts)
        thrust::fill_n(pol, cl_root.begin(), nr_comps,
                       std::numeric_limits<int>::max());
        run_m(m, ml_minroot_ftor{cc, p(cl_root)});
        thrust::fill_n(pol, cl_cst.begin(), nr_comps+1, 0);
        ml_count_kernel<<<(m+255)/256, 256>>>(m, cc, p(cl_cst));
        thrust::exclusive_scan(pol, cl_cst.begin(), cl_cst.begin()+nr_comps+1,
                               cl_cst.begin());
        thrust::copy_n(pol, lv.comp.begin(), m, cl_key.begin());
        thrust::sequence(pol, cl_map.begin(), cl_map.begin()+m);
        thrust::stable_sort_by_key(pol, cl_key.begin(), cl_key.begin()+m,
                                   cl_map.begin());
        thrust::copy_n(pol, cl_root.begin(), nr_comps, cl_root_opt.begin());
        thrust::copy_n(pol, cl_root.begin(), nr_comps, cl_optr2.begin());
        thrust::fill_n(pol, cl_optmode.begin(), nr_comps, 0);
        thrust::fill_n(pol, cl_ns.begin(), nr_comps,
                       std::numeric_limits<float>::max());
        for (int rep=0; rep<ml_reps; rep++) {
          if (rep > 0)
            thrust::for_each_n
              (pol, thrust::counting_iterator<int>(0), nr_comps,
               ml_rand_root_ftor{unsigned(ml_rng()), p(cl_cst), p(cl_map),
                                 p(cl_root)});
          thrust::fill_n(pol, cl_old.begin(), nr_comps, 1);
          do {
            bfs(cl_root);
            score(cl_root, nullptr, 0);
            if (less_equal_n(pol, nr_comps, cl_levels, cl_old)) break;
            far_end(cl_root);
            std::swap(cl_old, cl_levels);
          } while (1);
          if (!two_end) continue;
          // pair (r1, r2)
          thrust::copy_n(pol, cl_label.begin(), m, cl_d1.begin());
          thrust::copy_n(pol, cl_levels.begin(), nr_comps, cl_lev1.begin());
          thrust::copy_n(pol, cl_root.begin(), nr_comps, cl_r2.begin());
          far_end(cl_r2);
          bfs(cl_r2);
          thrust::copy_n(pol, cl_label.begin(), m, cl_d2.begin());
          thrust::copy_n(pol, cl_levels.begin(), nr_comps, cl_lev2.begin());
          combine(cl_d1, cl_r2, cl_lev1, nullptr);
          score(cl_root, p(cl_r2), 2);
          if (te_mode != 3) continue;
          // off-axis node r3, pairs (r1, r3) and (r2, r3)
          argmin_root(m, cc, nr_comps, offaxis_score
                      {ce, cc, cs, p(cl_d1), p(cl_d2)}, cl_r3);
          bfs(cl_r3);
          thrust::copy_n(pol, cl_label.begin(), m, cl_d3.begin());
          combine(cl_d1, cl_r3, cl_lev1, nullptr);
          score(cl_root, p(cl_r3), 2);
          thrust::copy_n(pol, cl_d3.begin(), m, cl_label.begin());
          combine(cl_d2, cl_r3, cl_lev2, nullptr);
          score(cl_r2, p(cl_r3), 2);
        }
        // rebuild the chosen level structure, then the parts
        bfs(cl_root_opt);
        if (two_end) {
          thrust::copy_n(pol, cl_label.begin(), m, cl_d1.begin());
          thrust::copy_n(pol, cl_levels.begin(), nr_comps, cl_lev1.begin());
          bfs(cl_optr2);
          combine(cl_d1, cl_optr2, cl_lev1, p(cl_optmode));
        }
        run_m(m, ml_cpart_ftor<integer>
              {cp, ci, cc, p(cl_levels), p(cl_label), p(cl_seplev),
               p(cl_sepside), p(lv.part)});
      };
      if (stats) { STRUMPACK_ND_DEVICE_SYNC(); tph0 = clk::now(); }
      do {
        // connected components: label[i] = smallest node index in the
        // component of i (union-find, see uf_hook_ftor)
        thrust::sequence(pol, label.begin(), label.end());
        run(uf_hook_ftor<integer>{p(ptr), p(ind), p(eliminated), p(label)});
        run(uf_root_ftor{p(eliminated), p(label)});
        // component roots (smallest node of each component) in
        // increasing order, and comp[root of c] = c
        const int nr_comps = int
          (thrust::copy_if
           (pol, thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(n), comp_root.begin(),
            comp_root_pred{p(eliminated), p(label)}) - comp_root.begin());
        thrust::scatter
          (pol, thrust::counting_iterator<int>(0),
           thrust::counting_iterator<int>(nr_comps), comp_root.begin(),
           comp.begin());
        // comp_root_opt is only updated by minimize_sep_ftor for
        // components that still have live nodes; components eliminated
        // by minimum degree in this step would keep a stale root from
        // the previous step (with different component numbering),
        // which then corrupts the level sets of another component.
        thrust::copy_n(pol, comp_root.begin(), nr_comps, comp_root_opt.begin());
        // parent tree node of each component, all nodes of a component
        // share the same parent, so use the component's root node
        const int tn_off = tree_parent.size();
        thrust::gather(pol, comp_root.begin(), comp_root.begin()+nr_comps,
                       tparent.begin(), old_levels.begin());
        tree_parent.resize(tn_off + nr_comps);
        thrust::copy_n(old_levels.begin(), nr_comps,
                       tree_parent.begin() + tn_off);
        auto tag_tree = [&]() {
          run(tag_tree_ftor(nr_comps, tn_off, p(comp), p(eliminated),
                            p(tnode), p(tparent)));
        };
        // comp[i] = comp[label[i]] (in place: only roots are read,
        // and their value does not change)
        thrust::gather
          (pol, label.begin(), label.end(), comp.begin(), comp.begin());
        thrust::fill_n(pol, comp_start.begin(), nr_comps+1, 0);
        count_comps_kernel<<<(n+255)/256, 256>>>
          (n, p(eliminated), nr_comps, p(comp), p(comp_start));
        phase_end(PH_CC);

        // BFS from roots[c] in each component, label[roots[c]] =
        // comp_start[c], the number of layers goes to levels
        auto find_level_sets = [&](thrust::device_vector<int>& roots) {
          thrust::fill_n(pol, label.begin(), n, n);
          bfs_run(n, p(ptr), p(ind), p(eliminated), p(label), p(comp),
                  p(comp_start), p(levels), p(roots), nr_comps, false);
        };
        // new BFS root r[c]: min degree node in the last layer of the
        // current level structure (smallest index among ties)
        auto far_end = [&](thrust::device_vector<int>& r) {
          argmin_root(n, p(comp), nr_comps, peripheral_score<integer>
                      {p(ptr), p(ind), p(eliminated), p(label), p(comp),
                       p(levels), p(comp_start)}, r);
        };

        auto nr_elim_MD = (md_max == 0) ? 0 :
          thrust::transform_reduce
          (pol, comp_start.begin(), comp_start.begin()+nr_comps,
           larger_to_zero_ftor(md_max), 0, thrust::plus<int>());
        if (nr_elim_MD) {
          auto& perm_off = sep_level;
          thrust::transform_exclusive_scan
            (pol, comp_start.begin(), comp_start.begin()+nr_comps,
             perm_off.begin(), larger_to_zero_ftor(md_max),
             0, thrust::plus<int>());
          thrust::exclusive_scan
            (pol, comp_start.begin(), comp_start.begin()+nr_comps+1,
             comp_start.begin());

          // nodes of the small components grouped by component; the
          // kernel sorts them and computes its own BFS tie breaking
          // order (this used to be a pseudo-peripheral BFS over all
          // components plus a sort of all n nodes, which also moved
          // comp_root of the large components)
          thrust::fill_n(pol, pbest.begin(), nr_comps, 0);
          run(md_gather_ftor{p(eliminated), p(comp), p(comp_start),
                             md_max, p(pbest), p(degree)});
          auto nel = nr_elim;
          auto t0 = clk::now();
          minimum_degree<MD_SWITCH,integer><<<nr_comps,MD_SWITCH>>>
            (p(ptr), p(ind), p(eliminated), p(label), p(degree),
             p(comp_start), p(perm)+nel, p(perm_off), md_max,
             md_fill_mode);
          if (stats) { STRUMPACK_ND_DEVICE_SYNC(); t_md += secs(t0); }
          if (nel + nr_elim_MD == n) {
            phase_end(PH_MD);
            tag_tree();
            if (stats)
              std::cout << "# ND step " << std::setw(2) << nd_step
                        << ": comps=" << nr_comps << " elim_before=" << nel
                        << " MD=" << nr_elim_MD << " (done)" << std::endl;
            break;
          }
          nr_elim = nel + nr_elim_MD;
        } else
          thrust::exclusive_scan
            (pol, comp_start.begin(), comp_start.begin()+nr_comps+1,
             comp_start.begin());

        phase_end(PH_MD);
        const bool cand_on = nd_step < cand_steps;
        if (cand_on) cand_load(nr_comps);
        if (two_end) {
          thrust::copy_n(pol, comp_root.begin(), nr_comps, te_opt_root2.begin());
          thrust::fill_n(pol, te_opt_mode.begin(), nr_comps, 0);
        }
        thrust::fill_n(pol, min_norm_sep.begin(), nr_comps,
                       std::numeric_limits<float>::max());
        if (nreps > 1) {
          // nodes sorted by component, and by index within a
          // component, to pick the random restart roots
          thrust::copy(pol, comp.begin(), comp.end(),
                       ekey.begin());
          thrust::sequence(pol, elist.begin(), elist.end());
          thrust::stable_sort_by_key(pol, ekey.begin(), ekey.end(), elist.begin());
        }
        for (int rep=0; rep<nreps; rep++) {
          // find level sets with different random initial nodes
          if (rep > 0)
            thrust::for_each_n
              (pol, thrust::counting_iterator<int>(0),
               nr_comps, random_roots_ftor(rng(), p(comp_root),
                                           p(comp_start), p(elist),
                                           root_mix));
          thrust::fill_n(pol, old_levels.begin(), nr_comps, 1);
          int pit = 0;
          do {
            find_level_sets(comp_root);
            thrust::fill_n
              (pol, thrust::make_zip_iterator
               (thrust::make_tuple(degree.begin(), sep_size.begin())),
               n, thrust::make_tuple(0, 0));
            cardinalities(n, cardinalities_ftor<integer>
                (p(ptr), p(ind), p(eliminated), p(label), p(comp),
                 p(levels), p(comp_start), p(degree), p(sep_size)));
            {
              minimize_sep_ftor<integer> ms
                (p(ptr), p(ind), p(eliminated), p(label), p(comp), p(levels),
                 p(comp_start), p(degree), p(sep_size),
                 p(comp_root), p(comp_root_opt), p(sep_level), p(sep_side),
                 p(min_norm_sep), sep_bal, nullptr, nullptr,
                 two_end ? p(te_opt_mode) : nullptr, 0);
              ms.per_comp = true; ms.nnodes = n;
              run_m(nr_comps, ms);
            }
            if (cand_on) cand_print(rep, pit++);
            if (less_equal_n(pol, nr_comps, levels, old_levels))
              break;
            far_end(comp_root);
            std::swap(old_levels, levels);
            // repeat level set BFS with new start nodes, so that the
            // start nodes converge to peripheral nodes of the
            // different components
          } while (1);
          if (two_end) {
            // score the current labels as the two-end structure of
            // (ra, rb); ra is recorded in comp_root_opt and rb in
            // te_opt_root2 if it wins, so that it can be rebuilt
            auto score_te = [&](thrust::device_vector<int>& ra,
                                thrust::device_vector<int>& rb, int tag) {
              thrust::fill_n
                (pol, thrust::make_zip_iterator
                 (thrust::make_tuple(degree.begin(), sep_size.begin())),
                 n, thrust::make_tuple(0, 0));
              cardinalities(n, cardinalities_ftor<integer>
                  (p(ptr), p(ind), p(eliminated), p(label), p(comp),
                   p(levels), p(comp_start), p(degree), p(sep_size)));
              minimize_sep_ftor<integer> ms
                (p(ptr), p(ind), p(eliminated), p(label), p(comp),
                 p(levels), p(comp_start), p(degree), p(sep_size), p(ra), p(comp_root_opt), p(sep_level),
                 p(sep_side), p(min_norm_sep), sep_bal, p(rb),
                 p(te_opt_root2), p(te_opt_mode), 2);
              ms.per_comp = true; ms.nnodes = n;
              run_m(nr_comps, ms);
              if (cand_on) cand_print(rep, tag);
            };
            // the labels are the BFS from r1 = comp_root; r2 = a
            // minimum degree node in its last layer. The labels are
            // kept in te_dist1 by a swap, label is completely
            // overwritten by the next BFS
            thrust::copy_n(pol, levels.begin(), nr_comps, te_lev1.begin());
            thrust::copy_n(pol, comp_root.begin(), nr_comps, te_root2.begin());
            far_end(te_root2);
            std::swap(label, te_dist1);
            find_level_sets(te_root2);
            if (te_mode == 3) {
              thrust::copy(pol, label.begin(), label.end(), te_dist2.begin());
              thrust::copy_n(pol, levels.begin(), nr_comps, te_lev2.begin());
            }
            run(twoend_combine_ftor
                {p(eliminated), p(comp), p(comp_start), p(te_dist1),
                 p(te_root2), p(te_lev1), nullptr, p(label), p(levels)});
            score_te(comp_root, te_root2, 100);
            if (te_mode == 3) {
              thrust::copy_n(pol, te_root2.begin(), nr_comps, te_root3.begin());
              // r3: node maximizing min(d1, d2)
              argmin_root(n, p(comp), nr_comps, offaxis_score
                          {p(eliminated), p(comp), p(comp_start),
                           p(te_dist1), p(te_dist2)}, te_root3);
              if (cand_on)
                std::cout << "# TE3 step=" << nd_step << " rep=" << rep
                          << " r1=" << int(comp_root[cand_c])
                          << " r2=" << int(te_root2[cand_c])
                          << " r3=" << int(te_root3[cand_c])
                          << " lev1=" << int(te_lev1[cand_c])
                          << " lev2=" << int(te_lev2[cand_c]) << std::endl;
              find_level_sets(te_root3);
              thrust::copy(pol, label.begin(), label.end(), te_dist3.begin());
              // pair (r1, r3)
              run(twoend_combine_ftor
                  {p(eliminated), p(comp), p(comp_start), p(te_dist1),
                   p(te_root3), p(te_lev1), nullptr, p(label),
                   p(levels)});
              score_te(comp_root, te_root3, 101);
              // pair (r2, r3)
              thrust::copy(pol, te_dist3.begin(), te_dist3.end(),
                           label.begin());
              run(twoend_combine_ftor
                  {p(eliminated), p(comp), p(comp_start), p(te_dist2),
                   p(te_root3), p(te_lev2), nullptr, p(label),
                   p(levels)});
              score_te(te_root2, te_root3, 102);
            }
          }
        }
        phase_end(PH_SEARCH);
        if (nr_elim == n) { tag_tree(); break; }
        find_level_sets(comp_root_opt);
        if (two_end) {
          // rebuild the chosen level structure: two-end for
          // components with te_opt_mode == 2, single root otherwise
          // (label is completely overwritten by the next BFS)
          std::swap(label, te_dist1);
          thrust::copy_n(pol, levels.begin(), nr_comps, te_lev1.begin());
          find_level_sets(te_opt_root2);
          run(twoend_combine_ftor
              {p(eliminated), p(comp), p(comp_start), p(te_dist1),
               p(te_opt_root2), p(te_lev1), p(te_opt_mode), p(label),
               p(levels)});
        }
        phase_end(PH_REBUILD);
        run(mark_sep_ftor<integer>
            (p(ptr), p(ind), p(eliminated), p(label), p(comp),
             p(levels), p(sep_level), p(sep_side), p(sepflag)));
        if (cand_on) cand_cut("bfs", sepflag, false);
        if (vc_thresh > 0) {
          auto t0 = clk::now();
          {
            thrust::fill_n(pol, vc_cnt.begin(),
                           2*nr_comps, 0);
            vc_ctx<integer> x{p(ptr), p(ind), p(eliminated), p(comp),
                p(levels), p(sep_level), p(sep_side), p(label),
                p(sepflag), p(vc_mate), p(vc_best), p(vc_z), p(vc_cnt),
                p(vc_cnt) + nr_comps, p(running), n, vc_thresh};
            run(vc_init_ftor<integer>{x});
            int rounds = 0;
            do { // greedy matching
              clear_running();
              for (int b=0; b<bfs_batch; b++) {
                run(vc_propose_ftor<integer>{x});
                run(vc_accept_ftor<integer>{x});
              }
              rounds += bfs_batch;
            } while (running[0] && rounds < vc_max_rounds);
            if (vc_max_phases > 0) {
              x.uroot = p(vc_aux);
              x.ulev = p(vc_aux) + n;
              x.rend = p(vc_aux) + 2*n;
              x.prop = p(vc_aux) + 3*n;
              for (int ph=0; ph<vc_max_phases; ph++) {
                run(vc_aug_init_ftor<integer>{x});
                int L = 0;
                do { // alternating BFS forest
                  clear_running();
                  for (int b=0; b<bfs_batch; b++, L++) {
                    run(vc_aug_expand_ftor<integer>{x, L});
                    run(vc_aug_claim_ftor<integer>{x, L});
                  }
                } while (running[0]);
                clear_running();
                run(vc_aug_apply_ftor<integer>{x});
                if (!running[0]) break; // matching is maximum
              }
            }
            // Z by a frontier queue BFS (see vc_z_layer_kernel)
            int *zqs = p(bfs_qbuf), *zqb = zqs + qlen, *zq = zqb + qlen;
            const int G = std::max(1, std::min((n+255)/256, 1024));
            thrust::fill_n(pol, bfs_qbuf.begin(), 2, 0);
            thrust::fill_n(pol, bfs_qbuf.begin()+qlen, 1, 0);
            run(vc_zinit_ftor<integer>{x, zq, zqs});
            int zit = 0;
            do { // alternating BFS
              for (int b=0; b<bfs_batch; b++, zit++)
                vc_z_layer_kernel<integer><<<G, 256>>>
                  (x, zit, zq, zqs, zqb);
            } while (int(bfs_qbuf[zit]) > 0);
            run(vc_count_new_ftor<integer>{x});
            run(vc_apply_ftor<integer>{x});
            if (cand_on) cand_cut("vc ", sepflag, false);
            if (stats) {
              thrust::host_vector<int> hc(2*nr_comps);
              thrust::copy_n(vc_cnt.begin(), 2*nr_comps, hc.begin());
              for (int c=0; c<nr_comps; c++)
                if (hc[c] >= vc_thresh && hc[nr_comps+c] < hc[c])
                  vc_saved += hc[c] - hc[nr_comps+c];
            }
          }
          phase_end(PH_VC);
          if (fm_passes > 0) {
            int* fb = p(fm_buf);
            fm_ctx<integer> f{p(ptr), p(ind), p(eliminated), p(comp),
                p(levels), p(sep_level), p(sep_side), p(label),
                p(sepflag), fb, fb + n, fb + 2*n, fb + 3*n, fb + 4*n,
                p(fm_pflag), fm_bal};
            run(fm_init_ftor<integer>{f});
            long long s0 = stats ?
              thrust::count(pol, fm_buf.begin(), fm_buf.begin()+n, 2) : 0;
            const int npass = fm_passes;
            auto fm_counts = [&]() {
              thrust::fill_n(pol, fm_buf.begin()+4*n,
                             3*nr_comps, 0);
              fm_count_kernel<<<(n+255)/256, 256>>>
                (n, fb, p(comp), fb + 4*n);
            };
            // one pass; sets fm_pflag[ps] = 1 if anything moved,
            // and updates the counts incrementally
            auto fm_pass = [&](int ps) {
              f.moved = p(fm_pflag) + ps;
              run(fm_propose_ftor<integer>{f});
              run(fm_select_ftor<integer>{f});
              run(fm_apply_ftor<integer>{f});
            };
            thrust::fill_n(pol, fm_pflag.begin(), npass, 0);
            fm_counts();
            // plain FM: fm_batch passes per host check; a pass
            // that moves nothing leaves the state unchanged, so
            // the extra passes after convergence are no-ops
            auto plain_fm = [&]() {
              thrust::fill_n(pol, fm_pflag.begin(),
                             npass, 0);
              for (int ps=0; ps<npass; ) {
                const int last = std::min(npass, ps + fm_batch);
                for (; ps<last; ps++) fm_pass(ps);
                if (!fm_pflag[last-1]) break;
              }
              if (stats)
                fm_moves += thrust::reduce
                  (pol, fm_pflag.begin(), fm_pflag.begin()+npass);
            };
            plain_fm();
            if (ml_on) {
              // ML=2: separator on the coarsest graph, projected and
              // refined back, FM on this graph, then per component
              // the better of the multilevel and BFS separators
              auto tm = clk::now();
              auto tph = clk::now();
              auto phase = [&](double& acc) {
                if (!stats) return;
                STRUMPACK_ND_DEVICE_SYNC();
                acc += secs(tph); tph = clk::now();
              };
              const int L = ml_build(nr_comps);
              phase(t_ml_build);
              if (L > 0) {
                ml_search(mlv[L-1], nr_comps);
                phase(t_ml_search);
                thrust::fill(pol, ml_fpart.begin(),
                             ml_fpart.end(), -1);
                ml_down(L, nr_comps, p(ml_fpart));
                if ((long long)ml_cnt.size() < 7LL*nr_comps)
                  ml_cnt.resize(7*nr_comps);
                int *c1 = p(ml_cnt), *c2 = c1 + 3*nr_comps,
                  *pk = c1 + 6*nr_comps;
                fm_ctx<integer> g = f;
                g.part = p(ml_fpart); g.cnt = c2; g.bal = fm_bal;
                thrust::fill_n(pol, ml_cnt.begin(),
                               6*nr_comps, 0);
                fm_count_kernel<<<(n+255)/256, 256>>>
                  (n, p(ml_fpart), p(comp), c2);
                thrust::fill_n(pol, fm_pflag.begin(),
                               npass, 0);
                for (int ps=0; ps<npass; ) {
                  const int last = std::min(npass, ps + fm_batch);
                  for (; ps<last; ps++) {
                    g.moved = p(fm_pflag) + ps;
                    run(fm_propose_ftor<integer>{g});
                    run(fm_select_ftor<integer>{g});
                    run(fm_apply_ftor<integer>{g});
                  }
                  if (!fm_pflag[last-1]) break;
                }
                fm_count_kernel<<<(n+255)/256, 256>>>(n, fb, p(comp), c1);
                thrust::for_each_n
                  (pol, thrust::counting_iterator<int>(0),
                   nr_comps, ml_pick_ftor{c1, c2, pk});
                if (stats) {
                  thrust::host_vector<int> h(7*nr_comps);
                  thrust::copy_n(ml_cnt.begin(), 7*nr_comps, h.begin());
                  long long s1 = 0, s2 = 0, sf = 0;
                  int np = 0, npart = 0, cm = -1;
                  for (int c=0; c<nr_comps; c++) {
                    const int* a = &h[3*c];
                    const int* b = &h[3*(nr_comps+c)];
                    if (b[0] + b[1] + b[2] == 0) continue;
                    npart++; s1 += a[2]; s2 += b[2];
                    sf += h[6*nr_comps+c] ? b[2] : a[2];
                    np += h[6*nr_comps+c];
                    if (cm < 0 || a[0]+a[1]+a[2] >
                        h[3*cm]+h[3*cm+1]+h[3*cm+2]) cm = c;
                  }
                  ml_saved += s1 - sf;
                  std::cout << "# ND ML2 step " << std::setw(2) << nd_step
                            << ": levels=" << L << " nodes=";
                  for (std::size_t q=0; q<ml_sizes.size(); q++)
                    std::cout << (q ? "/" : "") << ml_sizes[q];
                  std::cout << " comps=" << npart << " picked=" << np
                            << " S: bfs=" << s1 << " ml=" << s2
                            << " final=" << sf;
                  if (cm >= 0) {
                    const int* a = &h[3*cm];
                    const int* b = &h[3*(nr_comps+cm)];
                    std::cout << " | largest A/B/S bfs=" << a[0] << "/"
                              << a[1] << "/" << a[2] << " ml=" << b[0]
                              << "/" << b[1] << "/" << b[2];
                  }
                  std::cout << std::endl;
                }
                run(ml_apply_pick_ftor{p(comp), pk, p(ml_fpart), fb});
                phase(t_ml_down);
              }
              if (stats) { STRUMPACK_ND_DEVICE_SYNC(); t_ml += secs(tm); }
            }
            run(fm_final_ftor<integer>{f});
            if (cand_on) cand_cut("fm ", fm_buf, true);
            if (stats)
              fm_saved += s0 -
                thrust::count(pol, fm_buf.begin(), fm_buf.begin()+n, 2);
          }
          phase_end(PH_FM);
          if (stats) STRUMPACK_ND_DEVICE_SYNC();
          t_vc += secs(t0);
        }
        if (stats) {
          // nodes eliminated per component, for the statistics
          auto& sep_count = comp_root;
          thrust::fill_n(pol, sep_count.begin(), nr_comps, 0);
          run(count_flag_ftor(p(sepflag), p(comp), p(sep_count)));
          print_nd_stats(nd_step, n, nr_comps, nr_elim, nr_elim_MD,
                         sep_count, comp_start, levels, sep_level, sep_side,
                         label, comp, eliminated);
        }
        nd_step++;
        // eliminate the flagged nodes, grouped by component and in
        // increasing order within a component
        const int nr_elim_sep = int
          (thrust::copy_if
           (pol, thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(n), sepflag.begin(),
            elist.begin(), nonzero_pred{}) - elist.begin());
        thrust::gather(pol, elist.begin(),
                       elist.begin()+nr_elim_sep, comp.begin(), ekey.begin());
        thrust::stable_sort_by_key(pol, ekey.begin(), ekey.begin()+nr_elim_sep,
                                   elist.begin());
        thrust::for_each_n
          (pol, thrust::counting_iterator<int>(0),
           nr_elim_sep, elim_list_ftor<integer>
           {p(elist), p(eliminated), p(perm)+nr_elim});
        nr_elim += nr_elim_sep;
        tag_tree();
        phase_end(PH_ELIM);
      } while (nr_elim < n);
      if (stats) {
        std::cout << "# ND phases (s):";
        for (int ph=0; ph<PH_N; ph++)
          std::cout << " " << ph_name[ph] << "=" << t_ph[ph];
        std::cout << std::endl;
      }
      if (stats && vc_thresh > 0)
        std::cout << "# ND vertex cover refinement: " << vc_saved
                  << " separator nodes saved, " << t_vc << " s"
                  << " (incl. FM)" << std::endl;
      if (stats && fm_passes > 0 && vc_thresh > 0)
        std::cout << "# ND FM refinement: " << fm_saved
                  << " separator nodes saved, " << fm_moves
                  << " passes with moves" << std::endl;
      if (stats)
        std::cout << "# ND minimum degree kernels: " << t_md << " s"
                  << std::endl;
      if (stats && ml_on)
        std::cout << "# ND multilevel refinement: " << ml_saved
                  << " separator nodes saved, " << t_ml << " s"
                  << " (coarsen " << t_ml_build << " s, coarse search "
                  << t_ml_search << " s, uncoarsen+FM+pick " << t_ml_down
                  << " s)" << std::endl;
      thrust::reverse(perm.begin(), perm.end());
    }

    /**
     * Build the separator tree directly from the nested dissection
     * structure, instead of reconstructing it from the elimination
     * tree of the final permutation.
     *
     * Every ND tree node (separator, minimum degree leaf, or small
     * component) becomes one front. A SeparatorTree node must have 0
     * or 2 children:
     *  - a node with more than 2 children (e.g. a separator whose
     *    removal leaves more than 2 components, or multiple connected
     *    components at the top) is binarized with empty separators,
     *    always combining the two shallowest subtrees first, to keep
     *    the depth minimal,
     *  - a node with a single child is merged into the child's front
     *    (its columns are appended to the child's separator).
     *
     * On input perm is the ND ordering (new -> old); within each tree
     * node that order is preserved (e.g. the minimum degree order in
     * the leaves). On output perm (new -> old) and iperm (old -> new)
     * are the postorder of the tree.
     */
    template<typename integer> SeparatorTree<integer>
    nd_tree_from_structure(integer n, const integer* ptr, const integer* ind,
                           const std::vector<int>& tree_parent,
                           const std::vector<int>& tnode,
                           std::vector<integer>& perm,
                           std::vector<integer>& iperm) {
      const int T = tree_parent.size();
      std::vector<std::vector<integer>> cols(T);
      for (integer k=0; k<n; k++)
        cols[tnode[perm[k]]].push_back(perm[k]);
      std::vector<std::vector<int>> kids(T);
      std::vector<int> roots;
      for (int t=0; t<T; t++) {
        if (tree_parent[t] < 0) roots.push_back(t);
        else kids[tree_parent[t]].push_back(t);
      }
      // binary tree, ids >= T are empty (virtual) separators, or
      // fronts created when expanding a leaf (see below)
      std::vector<int> bl(T, -1), br(T, -1), height(T, 1);

      // STRUMPACK_ND_LEAF_AMALG=k (k > 0, default 32, 0 disables):
      // instead of treating each
      // leaf of the ND tree (minimum degree leaf, small component) as
      // a single dense front, expand it into the elimination tree of
      // its subgraph (in the leaf's elimination order). Subtrees with
      // at most k nodes become a single front, chains of single-child
      // nodes are merged, and nodes with several children are
      // binarized with empty separators. This keeps the sparsity
      // inside the leaves, only adding depth in the bottom subtrees.
      static const int leaf_amalg =
        nd_env_int("STRUMPACK_ND_LEAF_AMALG", 32);
      if (leaf_amalg > 0) {
        std::vector<int> loc(n, -1);
        auto newnode = [&]() {
          int id = bl.size();
          bl.push_back(-1); br.push_back(-1); height.push_back(1);
          cols.emplace_back();
          return id;
        };
        for (int t=0; t<T; t++) {
          if (!kids[t].empty()) continue;
          const int m = cols[t].size();
          if (m <= leaf_amalg || m > 4096) continue;
          const std::vector<integer> c = cols[t];
          // elimination tree of the leaf subgraph (Liu's algorithm,
          // with path compression), local indices in elimination order
          for (int j=0; j<m; j++) loc[c[j]] = j;
          std::vector<int> par(m, -1), anc(m, -1);
          for (int j=0; j<m; j++)
            for (auto kk=ptr[c[j]]; kk<ptr[c[j]+1]; kk++) {
              int r = loc[ind[kk]];
              if (r < 0 || r >= j) continue;
              while (anc[r] != -1 && anc[r] != j) {
                int nx = anc[r]; anc[r] = j; r = nx;
              }
              if (anc[r] == -1) { anc[r] = j; par[r] = j; }
            }
          for (int j=0; j<m; j++) loc[c[j]] = -1;
          std::vector<std::vector<int>> lk(m);
          std::vector<int> lroots, subsize(m, 1);
          for (int j=0; j<m; j++) {
            if (par[j] < 0) lroots.push_back(j);
            else {
              lk[par[j]].push_back(j);
              subsize[par[j]] += subsize[j];  // par[j] > j
            }
          }
          std::function<void(int,std::vector<integer>&)> collect =
            [&](int v, std::vector<integer>& out) {
            for (auto ch : lk[v]) collect(ch, out);
            out.push_back(c[v]);
          };
          std::function<void(int,int)> build;
          std::function<int(const int*,int)> bin =
            [&](const int* l, int cnt) -> int {
            if (cnt == 1) { int a = newnode(); build(l[0], a); return a; }
            int h = cnt / 2;
            int L = bin(l, h), R = bin(l+h, cnt-h);
            int e = newnode();
            bl[e] = L; br[e] = R;
            height[e] = std::max(height[L], height[R]) + 1;
            return e;
          };
          // build the binary (sub)tree for local etree node v into
          // global node tgt
          build = [&](int v, int tgt) {
            std::vector<integer> tmp;
            if (subsize[v] <= leaf_amalg) {
              collect(v, tmp);
              cols[tgt] = std::move(tmp);
              bl[tgt] = br[tgt] = -1;
              height[tgt] = 1;
              return;
            }
            std::vector<int> chain{v};
            int u = v;
            while (lk[u].size() == 1 && subsize[lk[u][0]] > leaf_amalg) {
              u = lk[u][0];
              chain.push_back(u);
            }
            for (int i=chain.size()-1; i>=0; i--) tmp.push_back(c[chain[i]]);
            cols[tgt] = std::move(tmp);
            const auto& ch = lk[u];
            if (ch.empty()) {
              bl[tgt] = br[tgt] = -1; height[tgt] = 1;
            } else if (ch.size() == 1) {
              // small child, merged into this front by emit
              int a = newnode(); build(ch[0], a);
              bl[tgt] = a; br[tgt] = -1; height[tgt] = height[a];
            } else {
              int h = ch.size() / 2;
              int L = bin(ch.data(), h), R = bin(ch.data()+h, ch.size()-h);
              bl[tgt] = L; br[tgt] = R;
              height[tgt] = std::max(height[L], height[R]) + 1;
            }
          };
          if (lroots.size() == 1) build(lroots[0], t);
          else {
            cols[t].clear();
            int h = lroots.size() / 2;
            int L = bin(lroots.data(), h);
            int R = bin(lroots.data()+h, lroots.size()-h);
            bl[t] = L; br[t] = R;
            height[t] = std::max(height[L], height[R]) + 1;
          }
        }
      }
      // combine a list of subtrees (by height) until target remain
      auto combine = [&](std::vector<int> list, std::size_t target) {
        using hp = std::pair<int,int>;
        std::priority_queue<hp, std::vector<hp>, std::greater<hp>> q;
        for (auto c : list) q.push({height[c], c});
        while (q.size() > target) {
          auto a = q.top(); q.pop();
          auto b = q.top(); q.pop();
          int v = bl.size();
          bl.push_back(a.second);
          br.push_back(b.second);
          height.push_back(std::max(a.first, b.first) + 1);
          cols.emplace_back();
          q.push({height[v], v});
        }
        std::vector<int> r;
        while (!q.empty()) { r.push_back(q.top().second); q.pop(); }
        return r;
      };
      // children have larger ids than their parent
      for (int t=T-1; t>=0; t--) {
        auto& k = kids[t];
        if (k.empty()) continue;
        if (k.size() == 1) {
          bl[t] = k[0];
          height[t] = height[k[0]];
          continue;
        }
        auto two = combine(k, 2);
        bl[t] = two[0];
        br[t] = two[1];
        height[t] = std::max(height[two[0]], height[two[1]]) + 1;
      }
      int root = combine(roots, 1)[0];

      std::vector<Separator<integer>> seps;
      std::vector<integer> order;
      order.reserve(n);
      auto append = [&](int x) {
        for (auto c : cols[x]) order.push_back(c);
      };
      std::function<int(int)> emit = [&](int x) -> int {
        if (bl[x] == -1 && br[x] == -1) {
          append(x);
          seps.emplace_back(order.size(), -1, -1, -1);
          return seps.size() - 1;
        }
        if (br[x] == -1) {
          // single child, merge into the child's front
          int id = emit(bl[x]);
          append(x);
          seps[id].sep_end = order.size();
          return id;
        }
        int a = emit(bl[x]), b = emit(br[x]);
        append(x);
        int id = seps.size();
        seps.emplace_back(order.size(), -1, a, b);
        seps[a].pa = id;
        seps[b].pa = id;
        return id;
      };
      emit(root);
      assert(integer(order.size()) == n);
      std::copy(order.begin(), order.end(), perm.begin());
      for (integer i=0; i<n; i++)
        iperm[perm[i]] = i;
      return SeparatorTree<integer>(seps);
    }

    /**
     * Graph compression (STRUMPACK_ND_COMPRESS, default 1): unknowns
     * with identical closed adjacency (the unknown itself plus its
     * neighbors), e.g. the 3 displacement unknowns of a mesh node in
     * 3D elasticity, are merged into one node of a smaller graph. The
     * nested dissection is computed on the merged graph, then
     * expanded back, keeping the unknowns of a group consecutive.
     */
    __host__ __device__ inline unsigned long long
    nd_mix64(unsigned long long x) { // splitmix64 finalizer
      x += 0x9e3779b97f4a7c15ull;
      x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
      x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
      return x ^ (x >> 31);
    }
    template<typename integer> struct nd_row_unsorted_ftor {
      const integer *ptr, *ind;
      __device__ bool operator()(int i) const {
        for (auto k=ptr[i]+1; k<ptr[i+1]; k++)
          if (ind[k-1] > ind[k]) return true;
        return false;
      }
    };
    template<typename integer> struct nd_row_keys_ftor {
      const integer *ptr, *ind;
      unsigned long long* key;
      __device__ void operator()(int i) const {
        for (auto k=ptr[i]; k<ptr[i+1]; k++)
          key[k] = (static_cast<unsigned long long>(i) << 32) |
            static_cast<unsigned long long>(ind[k]);
      }
    };
    template<typename integer> struct nd_low32_ftor {
      __host__ __device__ integer operator()(unsigned long long k) const
      { return integer(k & 0xffffffffull); }
    };
    template<typename integer> struct nd_high32_ftor {
      __host__ __device__ integer operator()(unsigned long long k) const
      { return integer(k >> 32); }
    };
    // order independent hash of the closed adjacency of each node
    template<typename integer> struct nd_hash_ftor {
      const integer *ptr, *ind;
      unsigned long long* h;
      __device__ void operator()(int i) const {
        auto s = nd_mix64(static_cast<unsigned long long>(i));
        for (auto k=ptr[i]; k<ptr[i+1]; k++)
          s += nd_mix64(static_cast<unsigned long long>(ind[k]));
        h[i] = s;
      }
    };
    // exact comparison of the closed (sorted) adjacency of a and b,
    // each node's own index is merged in on the fly
    template<typename integer> __device__ bool
    nd_closed_equal(const integer* ptr, const integer* ind, int a, int b) {
      auto ka = ptr[a], ea = ptr[a+1], kb = ptr[b], eb = ptr[b+1];
      if (ea - ka != eb - kb) return false;
      bool sa = false, sb = false;
      while (true) {
        long long va = 0, vb = 0;
        bool ha = true, hb = true;
        if (!sa && (ka == ea || ind[ka] > a)) { va = a; sa = true; }
        else if (ka < ea) va = ind[ka++];
        else ha = false;
        if (!sb && (kb == eb || ind[kb] > b)) { vb = b; sb = true; }
        else if (kb < eb) vb = ind[kb++];
        else hb = false;
        if (ha != hb) return false;
        if (!ha) return true;
        if (va != vb) return false;
      }
    }
    template<typename integer> struct nd_group_head_ftor {
      const integer *ptr, *ind;
      const unsigned long long* h;
      const int* ids;
      int* head;
      __device__ void operator()(int k) const {
        if (k == 0) { head[k] = 1; return; }
        int a = ids[k-1], b = ids[k];
        head[k] = (h[a] != h[b] || !nd_closed_equal(ptr, ind, a, b)) ? 1 : 0;
      }
    };
    struct nd_group_rep_ftor {
      const int *head, *gs, *ids;
      int *rep, *isrep;
      __device__ void operator()(int k) const {
        if (head[k]) { rep[gs[k]-1] = ids[k]; isrep[ids[k]] = 1; }
      }
    };
    struct nd_group_gid_ftor {
      const int *gs, *ids, *rep, *newid;
      int* gid;
      __device__ void operator()(int k) const {
        gid[ids[k]] = newid[rep[gs[k]-1]];
      }
    };
    template<typename integer> struct nd_rep_deg_ftor {
      const integer* ptr;
      const int *goff, *gmem;
      __host__ __device__ integer operator()(int g) const {
        int r = gmem[goff[g]];
        return ptr[r+1] - ptr[r];
      }
    };
    template<typename integer> struct nd_merged_keys_ftor {
      const integer *ptr, *ind, *toff;
      const int *goff, *gmem, *gid;
      unsigned long long* key;
      __device__ void operator()(int g) const {
        int r = gmem[goff[g]];
        auto o = toff[g] - ptr[r];
        for (auto k=ptr[r]; k<ptr[r+1]; k++) {
          int gj = gid[ind[k]];
          key[o + k] = (gj == g) ? ~0ull :
            ((static_cast<unsigned long long>(g) << 32) |
             static_cast<unsigned long long>(gj));
        }
      }
    };
    template<typename integer> struct nd_expand_ftor {
      const integer* cperm;
      const int *off, *goff, *gmem;
      integer* perm;
      __device__ void operator()(int k) const {
        int g = cperm[k], o = off[k] - goff[g];
        for (int m=goff[g]; m<goff[g+1]; m++)
          perm[o + m] = gmem[m];
      }
    };

    template<typename integer> struct nd_compressed {
      int ng = 0;                    // number of groups
      thrust::device_vector<int>
        gid,                         // group of each unknown
        vw,                          // group size (node weight)
        goff, gmem;                  // members of each group (CSR)
      thrust::device_vector<integer> cptr, cind; // merged graph
    };

    /**
     * Detect the groups and build the merged graph. Returns false
     * (and leaves C empty) if fewer than min_frac * n unknowns would
     * be removed. Groups are numbered in the order of their smallest
     * member, and the members of a group are in increasing order.
     */
    template<typename integer> bool
    nd_compress_graph(int n, integer nnz,
                      const thrust::device_vector<integer>& dptr,
                      const thrust::device_vector<integer>& dind,
                      double min_frac, nd_compressed<integer>& C,
                      bool stats) {
      using ull = unsigned long long;
      auto p = [](auto& v) { return thrust::raw_pointer_cast(v.data()); };
      auto cnt0 = thrust::counting_iterator<int>(0);
      const integer* ptr = p(dptr);
      const integer* ind = p(dind);
      // the comparison needs sorted rows, sort a copy if needed
      thrust::device_vector<integer> sind;
      if (thrust::count_if(thrust::device, cnt0, cnt0+n,
                           nd_row_unsorted_ftor<integer>{ptr, ind})) {
        thrust::device_vector<ull> key(nnz);
        thrust::for_each_n(thrust::device, cnt0, n,
                           nd_row_keys_ftor<integer>{ptr, ind, p(key)});
        thrust::sort(key.begin(), key.end());
        sind.resize(nnz);
        thrust::transform(key.begin(), key.end(), sind.begin(),
                          nd_low32_ftor<integer>());
        ind = p(sind);
      }
      thrust::device_vector<ull> h(n);
      thrust::for_each_n(thrust::device, cnt0, n,
                         nd_hash_ftor<integer>{ptr, ind, p(h)});
      thrust::device_vector<int> ids(n), head(n), gs(n);
      thrust::sequence(ids.begin(), ids.end());
      {
        thrust::device_vector<ull> hs(h);
        thrust::stable_sort_by_key(hs.begin(), hs.end(), ids.begin());
      }
      thrust::for_each_n
        (thrust::device, cnt0, n, nd_group_head_ftor<integer>
         {ptr, ind, p(h), p(ids), p(head)});
      thrust::inclusive_scan(head.begin(), head.end(), gs.begin());
      const int ng = gs[n-1];
      if (stats)
        std::cout << "# ND compression: n=" << n << " groups=" << ng
                  << " ratio=" << double(n) / ng << std::endl;
      if (double(n - ng) < min_frac * n) {
        if (stats)
          std::cout << "# ND compression: skipped (fewer than "
                    << min_frac * 100 << "% duplicates)" << std::endl;
        return false;
      }
      // number the groups in the order of their smallest member
      // (the stable sort keeps the members in increasing order)
      thrust::device_vector<int> rep(ng), isrep(n, 0), newid(n);
      thrust::for_each_n(thrust::device, cnt0, n, nd_group_rep_ftor
                         {p(head), p(gs), p(ids), p(rep), p(isrep)});
      thrust::exclusive_scan(isrep.begin(), isrep.end(), newid.begin());
      C.ng = ng;
      C.gid.resize(n);
      thrust::for_each_n(thrust::device, cnt0, n, nd_group_gid_ftor
                         {p(gs), p(ids), p(rep), p(newid), p(C.gid)});
      C.gmem.resize(n);
      thrust::sequence(C.gmem.begin(), C.gmem.end());
      C.vw.resize(ng);
      {
        thrust::device_vector<int> keys(C.gid);
        thrust::stable_sort_by_key(keys.begin(), keys.end(), C.gmem.begin());
        thrust::reduce_by_key(keys.begin(), keys.end(),
                              thrust::constant_iterator<int>(1),
                              thrust::make_discard_iterator(), C.vw.begin());
      }
      C.goff.resize(ng+1);
      thrust::exclusive_scan(C.vw.begin(), C.vw.end(), C.goff.begin());
      C.goff[ng] = n;
      // merged graph: adjacency of the first member of each group,
      // mapped to groups, without the group itself and duplicates
      thrust::device_vector<integer> toff(ng+1, 0);
      thrust::transform(cnt0, cnt0+ng, toff.begin(),
                        nd_rep_deg_ftor<integer>{ptr, p(C.goff), p(C.gmem)});
      thrust::exclusive_scan(toff.begin(), toff.end(), toff.begin());
      integer m = toff[ng];
      thrust::device_vector<ull> key(m);
      thrust::for_each_n
        (thrust::device, cnt0, ng, nd_merged_keys_ftor<integer>
         {ptr, ind, p(toff), p(C.goff), p(C.gmem), p(C.gid), p(key)});
      thrust::sort(key.begin(), key.end());
      auto kend = thrust::unique(key.begin(), key.end());
      kend = thrust::lower_bound(key.begin(), kend, ~0ull);
      integer m2 = kend - key.begin();
      C.cind.resize(m2);
      thrust::transform(key.begin(), kend, C.cind.begin(),
                        nd_low32_ftor<integer>());
      C.cptr.resize(ng+1);
      auto rows = thrust::make_transform_iterator
        (key.begin(), nd_high32_ftor<integer>());
      thrust::lower_bound(rows, rows + m2,
                          thrust::counting_iterator<integer>(0),
                          thrust::counting_iterator<integer>(ng+1),
                          C.cptr.begin());
      if (stats) {
        thrust::host_vector<int> hvw(C.vw);
        long long sz[5] = {0, 0, 0, 0, 0};
        for (auto w : hvw) sz[std::min(w, 4)]++;
        std::cout << "# ND compression: size1=" << sz[1]
                  << " size2=" << sz[2] << " size3=" << sz[3]
                  << " size4+=" << sz[4] << ", merged nnz=" << m2
                  << " (was " << nnz << ")" << std::endl;
      }
      return true;
    }

    template<typename integer> SeparatorTree<integer>
    nd_bfs_gpu(integer n, integer* ptr, integer* ind,
               std::vector<integer>& perm, std::vector<integer>& iperm) {
      using clk = std::chrono::steady_clock;
      auto secs = [](clk::time_point a) {
        return std::chrono::duration<double>(clk::now() - a).count();
      };
      // the device code uses int for node indices, labels, component
      // data and tree node ids, and n itself as the "unvisited" label
      if (n < 0 || static_cast<long long>(n) >=
          static_cast<long long>(std::numeric_limits<int>::max()))
        throw std::overflow_error
          ("nd_bfs_gpu: number of unknowns does not fit in an int");
      auto t0 = clk::now();
      integer nnz = ptr[n];
      thrust::device_vector<integer> dptr(n+1), dind(nnz), dperm(n);
      thrust::device_vector<int> dtnode(n, -1);
      std::vector<int> tree_parent;
      thrust::copy(ptr, ptr+n+1, dptr.begin());
      thrust::copy(ind, ind+nnz, dind.begin());
      double t_setup = secs(t0);
      // STRUMPACK_ND_COMPRESS=0 disables the graph compression,
      // STRUMPACK_ND_COMPRESS_MIN: minimum fraction of unknowns that
      // must be removed by the compression for it to be used
      static const bool compress =
        nd_env_int("STRUMPACK_ND_COMPRESS", 1) != 0;
      static const double compress_min = [] {
        auto s = std::getenv("STRUMPACK_ND_COMPRESS_MIN");
        return s ? std::atof(s) : 0.1;
      }();
      t0 = clk::now();
      nd_compressed<integer> C;
      bool compressed = compress && n > 0 &&
        nd_compress_graph(int(n), nnz, dptr, dind, compress_min, C,
                          nd_stats_enabled());
      double t_comp = secs(t0);
      t0 = clk::now();
      if (compressed) {
        // the original graph is not needed on the device anymore
        dptr.clear(); dptr.shrink_to_fit();
        dind.clear(); dind.shrink_to_fit();
        const int ng = C.ng;
        thrust::device_vector<integer> cperm(ng);
        thrust::device_vector<int> ctnode(ng, -1), off(ng);
        nd_bfs_device(ng, C.cptr, C.cind, cperm, ctnode, tree_parent);
        // expand: members of a group are consecutive in perm
        thrust::exclusive_scan
          (thrust::make_permutation_iterator(C.vw.begin(), cperm.begin()),
           thrust::make_permutation_iterator(C.vw.begin(), cperm.end()),
           off.begin());
        thrust::for_each_n
          (thrust::device, thrust::counting_iterator<int>(0), ng,
           nd_expand_ftor<integer>
           {thrust::raw_pointer_cast(cperm.data()),
            thrust::raw_pointer_cast(off.data()),
            thrust::raw_pointer_cast(C.goff.data()),
            thrust::raw_pointer_cast(C.gmem.data()),
            thrust::raw_pointer_cast(dperm.data())});
        thrust::gather(C.gid.begin(), C.gid.end(), ctnode.begin(),
                       dtnode.begin());
      } else
        nd_bfs_device(n, dptr, dind, dperm, dtnode, tree_parent);
      double t_dev = secs(t0);
      t0 = clk::now();
      thrust::copy(dperm.begin(), dperm.end(), perm.begin());
      // STRUMPACK_ND_ETREE_TREE=1: old behavior, rebuild the tree
      // from the elimination tree of the permutation
      static const bool etree_tree = [] {
        auto s = std::getenv("STRUMPACK_ND_ETREE_TREE");
        return s && std::atoi(s) != 0;
      }();
      std::vector<int> tnode(n);
      thrust::copy(dtnode.begin(), dtnode.end(), tnode.begin());
      bool tagged = std::all_of
        (tnode.begin(), tnode.end(), [&](int t) {
          return t >= 0 && t < int(tree_parent.size()); });
      if (!etree_tree && !tagged)
        std::cerr << "# WARNING: ND tree structure incomplete,"
                  << " falling back to elimination tree" << std::endl;
      if (etree_tree || !tagged) {
        for (integer i=0; i<n; i++)
          iperm[perm[i]] = i;
        return build_sep_tree_from_perm(ptr, ind, iperm, perm);
      }
      auto tree = nd_tree_from_structure
        (n, ptr, ind, tree_parent, tnode, perm, iperm);
      if (nd_stats_enabled()) {
        std::cout << "# ND tree: " << tree_parent.size()
                  << " ND nodes, " << tree.separators() << " fronts, "
                  << tree.levels() << " levels" << std::endl;
        std::cout << "# ND timing: setup+H2D " << t_setup
                  << " s, compression " << t_comp
                  << " s, device ND " << t_dev
                  << " s, D2H+tree build " << secs(t0) << " s" << std::endl;
      }
      // perm (new -> old) and iperm (old -> new), same convention as
      // the build_sep_tree_from_perm(ptr, ind, iperm, perm) path above
      return tree;
    }

    template SeparatorTree<int>
    nd_bfs_gpu(int n, int* ptr, int* ind,
               std::vector<int>& perm, std::vector<int>& iperm);
    template SeparatorTree<long int>
    nd_bfs_gpu(long int n, long int* ptr, long int* ind,
               std::vector<long int>& perm, std::vector<long int>& iperm);
    template SeparatorTree<long long int>
    nd_bfs_gpu(long long int n, long long int* ptr, long long int* ind,
               std::vector<long long int>& perm,
               std::vector<long long int>& iperm);

  } // end namespace ordering
} // end namespace strumpack
