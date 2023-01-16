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
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/linear_feedback_shift_engine.h>
#include <thrust/random/uniform_int_distribution.h>

#include "ANDSparspak.hpp"

namespace strumpack {
  namespace ordering {

    template<typename integer> struct bfs_label_prop_ftor {
      integer *ptr, *ind;
      bool *eliminated, *running;
      int *label;
      bfs_label_prop_ftor(integer* ptr_, integer* ind_,
                          bool* eliminated_, int* label_,
                          bool* running_)
        : ptr(ptr_), ind(ind_), eliminated(eliminated_),
          label(label_), running(running_) {}
      __device__ void operator()(int i) {
        if (eliminated[i]) return;
        auto khi = ptr[i+1];
        for (auto k=ptr[i]; k<khi; k++) {
          auto j = ind[k];
          if (eliminated[j]) continue;
          if (label[j] < label[i]) {
            label[i] = label[j];
            // TODO atomicExch?
            *running = true;
          }
        }
      }
    };

    // #define FRONT_BFS
#if defined(FRONT_BFS)
    template<typename integer> struct bfs_layer_front_ftor {
      integer *ptr, *ind;
      bool *eliminated;
      int n, *label, *current_front, *next_front, *next_front_size;
      bfs_layer_front_ftor(int n_, integer* ptr_, integer* ind_,
                           bool* eliminated_, int* label_,
                           int* current_front_, int* next_front_,
                           int* next_front_size_)
        : n(n_), ptr(ptr_), ind(ind_), eliminated(eliminated_),
          label(label_), current_front(current_front_),
          next_front(next_front_), next_front_size(next_front_size_) {}
      __device__ void operator()(int fi) {
        auto i = current_front[fi];
        auto li1 = label[i]+1;
        auto khi = ptr[i+1];
        for (auto k=ptr[i]; k<khi; k++) {
          auto j = ind[k];
          if (eliminated[j]) continue;
          if (atomicCAS(label+j, n, li1) == n)
            next_front[atomicAdd(next_front_size, 1)] = j;
        }
      }
    };
#else
    template<typename integer> struct bfs_layer_ftor {
      integer *ptr, *ind;
      bool *eliminated, *running;
      int *label, *comp, *comp_start, n, it;
      bfs_layer_ftor(int n_, int it_, integer* ptr_, integer* ind_,
                     bool* eliminated_, int* label_, int* comp_,
                     int* comp_start_, bool* running_)
        : n(n_), it(it_), ptr(ptr_), ind(ind_), eliminated(eliminated_),
          label(label_), comp(comp_), comp_start(comp_start_),
          running(running_) {}
      __device__ void operator()(int i) {
        // TODO write from the destination instead of the source?
        if (eliminated[i]) return;
        auto li = label[i];
        if (li == comp_start[comp[i]] + it) {
          bool run = false;
          auto khi = ptr[i+1];
          for (auto k=ptr[i]; k<khi; k++) {
            auto j = ind[k];
            if (eliminated[j]) continue;
            if (atomicCAS(label+j, n, li+1) == n)
              run = true;
          }
          if (run && !*running)
            // TODO atomicExch ?
            *running = true;
        }
      }
    };
#endif

    struct find_comps_ftor {
      int *nr_comps, *label, *comp, *comp_root;
      bool *eliminated;
      find_comps_ftor(int* nr_comps_, bool* eliminated_, int* label_,
                      int* comp_, int* comp_root_)
        : nr_comps(nr_comps_), eliminated(eliminated_), label(label_),
          comp(comp_), comp_root(comp_root_) {}
      __device__ void operator()(int i) {
        if (!eliminated[i] && label[i] == i) {
          auto c = atomicAdd(nr_comps, 1);
          comp[i] = c;
          comp_root[c] = i;
        }
      }
    };

    struct count_comps_ftor {
      bool *eliminated;
      int nr_comps, *comp, *comp_size;
      count_comps_ftor(bool* eliminated_, int nr_comps_,
                       int* comp_, int* comp_size_)
        : eliminated(eliminated_), nr_comps(nr_comps_), comp(comp_),
          comp_size(comp_size_) {}
      __device__ void operator()(int i) {
        if (!eliminated[i])
          atomicAdd(comp_size+comp[i], 1);
        else
          comp[i] = nr_comps;
      }
    };

    struct random_roots_ftor {
      int seed, *root, *start, *map;
      random_roots_ftor(int seed_, int* root_, int* start_, int* map_)
        : seed(seed_), root(root_), start(start_), map(map_) {}
      __device__ //__forceinline__
      void operator()(int i) {
        thrust::minstd_rand gen(seed);
        thrust::uniform_int_distribution<int>
          dist(start[i], start[i+1]-1);
        root[i] = map[dist(gen)];
      }
    };

    struct count_layers_ftor {
      bool *eliminated;
      int n, *label, *comp, *levels, *comp_start;
      count_layers_ftor(int n_, bool* eliminated_, int* label_,
                        int* comp_, int* levels_, int* comp_start_)
        : n(n_), eliminated(eliminated_), label(label_), comp(comp_),
          levels(levels_), comp_start(comp_start_) {}
      __device__ void operator()(int i) {
        if (eliminated[i]) return;
        auto li = label[i];
        if (li == n) return;
        auto ci = comp[i];
        auto lvl = li - comp_start[ci] + 1;
        if (lvl > levels[ci])
          atomicMax(levels+ci, lvl);
      }
    };

    template<typename integer> struct peripheral_node_1_ftor {
      integer *ptr, *ind;
      bool *eliminated;
      int *label, *comp, *levels, *old_levels, *d, *comp_start;
      peripheral_node_1_ftor(integer* ptr_, integer* ind_,
                             bool* eliminated_, int* label_, int* comp_,
                             int* levels_, int* old_levels_,
                             int* d_, int* comp_start_)
        : ptr(ptr_), ind(ind_), eliminated(eliminated_),
          label(label_), comp(comp_),
          levels(levels_), old_levels(old_levels_),
          d(d_), comp_start(comp_start_) {}
      __device__ void operator()(int i) {
        // find the node with the minimum degree in the last layer, for
        // each of the components. We cannot atomically update both the
        // degree and the index of the node which had the minimum
        // degree, hence the two loops.
        if (eliminated[i]) return;
        auto ci = comp[i];
        auto nlvls = levels[ci];
        if (nlvls <= old_levels[ci]) return;
        if (label[i] == comp_start[ci] + nlvls - 1) {
          // in the last layer
          int deg = 0;
          auto khi = ptr[i+1];
          for (auto k=ptr[i]; k<khi; k++)
            if (!eliminated[ind[k]])
              deg++;
          atomicMin(d+ci, deg);
        }
      }
    };

    template<typename integer> struct peripheral_node_2_ftor {
      integer *ptr, *ind;
      bool *eliminated;
      int *label, *comp, *levels, *old_levels,
        *d, *comp_start, *comp_root;
      peripheral_node_2_ftor(integer* ptr_, integer* ind_,
                             bool* eliminated_, int* label_, int* comp_,
                             int* levels_, int* old_levels_, int* d_,
                             int* comp_start_, int* comp_root_)
        : ptr(ptr_), ind(ind_), eliminated(eliminated_), label(label_),
          comp(comp_), levels(levels_), old_levels(old_levels_), d(d_),
          comp_start(comp_start_), comp_root(comp_root_) {}
      __device__ void operator()(int i) {
        if (eliminated[i]) return;
        auto ci = comp[i];
        auto nlvls = levels[ci];
        if (nlvls <= old_levels[ci]) return;
        if (label[i] == comp_start[ci] + nlvls - 1) {
          // in the last layer
          int deg = 0;
          auto khi = ptr[i+1];
          for (auto k=ptr[i]; k<khi; k++)
            if (!eliminated[ind[k]])
              deg++;
          if (deg == d[ci])
            // multiple threads might be writing concurrently here if
            // multiple nodes have the same (minimum) degree
            //atomicExch(comp_root+ci, i);
            comp_root[ci] = i;
        }
      }
    };

    // Store layer cardinalities for the component starting at node i
    // (== label[i]) at layer_size[i], and at layer_size[i+1] for the
    // next layer, etc. Count nr of nodes in each separator, where a
    // separator is all nodes in a layer that are also connected to
    // the next layer, store result in sep_size[l] for level l.
    template<typename integer> struct cardinalities_ftor {
      integer *ptr, *ind;
      bool *eliminated;
      int *label, *comp, *levels, *comp_start,
        *layer_size, *sep_size;
      cardinalities_ftor(integer* ptr_, integer* ind_,
                         bool* eliminated_, int* label_, int* comp_,
                         int* levels_, int* comp_start_, int* layer_size_,
                         int* sep_size_)
        : ptr(ptr_), ind(ind_), eliminated(eliminated_), label(label_),
          comp(comp_), levels(levels_), comp_start(comp_start_),
          layer_size(layer_size_), sep_size(sep_size_) {}
      __device__ void operator()(int i) {
        if (eliminated[i]) return;
        auto ci = comp[i];
        auto nlvls = levels[ci];
        if (nlvls < 3) // whole component eliminated
          return;
        auto l0 = comp_start[ci];
        auto li = label[i];
        atomicAdd(layer_size+li, 1);
        if (li == l0 || li == l0+nlvls-1) // ignore first/last
          return;
        auto khi = ptr[i+1];
        for (auto k=ptr[i]; k<khi; k++) {
          auto j = ind[k];
          if (eliminated[j]) continue;
          if (label[j] == li + 1) {
            atomicAdd(sep_size+li, 1);
            return;
          }
        }
      }
    };

    template<typename integer> struct minimize_sep_ftor {
      integer *ptr, *ind;
      bool *eliminated;
      int *label, *comp, *levels, *comp_start, *layer_size,
        *sep_size, *comp_root, *comp_root_opt, *sep_level;
      float *min_norm_sep;
      minimize_sep_ftor(integer* ptr_, integer* ind_,
                        bool* eliminated_, int* label_, int* comp_,
                        int* levels_, int* comp_start_,
                        int* layer_size_, int* sep_size_,
                        int* comp_root_, int* comp_root_opt_,
                        int* sep_level_, float* min_norm_sep_)
        : ptr(ptr_), ind(ind_),
          eliminated(eliminated_), label(label_), comp(comp_),
          levels(levels_), comp_start(comp_start_),
          layer_size(layer_size_), sep_size(sep_size_),
          comp_root(comp_root_), comp_root_opt(comp_root_opt_),
          sep_level(sep_level_), min_norm_sep(min_norm_sep_) {}
      __device__ void operator()(int i) {
        if (eliminated[i]) return;
        auto ci = comp[i];
        auto nlvls = levels[ci];
        if (nlvls < 3 || label[i] != comp_start[ci]) return;
        // I'm the root of this component
        auto c0 = comp_start[ci];
        auto cA = layer_size[c0];
        auto cB = comp_start[ci+1] - c0 - cA;
        float minNS;
        int minl;
        for (auto l=1; l<nlvls-1; l++) {
          auto cL = layer_size[c0+l];
          cB -= cL;
          auto cS = sep_size[c0+l];
          auto NS = cS * (1. / (cA+cL-cS) + 1. / cB);
          if (l == 1 || NS < minNS) {
            minNS = NS;
            minl = l;
          }
          cA += cL;
        }
        if (minNS < min_norm_sep[ci]) {
          sep_level[ci] = c0 + minl;
          comp_root_opt[ci] = comp_root[ci];
          min_norm_sep[ci] = minNS;
        }
      }
    };

    template<typename integer> struct eliminate_sep_ftor {
      integer *ptr, *ind, *perm;
      bool *eliminated;
      int *label, *comp, *levels, *sep_size, *nr_eliminated;
      eliminate_sep_ftor(integer* ptr_, integer* ind_,
                         bool* eliminated_, int* label_, int* comp_,
                         int* levels_, int* sep_size_,
                         int* nr_eliminated_, integer* perm_)
        : ptr(ptr_), ind(ind_), eliminated(eliminated_), label(label_),
          comp(comp_), levels(levels_), sep_size(sep_size_),
          nr_eliminated(nr_eliminated_), perm(perm_) {}
      __device__ void operator()(int i) {
        if (eliminated[i]) return;
        auto ci = comp[i];
        if (levels[ci] < 3) {
          eliminated[i] = true;
          perm[atomicAdd(nr_eliminated, 1)] = i;
          return;
        }
        auto li = label[i];
        if (li == sep_size[ci]) { // best layer
          auto khi = ptr[i+1];
          for (auto k=ptr[i]; k<khi; k++) {
            auto j = ind[k];
            if (eliminated[j]) continue;
            if (label[j] == li + 1) {
              eliminated[i] = true;
              perm[atomicAdd(nr_eliminated, 1)] = i;
              return;
            }
          }
        }
      }
    };

    struct tuple_less_equal_ftor {
      __device__ __forceinline__
      bool operator()(const thrust::tuple<int,int>& t) {
        return thrust::get<0>(t) <= thrust::get<1>(t);
      }
    };
    struct tuple_equal_ftor {
      __device__ __forceinline__
      bool operator()(const thrust::tuple<int,int>& t) {
        return thrust::get<0>(t) == thrust::get<1>(t);
      }
    };

    template<typename integer> void
    nd_bfs_device(int n, thrust::device_vector<integer>& ptr,
                  thrust::device_vector<integer>& ind,
                  thrust::device_vector<integer>& perm) {
      thrust::device_vector<int>
        d_nr_comps(1),        // nr of connected components,
                              // stored on device
        nr_eliminated(1, 0),  // nr of already eliminated nodes
        label(n),             // node label, as used in label
                              // propagation breadth first search for
                              // finding connected components, and in
                              // the BFS to find the level sets
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
#if defined(FRONT_BFS)
      thrust::device_vector<int>
        current_front(n),
        next_front(n),
        next_front_size(1, 0);
#endif
        thrust::device_vector<float>
        min_norm_sep(n);      // the minimal normalized separator
                              // value, for each component
      thrust::device_vector<bool>
        eliminated(n, false), // bool flags for each node indicating
                              // if the node has been eliminated yet
        running(1);           // bool flag to check whether the BFS
                              // has completed
      thrust::minstd_rand rng;
      auto p = [](auto& v) {
        return thrust::raw_pointer_cast(v.data());
      };
      auto run = [&n](const auto& f) {
        thrust::for_each_n
          (thrust::device, thrust::counting_iterator<int>(0), n, f);
      };
      // int lvl = 0;
      do {
        // std::cout << "lvl: " << lvl++
        //           << " nr_eliminated: " << nr_eliminated[0] << std::endl;
        thrust::sequence(label.begin(), label.end());
        do {
          // find all conn-comps using label prop BFS
          running[0] = false;
          run(bfs_label_prop_ftor<integer>
              (p(ptr), p(ind), p(eliminated), p(label), p(running)));
        } while (running[0]);
        d_nr_comps[0] = 0;
        run(find_comps_ftor
            (p(d_nr_comps), p(eliminated), p(label),
             p(comp), p(comp_root)));
        int nr_comps = d_nr_comps[0];
        thrust::gather // comp[i] = comp[label[i]];
          (label.begin(), label.end(), comp.begin(), comp.begin());
        thrust::fill_n(comp_start.begin(), nr_comps+1, 0);
        run(count_comps_ftor(p(eliminated), nr_comps, p(comp), p(comp_start)));
        thrust::exclusive_scan
          (comp_start.begin(), comp_start.begin()+nr_comps+1,
           comp_start.begin());
        thrust::fill_n(min_norm_sep.begin(), nr_comps,
                       std::numeric_limits<float>::max());

        // TODO eliminate small components with minimum degree

        auto find_level_sets = [&](thrust::device_vector<int>& roots) {
          thrust::fill_n(label.begin(), n, n);
          thrust::scatter // label[roots[i]] = comp_start[i]
            (comp_start.begin(), comp_start.begin()+nr_comps,
             roots.begin(), label.begin());
#if defined(FRONT_BFS)
          thrust::copy_n(roots.begin(), nr_comps, current_front.begin());
          int front_size = nr_comps;
          while (front_size) {
            // BFS from the root node of each component
            next_front_size[0] = 0;
            thrust::for_each_n
              (thrust::device, thrust::counting_iterator<int>(0), front_size,
               bfs_layer_front_ftor<integer>
                (n, p(ptr), p(ind), p(eliminated), p(label),
                 p(current_front), p(next_front), p(next_front_size)));
            std::swap(current_front, next_front);
            front_size = next_front_size[0];
            // repeat until BFS for each component is done
          }
#else
          int it = 0;
          do {
            // BFS from the root node of each component
            running[0] = false;
            run(bfs_layer_ftor<integer>
                (n, it++, p(ptr), p(ind), p(eliminated), p(label),
                 p(comp), p(comp_start), p(running)));
            // repeat until BFS for each component is done
          } while (running[0]);
#endif
          thrust::fill_n(levels.begin(), nr_comps, 1);
          run(count_layers_ftor
              (n, p(eliminated), p(label), p(comp),
               p(levels), p(comp_start)));
        };
        for (int rep=0; rep<3; rep++) {
          // find level sets with different random initial nodes
          if (rep > 0) {
            thrust::copy_n(comp.begin(), n, label.begin());
            thrust::sequence(degree.begin(), degree.end());
            thrust::sort_by_key(label.begin(), label.end(), degree.begin());
            thrust::for_each_n
              (thrust::device, thrust::counting_iterator<int>(0), nr_comps,
               random_roots_ftor
               (rng(), p(comp_root), p(comp_start), p(degree)));
          }
          thrust::fill_n(old_levels.begin(), nr_comps, 1);
          do {
            find_level_sets(comp_root);
            thrust::fill_n
              (thrust::make_zip_iterator(degree.begin(), sep_size.begin()),
               n, thrust::make_tuple(0, 0));
            run(cardinalities_ftor<integer>
                (p(ptr), p(ind), p(eliminated), p(label), p(comp),
                 p(levels), p(comp_start), p(degree), p(sep_size)));
            run(minimize_sep_ftor<integer>
                (p(ptr), p(ind), p(eliminated), p(label), p(comp), p(levels),
                 p(comp_start), p(degree), p(sep_size), p(comp_root),
                 p(comp_root_opt), p(sep_level), p(min_norm_sep)));
            if (thrust::all_of
                (thrust::make_zip_iterator
                 (thrust::make_tuple(levels.begin(), old_levels.begin())),
                 thrust::make_zip_iterator
                 (thrust::make_tuple(levels.begin()+nr_comps,
                                     old_levels.begin()+nr_comps)),
                 tuple_less_equal_ftor()))
              break;
            thrust::fill_n(degree.begin(), nr_comps, n);
            run(peripheral_node_1_ftor<integer>
                (p(ptr), p(ind), p(eliminated), p(label), p(comp),
                 p(levels), p(old_levels), p(degree), p(comp_start)));
            run(peripheral_node_2_ftor<integer>
                (p(ptr), p(ind), p(eliminated), p(label), p(comp),
                 p(levels), p(old_levels), p(degree),
                 p(comp_start), p(comp_root)));
            old_levels = levels;
            // repeat level set BFS with new start nodes, so that the
            // start nodes converge to peripheral nodes of the
            // different components
          } while (1);
        }
        find_level_sets(comp_root_opt);
        run(eliminate_sep_ftor<integer>
            (p(ptr), p(ind), p(eliminated), p(label), p(comp),
             p(levels), p(sep_level), p(nr_eliminated), p(perm)));
      } while (nr_eliminated[0] < n);
      thrust::reverse(perm.begin(), perm.end());
    }

    template<typename integer> SeparatorTree<integer>
    nd_bfs_cuda(integer n, integer* ptr, integer* ind,
                std::vector<integer>& perm, std::vector<integer>& iperm) {
      integer nnz = ptr[n];
      thrust::device_vector<integer> dptr(n+1), dind(nnz), dperm(n);
      thrust::copy(ptr, ptr+n+1, dptr.begin());
      thrust::copy(ind, ind+nnz, dind.begin());
      nd_bfs_device(n, dptr, dind, dperm);
      thrust::copy(dperm.begin(), dperm.end(), perm.begin());
      for (integer i=0; i<n; i++)
        iperm[perm[i]] = i;
      return build_sep_tree_from_perm(ptr, ind, iperm, perm);
    }

    template SeparatorTree<int>
    nd_bfs_cuda(int n, int* ptr, int* ind,
                std::vector<int>& perm, std::vector<int>& iperm);
    template SeparatorTree<long int>
    nd_bfs_cuda(long int n, long int* ptr, long int* ind,
                std::vector<long int>& perm, std::vector<long int>& iperm);
    template SeparatorTree<long long int>
    nd_bfs_cuda(long long int n, long long int* ptr, long long int* ind,
                std::vector<long long int>& perm,
                std::vector<long long int>& iperm);

  } // end namespace ordering
} // end namespace strumpack
