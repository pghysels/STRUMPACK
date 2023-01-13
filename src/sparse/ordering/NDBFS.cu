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

#include "ANDSparspak.hpp"

namespace strumpack {
  namespace ordering {

    template<typename integer> struct bfs_label_prop_ftor {
      integer *ptr, *ind;
      bool *elim, *running;
      int *root;
      bfs_label_prop_ftor(integer* ptr_, integer* ind_,
                          bool* elim_, int* root_, bool* running_)
        : ptr(ptr_), ind(ind_), elim(elim_),
          root(root_), running(running_) {}
      __device__ void operator()(int i) {
        if (elim[i]) return;
        for (auto k=ptr[i]; k<ptr[i+1]; k++) {
          auto j = ind[k];
          if (elim[j]) continue;
          if (root[j] < root[i]) {
            root[i] = root[j];
            *running = true;
          }
        }
      }
    };

    template<typename integer> struct bfs_layer_ftor {
      integer *ptr, *ind;
      bool *elim, *running;
      int *root, *mask, *sc, n, it;
      bfs_layer_ftor(int n_, int it_, integer* ptr_, integer* ind_,
                     bool* elim_, int* root_, int* mask_,
                     int* sc_, bool* running_)
        : n(n_), it(it_), ptr(ptr_), ind(ind_), elim(elim_),
          root(root_), mask(mask_), sc(sc_), running(running_) {}
      __device__ void operator()(int i) {
        if (elim[i]) return;
        auto ri = root[i];
        auto l = sc[mask[i]] + it;
        if (ri == l) {
          bool run = false;
          for (auto k=ptr[i]; k<ptr[i+1]; k++) {
            auto j = ind[k];
            if (elim[j]) continue;
            if (root[j] == n) {
              root[j] = l + 1;
              run = true;
            }
          }
          if (run && !*running)
            *running = true;
        }
      }
    };

    struct find_components_ftor {
      int *nc, *root, *mask, *rc;
      bool *elim;
      find_components_ftor(int* nc_, bool* elim_, int* root_,
                           int* mask_, int* rc_)
        : nc(nc_), elim(elim_), root(root_), mask(mask_), rc(rc_) {}
      __device__ void operator()(int i) {
        if (!elim[i] && root[i] == i) {
          auto c = atomicAdd(nc, 1);
          mask[i] = c;
          rc[c] = i;
        }
      }
    };

    struct assign_component_ftor {
      bool *elim;
      int *root, *mask;
      assign_component_ftor(bool* elim_, int* root_, int* mask_)
        : elim(elim_), root(root_), mask(mask_) {}
      __device__ void operator()(int i) {
        if (!elim[i])
          mask[i] = mask[root[i]];
      }
    };

    struct count_components_ftor {
      bool* elim;
      int *mask, *sc;
      count_components_ftor(bool* elim_, int* mask_, int* sc_)
        : elim(elim_), mask(mask_), sc(sc_) {}
      __device__ void operator()(int i) {
        if (!elim[i])
          atomicAdd(sc+mask[i], 1);
      }
    };

    struct count_layers_ftor {
      bool *elim;
      int n, *root, *mask, *ls, *sc;
      count_layers_ftor(int n_, bool* elim_, int* root_,
                        int* mask_, int* ls_, int* sc_)
        : n(n_), elim(elim_), root(root_), mask(mask_), ls(ls_), sc(sc_) {}
      __device__ void operator()(int i) {
        if (elim[i]) return;
        auto ri = root[i];
        if (ri == n) return;
        auto m = mask[i];
        auto l = ri - sc[m] + 1;
        if (l > ls[m])
          atomicMax(ls+m, l);
      }
    };

    template<typename integer> struct find_peripheral_node_1_ftor {
      integer *ptr, *ind;
      bool *elim;
      int *root, *mask, *ls, *old_ls, *d, *sc;
      find_peripheral_node_1_ftor(integer* ptr_, integer* ind_,
                                  bool* elim_, int* root_, int* mask_,
                                  int* ls_, int* old_ls_, int* d_, int* sc_)
        : ptr(ptr_), ind(ind_), elim(elim_), root(root_), mask(mask_),
          ls(ls_), old_ls(old_ls_), d(d_), sc(sc_) {}
      __device__ void operator()(int i) {
        // find the node with the minimum degree in the last layer, for
        // each of the components. We cannot atomically update both the
        // degree and the index of the node which had the minimum
        // degree, hence the two loops.
        if (elim[i]) return;
        auto mi = mask[i];
        if (ls[mi] <= old_ls[mi]) return;
        if (root[i] == sc[mi] + ls[mi] - 1) {
          // in the last layer
          int deg = 0;
          for (auto k=ptr[i]; k<ptr[i+1]; k++)
            if (!elim[ind[k]])
              deg++;
          atomicMin(d+mi, deg);
        }
      }
    };

    template<typename integer> struct find_peripheral_node_2_ftor {
      integer *ptr, *ind;
      bool *elim;
      int *root, *mask, *ls, *old_ls, *d, *sc, *rc;
      find_peripheral_node_2_ftor(integer* ptr_, integer* ind_,
                                  bool* elim_, int* root_, int* mask_,
                                  int* ls_, int* old_ls_,
                                  int* d_, int* sc_, int* rc_)
        : ptr(ptr_), ind(ind_), elim(elim_), root(root_), mask(mask_),
          ls(ls_), old_ls(old_ls_), d(d_), sc(sc_), rc(rc_) {}
      __device__ void operator()(int i) {
        if (elim[i]) return;
        auto mi = mask[i];
        if (ls[mi] <= old_ls[mi]) return;
        if (root[i] == sc[mi] + ls[mi] - 1) {
          // in the last layer
          int deg = 0;
          for (auto k=ptr[i]; k<ptr[i+1]; k++)
            if (!elim[ind[k]])
              deg++;
          if (deg == d[mi])
            // multiple threads might be writing concurrently here if
            // multiple nodes have the same (minimum) degree
            rc[mi] = i;
        }
      }
    };

    // store layer cardinalities for the component starting at node i
    // (== root[i]) at d[i], and at d[i+1] for the next layer, etc.
    // count nr of nodes in each separator, where a separator is all
    // nodes in a layer that are also connected to the next layer,
    // store result in ns[i] for the root, in ns[i+1] for the next
    // layer, etc
    template<typename integer> struct cardinalities_ftor {
      integer *ptr, *ind;
      bool *elim;
      int *root, *mask, *ls, *sc, *d, *ns, *ns2;
      cardinalities_ftor(integer* ptr_, integer* ind_,
                         bool* elim_, int* root_, int* mask_,
                         int* ls_, int* sc_, int* d_,
                         int* ns_, int* ns2_)
        : ptr(ptr_), ind(ind_), elim(elim_), root(root_), mask(mask_),
          ls(ls_), sc(sc_), d(d_), ns(ns_), ns2(ns2_) {}
      __device__ void operator()(int i) {
        if (elim[i]) return;
        auto mi = mask[i];
        auto nl = ls[mi];
        if (nl < 3) return; // whole component eliminated
        auto lr = sc[mi];
        auto ri = root[i];
        atomicAdd(d+ri, 1);
        if (ri == lr || ri == lr+nl-1) return; // ignore first/last
        for (auto k=ptr[i]; k<ptr[i+1]; k++) {
          auto j = ind[k];
          if (elim[j]) continue;
          if (root[j] == ri - 1) {
            atomicAdd(ns+ri, 1);
            // return;
            break;
          }
        }
        for (auto k=ptr[i]; k<ptr[i+1]; k++) {
          auto j = ind[k];
          if (elim[j]) continue;
          if (root[j] == ri + 1) {
            atomicAdd(ns2+ri, 1);
            // return;
            break;
          }
        }
      }
    };

    template<typename integer> struct minimize_separator_ftor {
      integer *ptr, *ind;
      bool *elim;
      int *root, *mask, *ls, *sc, *ns, *ns2, *d;
      minimize_separator_ftor(integer* ptr_, integer* ind_,
                              bool* elim_, int* root_, int* mask_,
                              int* ls_, int* sc_,
                              int* ns_, int* ns2_, int* d_)
        : ptr(ptr_), ind(ind_), elim(elim_), root(root_), mask(mask_),
          ls(ls_), sc(sc_), ns(ns_), ns2(ns2_), d(d_) {}
      __device__ void operator()(int i) {
        if (elim[i]) return;
        auto mi = mask[i];
        auto nl = ls[mi];
        if (nl < 3 || root[i] != sc[mi]) return;
        // I'm the root of this component
        auto scmi = sc[mi];
        auto cG = sc[mi+1] - scmi;
        auto cA = d[scmi];
        auto cB = cG - cA;
        float minNS;
        int minl;
        for (auto l=1; l<nl-1; l++) {
          auto dl = d[scmi+l];
          auto sl = ns[scmi+l];
          cB -= dl;
          auto NS = sl * (1. / cA + 1. / (cB+dl-sl));
          if (l == 1 || NS < minNS) {
            minNS = NS;
            minl = -l;
          }
          sl = ns2[scmi+l];
          NS = sl * (1. / (cA+dl-sl) + 1. / cB);
          if (NS < minNS) {
            minNS = NS;
            minl = l;
          }
          cA += dl;
        }
        ns[mi] = (minl < 0) ? -scmi + minl : scmi + minl;
      }
    };

    template<typename integer> struct eliminate_separators_ftor {
      integer *ptr, *ind, *perm;
      bool *elim;
      int *root, *mask, *ls, *ns, *nelim;
      eliminate_separators_ftor(integer* ptr_, integer* ind_, bool* elim_,
                                int* root_, int* mask_, int* ls_, int* ns_,
                                int* nelim_, integer* perm_)
        : ptr(ptr_), ind(ind_), elim(elim_), root(root_), mask(mask_),
          ls(ls_), ns(ns_), nelim(nelim_), perm(perm_) {}
      __device__ void operator()(int i) {
        if (elim[i]) return;
        auto mi = mask[i];
        if (ls[mi] < 3) {
          elim[i] = true;
          perm[atomicAdd(nelim, 1)] = i;
          return;
        }
        // the separator are the nodes in layer ns[root[i]] (see
        // above) which are also connected to the next layer.
        auto ri = root[i];
        if (ri == -ns[mi]) { // best layer
          for (auto k=ptr[i]; k<ptr[i+1]; k++) {
            auto j = ind[k];
            if (elim[j]) continue;
            if (root[j] == ri - 1) {
              elim[i] = true;
              perm[atomicAdd(nelim, 1)] = i;
              return;
            }
          }
        }
        if (ri == ns[mi]) { // best layer
          for (auto k=ptr[i]; k<ptr[i+1]; k++) {
            auto j = ind[k];
            if (elim[j]) continue;
            if (root[j] == ri + 1) {
              elim[i] = true;
              perm[atomicAdd(nelim, 1)] = i;
              return;
            }
          }
        }
      }
    };

    struct tuple_less_equal_ftor {
      __device__ bool operator()(const thrust::tuple<int,int>& t) {
        return thrust::get<0>(t) <= thrust::get<1>(t);
      }
    };
    struct tuple_equal_ftor {
      __device__ bool operator()(const thrust::tuple<int,int>& t) {
        return thrust::get<0>(t) == thrust::get<1>(t);
      }
    };

    template<typename integer> void
    nd_bfs_device(int n, thrust::device_vector<integer>& ptr,
                  thrust::device_vector<integer>& ind,
                  thrust::device_vector<integer>& perm) {
      thrust::device_vector<int> root(n), mask(n), dnc(1), nelim(1, 0),
        d(n), old_ls(n), ls(n), sc(n+1), rc(n), ns2(n);
      thrust::device_vector<bool> elim(n, false), running(1);
      thrust::counting_iterator<int> iter(0);
      auto p = [](auto& v) { return thrust::raw_pointer_cast(v.data()); };
      auto run = [&iter,&n](const auto& f) {
        thrust::for_each_n(thrust::device, iter, n, f); };
      // int lvl = 0;
      do {
        // std::cout << "lvl: " << lvl++ << " nelim: " << nelim[0] << std::endl;
        thrust::sequence(root.begin(), root.end());
        do { // find all conn. components using label propagation BFS
          running[0] = false;
          run(bfs_label_prop_ftor<integer>
              (p(ptr), p(ind), p(elim), p(root), p(running)));
        } while (running[0]);
        // count components. TODO thrust::reduce_by_key (after sort)?
        dnc[0] = 0;
        run(find_components_ftor(p(dnc), p(elim), p(root), p(mask), p(rc)));
        int nc = dnc[0];
        run(assign_component_ftor(p(elim), p(root), p(mask)));
        thrust::fill_n(sc.begin(), nc+1, 0);
        run(count_components_ftor(p(elim), p(mask), p(sc)));
        thrust::exclusive_scan(sc.begin(), sc.begin()+nc+1, sc.begin());
        thrust::fill_n(root.begin(), n, n);
        thrust::scatter(sc.begin(), sc.begin()+nc, rc.begin(), root.begin());
        thrust::fill_n(old_ls.begin(), nc, 1);
        do {
          int it = 0;
          do { // BFS from the root node of each component
            running[0] = false;
            run(bfs_layer_ftor<integer>
                (n, it++, p(ptr), p(ind), p(elim), p(root),
                 p(mask), p(sc), p(running)));
          } while (running[0]); // repeat until BFS for each component is done
          thrust::fill_n(ls.begin(), nc, 1);
          run(count_layers_ftor(n, p(elim), p(root), p(mask), p(ls), p(sc)));
          if (thrust::all_of
              (thrust::make_zip_iterator
               (thrust::make_tuple(ls.begin(), old_ls.begin())),
               thrust::make_zip_iterator
               (thrust::make_tuple(ls.begin()+nc, old_ls.begin()+nc)),
               tuple_less_equal_ftor())) break;
          thrust::fill_n(d.begin(), nc, n);
          run(find_peripheral_node_1_ftor<integer>
              (p(ptr), p(ind), p(elim), p(root), p(mask),
               p(ls), p(old_ls), p(d), p(sc)));
          run(find_peripheral_node_2_ftor<integer>
              (p(ptr), p(ind), p(elim), p(root), p(mask),
               p(ls), p(old_ls), p(d), p(sc), p(rc)));
          old_ls = ls;
          thrust::fill_n(root.begin(), n, n);
          thrust::scatter(sc.begin(), sc.begin()+nc, rc.begin(), root.begin());
        } while (1); // repeat BFS with new start nodes
        auto& ns = rc;
        thrust::fill_n(d.begin(), n, 0);
        thrust::fill_n(ns.begin(), n, 0);
        thrust::fill_n(ns2.begin(), n, 0);
        run(cardinalities_ftor<integer>
            (p(ptr), p(ind), p(elim), p(root), p(mask),
             p(ls), p(sc), p(d), p(ns), p(ns2)));
        run(minimize_separator_ftor<integer>
            (p(ptr), p(ind), p(elim), p(root), p(mask),
             p(ls), p(sc), p(ns), p(ns2), p(d)));
        run(eliminate_separators_ftor<integer>
            (p(ptr), p(ind), p(elim), p(root), p(mask),
             p(ls), p(ns), p(nelim), p(perm)));
      } while (nelim[0] < n);
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
    nd_bfs_cuda(int neqns, int* ptr, int* ind,
                std::vector<int>& perm, std::vector<int>& iperm);
    template SeparatorTree<long int>
    nd_bfs_cuda(long int neqns, long int* ptr, long int* ind,
                std::vector<long int>& perm, std::vector<long int>& iperm);
    template SeparatorTree<long long int>
    nd_bfs_cuda(long long int neqns, long long int* ptr, long long int* ind,
                std::vector<long long int>& perm,
                std::vector<long long int>& iperm);

  } // end namespace ordering
} // end namespace strumpack
