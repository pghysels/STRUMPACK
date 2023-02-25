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
#include <algorithm>
#include <stack>
#include <vector>
#include <bitset>

#include "ANDSparspak.hpp"

namespace strumpack {
  namespace ordering {

    template<typename integer> integer
    rootls(integer root, integer* xadj, integer* adjncy,
           integer* mask, integer* xls, integer* ls) {
      integer nlvl = 0;
      mask[root] = 0;
      ls[0] = root;
      /* LBEGIN IS THE POINTER TO THE BEGINNING OF THE CURRENT */
      /* LEVEL, AND LVLEND POINTS TO THE END OF THIS LEVEL. */
      integer lbegin, lvlend = 0, ccsize = 1;
      do {
        lbegin = lvlend;
        lvlend = ccsize;
        xls[nlvl++] = lbegin;
        /* GENERATE THE NEXT LEVEL BY FINDING ALL THE MASKED */
        /* NEIGHBORS OF NODES IN THE CURRENT LEVEL. */
        for (integer i=lbegin; i<lvlend; ++i) {
          integer node = ls[i], jstop = xadj[node+1];
          for (integer j=xadj[node]; j<jstop; ++j) {
            integer nbr = adjncy[j];
            if (!mask[nbr]) continue;
            ls[ccsize++] = nbr;
            mask[nbr] = 0;
          }
        }
        /* COMPUTE THE CURRENT LEVEL WIDTH. */
        /* IF IT IS NONZERO, GENERATE THE NEXT LEVEL. */
      } while (ccsize > lvlend);
      /* RESET MASK TO ONE FOR THE NODES IN THE LEVEL STRUCTURE. */
      xls[nlvl] = lvlend;
      for (integer i=0; i<ccsize; ++i)
        mask[ls[i]] = 1;
      return nlvl;
    }

    template<typename integer> integer
    fndsep(integer root, integer* xadj, integer* adjncy,
           integer* mask, integer* sep, integer* xls, integer* ls) {
      /* DETERMINE THE LEVEL STRUCTURE ROOTED AT ROOT. */
      integer nlvl = rootls(root, xadj, adjncy, mask, xls, ls);
      integer ccsize = xls[nlvl];
      if (!(nlvl == 1 || nlvl == ccsize)) {
        /* PICK A NODE WITH MINIMUM DEGREE FROM THE LAST LEVEL. */
        do {
          integer jstrt = xls[nlvl-1], mindeg = ccsize;
          root = ls[jstrt];
          if (ccsize != jstrt+1) {
            for (integer j=jstrt; j<ccsize; ++j) {
              integer node = ls[j], ndeg = 0, kstop = xadj[node+1];
              for (integer k=xadj[node]; k<kstop; ++k)
                if (mask[adjncy[k]] > 0) ++ndeg;
              if (ndeg >= mindeg) continue;
              root = node;
              mindeg = ndeg;
            }
          }
          /* AND GENERATE ITS ROOTED LEVEL STRUCTURE. */
          integer nunlvl = rootls(root, xadj, adjncy, mask, xls, ls);
          if (nunlvl <= nlvl) break;
          nlvl = nunlvl;
        } while (nlvl < ccsize);
      }
      /* IF THE NUMBER OF LEVELS IS LESS THAN 3, RETURN */
      /* THE WHOLE COMPONENT AS THE SEPARATOR. */
      if (nlvl < 3) {
        integer nsep = xls[nlvl];
        for (integer i=0; i<nsep; ++i) {
          integer node = ls[i];
          sep[i] = node;
          mask[node] = 0;
        }
        return nsep;
      }
      /* FIND THE MIDDLE LEVEL OF THE ROOTED LEVEL STRUCTURE. */
      integer midlvl = (nlvl + 2) / 2;
      integer midbeg = xls[midlvl - 1], mp1beg = xls[midlvl];
      integer midend = mp1beg, mp1end = xls[midlvl + 1];
      /* THE SEPARATOR IS OBTAINED BY INCLUDING ONLY THOSE */
      /* MIDDLE-LEVEL NODES WITH NEIGHBORS IN THE MIDDLE+1 */
      /* LEVEL. XADJ IS USED TEMPORARILY TO MARK THOSE */
      /* NODES IN THE MIDDLE+1 LEVEL. */
      for (integer i=mp1beg; i<mp1end; ++i)
        xadj[ls[i]] = -(xadj[ls[i]] + 1);
      integer nsep = 0;
      for (integer i=midbeg; i<midend; ++i) {
        integer node = ls[i], i2 = xadj[node+1];
        integer jstop = (i2 < 0) ? -(i2 + 1) : i2;
        for (integer j=xadj[node]; j<jstop; ++j) {
          if (xadj[adjncy[j]] >= 0) continue;
          sep[nsep++] = node;
          mask[node] = 0;
          break;
        }
      }
      /* TRY TO REFINE THIS MIDDLE SEPARATOR WITH */
      /* ADJACENT NODES IN LEVEL + 1. */
      /* ------------------------------- */
      /* RESET XADJ TO ITS CORRECT SIGN. */
      for (integer i=mp1beg; i<mp1end; ++i)
        xadj[ls[i]] = -(xadj[ls[i]] + 1);
      /* Try to improve the separator */
      return nsep;
    }


    template<typename integer> struct Comp {
      Comp(integer r, integer s) : root(r), size(s) {}
      integer root, size;
    };

    template<typename integer> integer
    comps(integer n, integer nsub, integer* xadj, integer* adjncy,
          integer* mask, integer* base, std::vector<Comp<integer>>& C,
          integer* ls=nullptr) {
      integer ncomps = 0;
      for (integer i=0; i<nsub; ++i) {
        integer node = ls ? ls[i] : i;
        if (mask[node] <= 0) continue;
        auto stack = base;
        *stack++ = node;
        mask[node] = -1;
        integer size = 1;
        while (stack != base) {
          auto k = *(--stack);
          for (auto j=xadj[k]; j<xadj[k+1]; ++j) {
            auto nbr = adjncy[j];
            if (mask[nbr] == 1) {
              *stack++ = nbr;
              mask[nbr] = -1;
              size++;
            }
          }
        }
        C.emplace_back(node, size);
        ncomps++;
      }
      for (integer i=0; i<nsub; ++i) {
        integer node = ls ? ls[i] : i;
        if (mask[node] == -1) mask[node] = 1;
      }
      return ncomps;
    }


    template<typename integer> SeparatorTree<integer>
    gennd(integer n, integer* xadj, integer* adjncy, integer* perm) {
      if (n <= 0) return SeparatorTree<integer>();
      std::vector<Separator<integer>> tree;
      tree.reserve(n);
      integer num = 0;
      std::vector<integer,NoInit<integer>> iwork(4*n);
      auto mask = iwork.data();
      auto xls = mask + n;
      auto ls = mask + 2*n;
      auto base = mask + 3*n;
      std::fill(mask, mask+n, 1);
      struct NDData {
        integer ncomps, nsep, pa;
        bool left;
      };
      std::vector<Comp<integer>> C;
      std::stack<NDData,std::vector<NDData>> ndstack;
      ndstack.emplace
        (NDData{comps(n, n, xadj, adjncy, mask, base, C), 0, -1, false});
      while (!ndstack.empty()) {
        auto s = ndstack.top();
        ndstack.pop();
        if (s.ncomps == 1) {
          auto& c = C.back();
          C.pop_back();
          auto nsep = fndsep(c.root, xadj, adjncy, mask, perm+num, xls, ls);
          if (nsep == c.size || c.size <= 8) { // TODO get from options?
            integer id = tree.size();
            tree.emplace_back(c.size, s.pa, -1, -1);
            if (s.pa != -1) {
              if (s.left) tree[s.pa].lch = id;
              else tree[s.pa].rch = id;
            }
            for (integer i=0; i<c.size; i++)
              perm[num++] = ls[i];
            continue;
          }
          ndstack.emplace
            (NDData{comps(n, c.size, xadj, adjncy, mask, base, C, ls),
               nsep, s.pa, s.left});
          num += nsep;
        } else {
          integer id = tree.size();
          tree.emplace_back(s.nsep, s.pa, -1, -1);
          if (s.pa != -1) {
            if (s.left) tree[s.pa].lch = id;
            else tree[s.pa].rch = id;
          }
          auto Cend = C.end();
          auto Cbeg = Cend - s.ncomps;
          std::sort(Cbeg, Cend, [](auto& a, auto& b) {
            return a.size > b.size; });
          std::vector<Comp<integer>> cc(Cbeg, Cend);
          integer nl = 0, nr = 0, ncl = 0, ncr = 0;
          for (auto ci : cc) {
            if (nl <= nr) {
              *Cbeg = ci;
              Cbeg++;
              nl += ci.size;
              ncl++;
            } else {
              Cend--;
              *Cend = ci;
              nr += ci.size;
              ncr++;
            }
          }
          ndstack.emplace(NDData{ncl, 0, id, true});
          ndstack.emplace(NDData{ncr, 0, id, false});
        }
      }
      std::reverse(perm, perm+n);
      std::reverse(tree.begin(), tree.end());
      auto nbsep = tree.size() - 1;
      for (auto& s : tree) {
        if (s.pa  != -1) s.pa  = nbsep - s.pa;
        if (s.lch != -1) s.lch = nbsep - s.lch;
        if (s.rch != -1) s.rch = nbsep - s.rch;
      }
      for (std::size_t i=1; i<tree.size(); i++)
        tree[i].sep_end = tree[i].sep_end + tree[i-1].sep_end;
      return SeparatorTree<integer>(tree);
    }

    // explicit template instantiation
    template SeparatorTree<int>
    gennd(int neqns, int* xadj, int* adjncy, int* perm);
    template SeparatorTree<long int>
    gennd(long int neqns, long int* xadj, long int* adjncy, long int* perm);
    template SeparatorTree<long long int>
    gennd(long long int neqns, long long int* xadj, long long int* adjncy,
          long long int* perm);

  } // end namespace ordering
} // end namespace strumpack
