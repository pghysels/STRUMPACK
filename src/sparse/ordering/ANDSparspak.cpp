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

    template<typename integer> std::vector<Comp<integer>>
    comps(integer n, integer nsub, integer* xadj, integer* adjncy,
          integer* mask, integer* base,
          integer* ls=nullptr) {
      std::vector<Comp<integer>> C;
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
      }
      for (integer i=0; i<nsub; ++i) {
        integer node = ls ? ls[i] : i;
        if (mask[node] == -1) mask[node] = 1;
      }
      return C;
    }

    template<typename integer> void
    recnd(std::vector<Comp<integer>>& C,
          std::vector<Separator<integer>>& tree,
          integer n, integer* xadj, integer* adjncy,
          integer& num, integer* perm, integer* mask,
          integer* base, integer* xls, integer* ls, integer* work) {
      if (C.size() == 1) {
        auto& c = C[0];
        auto nsep = fndsep(c.root, xadj, adjncy, mask, work, xls, ls);
        if (nsep == c.size || c.size <= 8) { // TODO get from options?
          tree.emplace_back(c.size, -1, -1, -1);
          for (integer i=0; i<c.size; i++)
            perm[num + i] = ls[i];
          num += c.size;
          return;
        }
        auto nC = comps(n, c.size, xadj, adjncy, mask, base, ls);
        recnd(nC, tree, n, xadj, adjncy, num, perm,
              mask, base, xls, ls, work+nsep);
        tree.back().sep_end = nsep;
        for (integer i=0; i<nsep; i++)
          perm[num + i] = work[i];
        num += nsep;
      } else {
        std::sort(C.begin(), C.end(),
                  [](auto& a, auto& b) { return a.size < b.size; });
        std::vector<Comp<integer>> Cl, Cr;
        integer nl = 0, nr = 0;
        for (auto& ci : C) {
          if (nl <= nr) {
            Cl.push_back(ci);
            nl += ci.size;
          } else {
            Cr.push_back(ci);
            nr += ci.size;
          }
        }
        recnd(Cl, tree, n, xadj, adjncy, num, perm,
              mask, base, xls, ls, work);
        auto lid = tree.size() - 1;
        recnd(Cr, tree, n, xadj, adjncy, num, perm,
              mask, base, xls, ls, work);
        auto rid = tree.size() - 1;
        tree.emplace_back(0, -1, lid, rid);
        tree[lid].pa = rid + 1;
        tree[rid].pa = rid + 1;
      }
    }

    template<typename integer>
    std::unique_ptr<SeparatorTree<integer>>
    gennd(integer n, integer* xadj, integer* adjncy, integer* perm) {
      if (n <= 0) return nullptr;
      std::vector<integer,NoInit<integer>> iwork(5*n);
      auto mask = iwork.data();
      auto xls = mask + n;
      auto ls = mask + 2*n;
      auto base = mask + 3*n;
      auto work = mask + 4*n;
      std::fill(mask, mask+n, 1);

#if 1
      auto C = comps(n, n, xadj, adjncy, mask, base);
      std::vector<Separator<integer>> tree;
      tree.reserve(n);
      integer num = 0;
      recnd(C, tree, n, xadj, adjncy, num, perm, mask, base, xls, ls, work);
      for (std::size_t i=1; i<tree.size(); i++)
        tree[i].sep_end = tree[i].sep_end + tree[i-1].sep_end;
      return std::unique_ptr<SeparatorTree<integer>>
        (new SeparatorTree<integer>(tree));

#else
      integer lvl = 0, num = 0;
      std::fill(mask, mask+n, 1);
      for (integer i=0, num=0; i<n && num<n; ++i) {
        do {
          if (mask[i] == 0) break;
          /* FIND A SEPARATOR AND NUMBER THE NODES NEXT. */
          auto nsep = fndsep(i, xadj, adjncy, mask, perm+num, xls, ls);
          num += nsep;
        } while (num < n);
      }
      /* SEPARATORS FOUND FIRST SHOULD BE ORDERED LAST */
      std::reverse(perm, perm+n);
      return nullptr;
#endif
    }

    // explicit template instantiation
    template std::unique_ptr<SeparatorTree<int>>
    gennd(int neqns, int* xadj, int* adjncy, int* perm);

    template std::unique_ptr<SeparatorTree<long int>>
    gennd(long int neqns, long int* xadj, long int* adjncy, long int* perm);

    template std::unique_ptr<SeparatorTree<long long int>>
    gennd(long long int neqns, long long int* xadj, long long int* adjncy, long long int* perm);

  } // end namespace ordering
} // end namespace strumpack
