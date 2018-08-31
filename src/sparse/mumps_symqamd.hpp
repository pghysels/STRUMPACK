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
#ifndef STRUMPACK_MUMPS_SYMQAMD_HPP
#define STRUMPACK_MUMPS_SYMQAMD_HPP

#include "StrumpackFortranCInterface.h"

#define MUMPS_SYMQAMD_FC FC_GLOBAL_(mumps_symqamd, MUMPS_SYMQAMD)

extern "C" void MUMPS_SYMQAMD_FC
(int* TRESH, int* NDENSE, int* N, int* IWLEN, int* PE, int* PFREE,
 int* LEN, int* IW, int* NV, int* ELEN, int* LAST, int* NCMPA,
 int* DEGREE, int* HEAD, int* NEXT, int* W, int* PERM,
 int* LISTVAR_SCHUR, int* SIZE_SCHUR, int* AGG6);

namespace strumpack {

  template<typename scalar_t,typename integer_t>
  std::unique_ptr<SeparatorTree<integer_t>> aggressive_amalgamation
  (const CompressedSparseMatrix<scalar_t,integer_t>& A,
   std::vector<integer_t>& perm, std::vector<integer_t>& iperm,
   const SPOptions<scalar_t>& opts) {
    auto ptr = A.ptr();
    auto ind = A.ind();
    int N = A.size();
    int NNZ = A.nnz();

    int TRESH = 0;              /* <= 0 Recommended value. Automatic setting will be done. */
    int* NDENSE = new int[N];   /* [N] used internally */
    int IWLEN = NNZ;            /* length of workspace IW */
    int* PE = new int[N];       /* [N] On input PE(i) contains the pointers in IW to
                                 * (the column indices of) row i of the matrix.  On
                                 * output it contains the tree:
                                 * - if I is a principal variable (NV(I) >0) then -pe(I) is the
                                 *   principal variable of the father, or 0 if I is a root node.
                                 * - if I is a secondary variable (NV(I)=0) then -pe(I) is the
                                 *   principal variable of the node it belongs to.      */
    int PFREE = 0;              /* on input, the matrix is stored in IW(1:PFREE-1) */
    int* LEN = new int[N];      /* [N], on input LEN(i) holds the number of entries in
                                 * row i of the matrix, excluding the diagonal */
    int* IW = new int[IWLEN];   /* [IWLEN] Column indices */
    int* NV = new int[N];       /* - if i is a principal variable, NV(i) is the size of the front
                                 *    in the multifrontal terminology. ENTIRE FRONT, [F11 F12; F21 F22]
                                 * - if i is a secondary variable, NV(i)=0 */
    int* ELEN = new int[N];     /* [N], on output contains the inverse permutation */
    int* LAST = new int[N];     /* [N], on output last(1..n) holds the permutation */
    int NCMPA;                  /* (out) number of compressions */
    int* DEGREE = new int[N];   /* [N] internal */
    int* HEAD = new int[N];     /* [N] internal */
    int* NEXT = new int[N];     /* [N] internal */
    int* W = new int[N];        /* [N] internal */
    int* PERM = new int[N];     /* [N], on input, the permutation */
    int SIZE_SCHUR = 0;         /* > 0 means that the last SIZE_SCHUR variable
                                 *  in the order (such that PERM(I) > N-SIZE_SCHUR) are
                                 *  part of the schur decompositon and should remain
                                 *   ordered last and amalgamated at the root of the
                                 *  elimination tree. */
    int* LISTVAR_SCHUR = new int[std::max(1, SIZE_SCHUR)];  /* [max(1,SIZE_SCHUR)] */
    int AGG6 = opts.use_agg_amalg();    /* bool perform aggressive absortion */

    for (int r=0; r<N; r++) {
      PERM[r] = perm[r] + 1;           /* permutation from integer_t to int, from C to Fortran */
      LEN[r] = ptr[r+1] - ptr[r] - 1;  /* nonzeros in rows r, excluding diagonal */
      PE[r] = PFREE + 1;               /* row pointer */
      for (int j=ptr[r]; j<ptr[r+1]; j++) {
        if (ind[j] != r) IW[PFREE++] = ind[j]+1;
      }
    }
    PFREE++;
    MUMPS_SYMQAMD_FC
      (&TRESH, NDENSE, &N, &IWLEN, PE, &PFREE, LEN, IW, NV, ELEN,
       LAST, &NCMPA, DEGREE, HEAD, NEXT, W, PERM,
       LISTVAR_SCHUR, &SIZE_SCHUR, &AGG6);
    for (int i=0; i<N; i++) {
      iperm[i] = LAST[i] - 1;
      perm[i] = ELEN[i] - 1;
    }

    // permute NV and PE using PERM from MUMPS_SYMQAMD
    int* tmp = new int[N];
    for (int i=0; i<N; i++)
      if (PE[i] != 0) tmp[perm[i]] = perm[-PE[i]-1];
      else tmp[perm[i]] = N;
    for (int i=0; i<N; i++) PE[i] = tmp[i];
    for (int i=0; i<N; i++) tmp[perm[i]] = NV[i];
    for (int i=0; i<N; i++) NV[i] = tmp[i];

    // post-order the assembly tree
    auto po = etree_postorder(N, PE);
    for (int i=0; i<N; i++) tmp[po[i]] = po[PE[i]];
    for (int i=0; i<N; i++) PE[i] = tmp[i];
    for (int i=0; i<N; i++) tmp[po[i]] = NV[i];
    for (int i=0; i<N; i++) NV[i] = tmp[i];
    std::replace(PE, PE+N, N, -1);

    // combine perm with postordering of the tree
    for (int i=0; i<N; i++) tmp[i] = po[perm[i]];
    for (int i=0; i<N; i++) perm[i] = tmp[i];
    for (int i=0; i<N; i++) iperm[perm[i]] = i;
    delete[] tmp;

    int* fid = DEGREE;
    int fronts = 0;
    for (int i=0; i<N; i++) if (NV[i]) fid[i] = fronts++;

    // count will have the number of children (principal) for each supernode
    std::vector<int> count(fronts);
    int roots = 0;
    for (int i=0; i<N; i++)
      if (NV[i]) {             // principal
        if (PE[i] != -1)       // not the root
          count[fid[PE[i]]]++; // count i as a child of PE[i]
        else roots++;          // root
      }
    int dummies=0, merges=0;
    for (int f=0; f<fronts; f++) {
      if (count[f]  > 2) dummies += count[f] - 2; // if there are more than 2 children, add dummies
      if (count[f] == 1) merges++;                // if only 1 child, then merge
    }
    for (int i=0; i<N; i++)
      if (NV[i]) { // principal
        int p = PE[i];
        // while I'm not the root and my parent has only 1 child
        while (p != -1 && count[fid[p]] == 1) {
          NV[p] = 0;      // make parent secondary
          PE[i] = PE[p];  // set parent of i its former grandparent
          PE[p] = i;      // set primary of parent to i
          p = PE[i];
        }
      }
    fronts -= merges;
    for (int i=0, f=0; i<N; i++) if (NV[i]) fid[i] = f++;

    auto sep_tree = std::unique_ptr<SeparatorTree<integer_t>>
      (new SeparatorTree<integer_t>(fronts + dummies + roots-1));
    for (int f=0; f<sep_tree->separators(); f++)
      sep_tree->lch(f) = sep_tree->rch(f) = -1;
    for (int f=0; f<=sep_tree->separators(); f++)
      sep_tree->sizes(f) = 0;

    for (int i=0; i<N; i++) {
      if (NV[i]) {
        sep_tree->pa(fid[i]) = (PE[i] == -1) ? -1 : fid[PE[i]];
        sep_tree->sizes(fid[i]+1)++;
      } else { // secondary
        int p = PE[i];
        while (NV[p] == 0) p = PE[p]; // find the principal node
        sep_tree->sizes(fid[p]+1)++;
      }
    }
    if (roots > 1) {
      // add a single empty root, with the other roots as children
      std::replace(sep_tree->pa(), sep_tree->pa()+fronts,
                   integer_t(-1), integer_t(fronts));
      sep_tree->pa(fronts) = -1;
      fronts++;
    }
    std::fill(count.begin(), count.end(), 0);
    for (int f=0, ft=fronts; f<fronts; f++) {
      auto p = sep_tree->pa(f);
      if (p != -1) {
        count[p]++;
        switch (count[p]) {
        case 1: sep_tree->lch(p) = f; break;
        case 2: sep_tree->rch(p) = f; break;
        default:
          // create a dummy holding the 2 existing children of p, with
          // parent p
          sep_tree->pa(ft) = p;
          sep_tree->lch(ft) = sep_tree->lch(p);
          sep_tree->rch(ft) = sep_tree->rch(p);
          sep_tree->sizes(ft+1) = 0;
          sep_tree->pa(sep_tree->lch(p)) = ft;
          sep_tree->pa(sep_tree->rch(p)) = ft;
          // make the dummy the left child of p, make the current node
          // the right child of p
          sep_tree->lch(p) = ft;
          sep_tree->rch(p) = f;
          ft++;
          break;
        }
      }
    }
    for (integer_t f=0; f<sep_tree->separators(); f++)
      sep_tree->sizes(f+1) = sep_tree->sizes(f) + sep_tree->sizes(f+1);
    assert(sep_tree->sizes(sep_tree->separators()) == N);
    delete[] NDENSE;
    delete[] PE;
    delete[] LEN;
    delete[] IW;
    delete[] NV;
    delete[] ELEN;
    delete[] LAST;
    delete[] DEGREE;
    delete[] HEAD;
    delete[] NEXT;
    delete[] W;
    delete[] PERM;
    delete[] LISTVAR_SCHUR;
    return sep_tree;
  }

} // end namespace strumpack

#endif // STRUMPACK_MUMPS_SYMQAMD_HPP

