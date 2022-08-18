C***********************************************************************
C***********************************************************************
C
C   Version:        0.4
C   Last modified:  December 27, 1994
C   Authors:        Joseph W.H. Liu
C
C   Mathematical Sciences Section, Oak Ridge National Laboratory
C
C***********************************************************************
C***********************************************************************
C--- SPARSPAK-A (ANSI FORTRAN) RELEASE III --- NAME = MMDNUM
C  (C)  UNIVERSITY OF WATERLOO   JANUARY 1984
C***********************************************************************
C***********************************************************************
C*****     MMDNUM ..... MULTI MINIMUM DEGREE NUMBERING     *************
C***********************************************************************
C***********************************************************************
C
C     PURPOSE - THIS ROUTINE PERFORMS THE FINAL STEP IN
C        PRODUCING THE PERMUTATION AND INVERSE PERMUTATION
C        VECTORS IN THE MULTIPLE ELIMINATION VERSION OF THE
C        MINIMUM DEGREE ORDERING ALGORITHM.
C
C     INPUT PARAMETERS -
C        NEQNS  - NUMBER OF EQUATIONS.
C        QSIZE  - SIZE OF SUPERNODES AT ELIMINATION.
C
C     UPDATED PARAMETERS -
C        INVP   - INVERSE PERMUTATION VECTOR.  ON INPUT,
C                 IF QSIZE(NODE)=0, THEN NODE HAS BEEN MERGED
C                 INTO THE NODE -INVP(NODE); OTHERWISE,
C                 -INVP(NODE) IS ITS INVERSE LABELLING.
C
C     OUTPUT PARAMETERS -
C        PERM   - THE PERMUTATION VECTOR.
C
C***********************************************************************
C
      subroutine  mmdnum ( neqns, perm, invp, qsize )
c
c***********************************************************************
c
#if MMDIDXSIZE == 64
         integer*8  neqns
         integer*8    invp(neqns), perm(neqns), qsize(neqns)
         integer*8    father, nextf , node  , nqsize,
     1              num   , root
#else
         integer*4  neqns
         integer*4    invp(neqns), perm(neqns), qsize(neqns)
         integer*4    father, nextf , node  , nqsize,
     1              num   , root
#endif
c
c***********************************************************************
c
         do  100  node = 1, neqns
             nqsize = qsize(node)
             if  ( nqsize .le. 0 )  perm(node) = invp(node)
             if  ( nqsize .gt. 0 )  perm(node) = - invp(node)
  100    continue
c        ------------------------------------------------------
c        for each node which has been merged, do the following.
c        ------------------------------------------------------
         do  500  node = 1, neqns
             if  ( perm(node) .gt. 0 )  go to 500
c                -----------------------------------------
c                trace the merged tree until one which has
c                not been merged, call it root.
c                -----------------------------------------
                 father = node
  200            continue
                     if  ( perm(father) .gt. 0 )  go to 300
                         father = - perm(father)
                         go to 200
  300            continue
c                -----------------------
c                number node after root.
c                -----------------------
                 root = father
                 num = perm(root) + 1
                 invp(node) = - num
                 perm(root) = num
c                ------------------------
c                shorten the merged tree.
c                ------------------------
                 father = node
  400            continue
                     nextf = - perm(father)
                     if  ( nextf .le. 0 )  go to 500
                         perm(father) = - root
                         father = nextf
                         go to 400
  500    continue
c        ----------------------
c        ready to compute perm.
c        ----------------------
         do  600  node = 1, neqns
             num = - invp(node)
             invp(node) = num
             perm(num) = node
  600    continue
         return
c
      end
