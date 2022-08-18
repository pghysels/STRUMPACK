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
C--- SPARSPAK-A (ANSI FORTRAN) RELEASE III --- NAME = GENMMD
C  (C)  UNIVERSITY OF WATERLOO   JANUARY 1984
C***********************************************************************
C***********************************************************************
C****     GENMMD ..... MULTIPLE MINIMUM EXTERNAL DEGREE     ************
C***********************************************************************
C***********************************************************************
C
C     PURPOSE - THIS ROUTINE IMPLEMENTS THE MINIMUM DEGREE
C        ALGORITHM.  IT MAKES USE OF THE IMPLICIT REPRESENTATION
C        OF ELIMINATION GRAPHS BY QUOTIENT GRAPHS, AND THE
C        NOTION OF INDISTINGUISHABLE NODES.  IT ALSO IMPLEMENTS
C        THE MODIFICATIONS BY MULTIPLE ELIMINATION AND MINIMUM
C        EXTERNAL DEGREE.
C        ---------------------------------------------
C        CAUTION - THE ADJACENCY VECTOR ADJNCY WILL BE
C        DESTROYED.
C        ---------------------------------------------
C
C     INPUT PARAMETERS -
C        NEQNS  - NUMBER OF EQUATIONS.
C        (XADJ,ADJNCY) - THE ADJACENCY STRUCTURE.
C        DELTA  - TOLERANCE VALUE FOR MULTIPLE ELIMINATION.
C        MAXINT - MAXIMUM MACHINE REPRESENTABLE (SHORT) INTEGER
C                 (ANY SMALLER ESTIMATE WILL DO) FOR MARKING
C                 NODES.
C
C     OUTPUT PARAMETERS -
C        PERM   - THE MINIMUM DEGREE ORDERING.
C        INVP   - THE INVERSE OF PERM.
C        NOFSUB - AN UPPER BOUND ON THE NUMBER OF NONZERO
C                 SUBSCRIPTS FOR THE COMPRESSED STORAGE SCHEME.
C
C     WORKING PARAMETERS -
C        DHEAD  - VECTOR FOR HEAD OF DEGREE LISTS.
C        INVP   - USED TEMPORARILY FOR DEGREE FORWARD LINK.
C        PERM   - USED TEMPORARILY FOR DEGREE BACKWARD LINK.
C        QSIZE  - VECTOR FOR SIZE OF SUPERNODES.
C        LLIST  - VECTOR FOR TEMPORARY LINKED LISTS.
C        MARKER - A TEMPORARY MARKER VECTOR.
C
C     PROGRAM SUBROUTINES -
C        MMDELM, MMDINT, MMDNUM, MMDUPD.
C
C***********************************************************************
C
      subroutine  genmmd ( neqns, nadj, xadj, adjncy, invp,
     1                     perm, delta, dhead, qsize, llist,
     1                     marker, maxint, nofsub )
c
c***********************************************************************
c

#if MMDIDXSIZE==64
         integer*8  nadj,neqns
         integer*8  adjncy(nadj), dhead(neqns), invp(neqns), 
     &              llist(neqns), marker(neqns), perm(neqns), 
     &              qsize(neqns)
         integer*8  xadj(neqns+1)
         integer*8  delta , ehead , i     , maxint, mdeg  ,
     &              mdlmt , mdnode, nextmd,
     &              nofsub, num, tag
#else
         integer*4  nadj,neqns
         integer*4  adjncy(nadj), dhead(neqns), invp(neqns), 
     &              llist(neqns), marker(neqns), perm(neqns), 
     &              qsize(neqns)
         integer*4  xadj(neqns+1)
         integer*4  delta , ehead , i     , maxint, mdeg  ,
     &              mdlmt , mdnode , nextmd,
     &              nofsub, num, tag
#endif
c
c***********************************************************************
c
c
         if  ( neqns .le. 0 )  return
c        ------------------------------------------------
c        initialization for the minimum degree algorithm.
c        ------------------------------------------------
         nofsub = 0
         call  mmdint ( neqns, xadj, dhead, invp, perm, qsize,
     1                  llist, marker )
c
c        ----------------------------------------------
c        num counts the number of ordered nodes plus 1.
c        ----------------------------------------------
         num = 1
c
c        -----------------------------
c        eliminate all isolated nodes.
c        -----------------------------
         nextmd = dhead(1)
  100    continue
             if  ( nextmd .le. 0 )  go to 200
                 mdnode = nextmd
                 nextmd = invp(mdnode)
                 marker(mdnode) = maxint
                 invp(mdnode) = - num
                 num = num + 1
                 go to 100
c
  200    continue
c        ----------------------------------------
c        search for node of the minimum degree.
c        mdeg is the current minimum degree;
c        tag is used to facilitate marking nodes.
c        ----------------------------------------
         if  ( num .gt. neqns )  go to 1000
         tag = 1
         dhead(1) = 0
         mdeg = 2
  300    continue
             if  ( dhead(mdeg) .gt. 0 )  go to 400
                 mdeg = mdeg + 1
                 go to 300
  400        continue
c            -------------------------------------------------
c            use value of delta to set up mdlmt, which governs
c            when a degree update is to be performed.
c            -------------------------------------------------
             mdlmt = mdeg + delta
             ehead = 0
c
  500        continue
                 mdnode = dhead(mdeg)
                 if  ( mdnode .gt. 0 )  go to 600
                     mdeg = mdeg + 1
                     if  ( mdeg .gt. mdlmt )  go to 900
                         go to 500
  600            continue
c                ----------------------------------------
c                remove mdnode from the degree structure.
c                ----------------------------------------
                 nextmd = invp(mdnode)
                 dhead(mdeg) = nextmd
                 if  ( nextmd .gt. 0 )  perm(nextmd) = - mdeg
                 invp(mdnode) = - num
                 nofsub = nofsub + mdeg + qsize(mdnode) - 2
                 if  ( num+qsize(mdnode) .gt. neqns )  go to 1000
c                ----------------------------------------------
c                eliminate mdnode and perform quotient graph
c                transformation.  reset tag value if necessary.
c                ----------------------------------------------
                 tag = tag + 1
                 if  ( tag .lt. maxint )  go to 800
                     tag = 1
                     do  700  i = 1, neqns
                         if  ( marker(i) .lt. maxint )  marker(i) = 0
  700                continue
  800            continue
                 call  mmdelm ( mdnode, neqns, nadj, xadj, adjncy, 
     &                          dhead, invp, perm, qsize, llist, 
     &                          marker, maxint, tag )
                 num = num + qsize(mdnode)
                 llist(mdnode) = ehead
                 ehead = mdnode
                 if  ( delta .ge. 0 )  go to 500
  900        continue
c            -------------------------------------------
c            update degrees of the nodes involved in the
c            minimum degree nodes elimination.
c            -------------------------------------------
             if  ( num .gt. neqns )  go to 1000
             call  mmdupd ( ehead, neqns, nadj, xadj, adjncy, delta, 
     &                      mdeg, dhead, invp, perm, qsize, llist, 
     &                      marker, maxint, tag )
             go to 300
c
 1000    continue
         call  mmdnum ( neqns, perm, invp, qsize )
         return
c
      end
