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
C--- SPARSPAK-A (ANSI FORTRAN) RELEASE III --- NAME = MMDINT
C  (C)  UNIVERSITY OF WATERLOO   JANUARY 1984
C***********************************************************************
C***********************************************************************
C***     MMDINT ..... MULT MINIMUM DEGREE INITIALIZATION     ***********
C***********************************************************************
C***********************************************************************
C
C     PURPOSE - THIS ROUTINE PERFORMS INITIALIZATION FOR THE
C        MULTIPLE ELIMINATION VERSION OF THE MINIMUM DEGREE
C        ALGORITHM.
C
C     INPUT PARAMETERS -
C        NEQNS  - NUMBER OF EQUATIONS.
C        XADJ - POINTER VECTOR FOR ADJACENCY STRUCTURE.
C
C     OUTPUT PARAMETERS -
C        (DHEAD,DFORW,DBAKW) - DEGREE DOUBLY LINKED STRUCTURE.
C        QSIZE  - SIZE OF SUPERNODE (INITIALIZED TO ONE).
C        LLIST  - LINKED LIST.
C        MARKER - MARKER VECTOR.
C
C***********************************************************************
C
      subroutine  mmdint ( neqns, xadj, dhead, dforw, dbakw,
     1                     qsize, llist, marker )
c
c***********************************************************************
c
#if MMDIDXSIZE == 64
         integer*8  neqns
         integer*8    dbakw(neqns) , dforw(neqns) , dhead(neqns),
     &              llist(neqns) , marker(neqns), qsize(neqns)
         integer*8    xadj(neqns+1)
         integer*8    fnode , ndeg  , node
#else
         integer*4  neqns
         integer*4    dbakw(neqns) , dforw(neqns) , dhead(neqns),
     &              llist(neqns) , marker(neqns), qsize(neqns)
         integer*4    xadj(neqns+1)
         integer*4    fnode , ndeg  , node
#endif
c
c***********************************************************************
c
         do  100  node = 1, neqns
             dhead(node) = 0
             qsize(node) = 1
             marker(node) = 0
             llist(node) = 0
  100    continue
c        ------------------------------------------
c        initialize the degree doubly linked lists.
c        ------------------------------------------
         do  200  node = 1, neqns
             ndeg = xadj(node+1) - xadj(node) + 1
             fnode = dhead(ndeg)
             dforw(node) = fnode
             dhead(ndeg) = node
             if  ( fnode .gt. 0 )  dbakw(fnode) = node
             dbakw(node) = - ndeg
  200    continue
         return
c
      end
