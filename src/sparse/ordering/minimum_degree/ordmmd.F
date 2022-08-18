C***********************************************************************
C***********************************************************************
C
C   Version:        0.4
C   Last modified:  December 27, 1994
C   Authors:        Esmond G. Ng and Barry W. Peyton
C
C   Mathematical Sciences Section, Oak Ridge National Laboratory
C
C***********************************************************************
C***********************************************************************
C****     ORDMMD ..... MULTIPLE MINIMUM EXTERNAL DEGREE     ************
C***********************************************************************
C***********************************************************************
C
C     PURPOSE - THIS ROUTINE CALLS LIU'S MULTIPLE MINIMUM DEGREE
C               ROUTINE.
C
C     INPUT PARAMETERS -
C        NEQNS  - NUMBER OF EQUATIONS.
C        (XADJ,ADJNCY) - THE ADJACENCY STRUCTURE.
C        IWSIZ  - SIZE OF INTEGER WORKING STORAGE.
C
C     OUTPUT PARAMETERS -
C        PERM   - THE MINIMUM DEGREE ORDERING.
C        INVP   - THE INVERSE OF PERM.
C        NOFSUB - AN UPPER BOUND ON THE NUMBER OF NONZERO
C                 SUBSCRIPTS FOR THE COMPRESSED STORAGE SCHEME.
C        IFLAG  - ERROR FLAG.
C                   0: SUCCESSFUL ORDERING
C                  -1: INSUFFICIENT WORKING STORAGE
C                      [IWORK(*)].
C
C     WORKING PARAMETERS -
C        IWORK  - INTEGER WORKSPACE OF LENGTH 4*NEQNS.
C
C***********************************************************************
C
      subroutine ordmmd  (  neqns , nadj  , xadj  , adjncy, invp  ,
     1                      perm  , iwsiz , iwork , nofsub, iflag	)
c
c***********************************************************************
c
#if MMDIDXSIZE == 64
         integer*8  iwsiz
         integer*8  adjncy(1), invp(1)  , iwork(iwsiz) , perm(1)
         integer*8  xadj(1)
         integer*8  delta , iflag , maxint, nadj  , 
     &              neqns , nofsub
#else
         integer*4  iwsiz
         integer*4  adjncy(1), invp(1)  , iwork(iwsiz) , perm(1)
         integer*4  xadj(1)
         integer*4  delta , iflag , maxint, nadj  , 
     &              neqns , nofsub
#endif
c
c*********************************************************************
c
        iflag = 0
        if  ( iwsiz .lt. 4*neqns )  then
            iflag = -1
            return
        endif
c
c       delta  - tolerance value for multiple elimination.
c       maxint - maximum machine representable (short) integer
c                (any smaller estimate will do) for marking
c                nodes.
c
        delta  = 0
        maxint = 2 000 000 000
        call genmmd  (  neqns , nadj  , xadj  , adjncy, invp  ,
     1                  perm  , delta , 
     1                  iwork(1)              ,
     1                  iwork(neqns+1)        ,
     1                  iwork(2*neqns+1)      ,
     1                  iwork(3*neqns+1)      ,
     1                  maxint, nofsub          )
         return
c
      end
