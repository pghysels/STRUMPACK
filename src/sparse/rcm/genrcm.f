C--- SPARSPAK-A (ANSI FORTRAN) RELEASE III --- NAME = GENRCM
C  (C)  UNIVERSITY OF WATERLOO   JANUARY 1984
C***************************************************************
C***************************************************************
C******     GENRCM ..... GENERAL REVERSE CUTHILL MCKEE     *****
C***************************************************************
C***************************************************************
C
C     PURPOSE - GENRCM FINDS THE REVERSE CUTHILL-MCKEE ORDERING
C        FOR A GENERAL GRAPH.  FOR EACH CONNECTED COMPONENT IN
C        THE GRAPH, GENRCM OBTAINS THE ORDERING BY CALLING THE
C        SUBROUTINE RCM.
C
C     INPUT PARAMETERS -
C        NEQNS  - NUMBER OF EQUATIONS
C        (XADJ,ADJNCY) - ARRAY PAIR CONTAINING THE ADJACENCY
C                 STRUCTURE OF THE GRAPH OF THE MATRIX.
C
C     OUTPUT PARAMETER -
C        PERM   - VECTOR THAT CONTAINS THE RCM ORDERING.
C
C     WORKING PARAMETERS -
C        MASK   - IS USED TO MARK VARIABLES THAT HAVE BEEN
C                 NUMBERED DURING THE ORDERING PROCESS.  IT
C                 IS INITIALIZED TO 1, AND SET TO ZERO AS
C                 EACH NODE IS NUMBERED.
C        XLS    - THE INDEX VECTOR FOR A LEVEL STRUCTURE.  THE
C                 LEVEL STRUCTURE IS STORED IN THE CURRENTLY
C                 UNUSED SPACES IN THE PERMUTATION VECTOR PERM.
C
C     PROGRAM SUBROUTINES -
C        FNROOT, RCM   .
C
C***************************************************************
C
      SUBROUTINE  GENRCM ( NEQNS, XADJ, ADJNCY, PERM, MASK,
     1                     XLS )
C
C***************************************************************
C
         INTEGER    ADJNCY(1), MASK(1)  , PERM(1)  , XLS(1)
         INTEGER    XADJ(1)
         INTEGER    CCSIZE, I     , NEQNS , NLVL  , NUM   ,
     1              ROOT
C
C***************************************************************
C
         IF  ( NEQNS .LE. 0 )  RETURN
C
         DO  100  I = 1, NEQNS
             MASK(I) = 1
  100    CONTINUE
         NUM = 1
C
         DO  200  I = 1, NEQNS
C            ---------------------------------------
C            FOR EACH MASKED CONNECTED COMPONENT ...
C            ---------------------------------------
             IF  ( MASK(I) .EQ. 0 )  GO TO 200
                 ROOT = I
C                -----------------------------------------
C                FIRST FIND A PSEUDO-PERIPHERAL NODE ROOT.
C                NOTE THAT THE LEVEL STRUCTURE FOUND BY
C                FNROOT IS STORED STARTING AT PERM(NUM).
C                THEN RCM IS CALLED TO ORDER THE COMPONENT
C                USING ROOT AS THE STARTING NODE.
C                -----------------------------------------
                 CALL  FNROOT ( ROOT, XADJ, ADJNCY, MASK,
     1                          NLVL, XLS, PERM(NUM) )
                 CALL  RCM ( ROOT, XADJ, ADJNCY, MASK,
     1                       PERM(NUM), CCSIZE, XLS )
                 NUM = NUM + CCSIZE
                 IF  ( NUM .GT. NEQNS )  RETURN
  200    CONTINUE
         RETURN
C
      END




      subroutine perm_inverse ( neqn, perm, invprm ) 

c*********************************************************************72
c
cc PERM_INVERSE produces the inverse permutation.
c
c  Licensing:
c
c    This code is distributed under the GNU LGPL license.
c
c  Modified:
c
c    01 January 2009
c
c  Author:
c
c    John Burkardt
c
c  Parameters:
c
c    Input, integer NEQN, the number of equations.
c
c    Input, integer PERM(NEQN), the reordering of the variables and equations.
c
c    Output, integer INVPRM(NEQN), the inverse ordering, with the property 
c    that INVPRM(PERM(K))=K.
c
      integer neqn

      integer i
      integer invprm(neqn)
      integer k
      integer perm(neqn)

      do i = 1, neqn
        k = perm(i)
        invprm(k) = i
      end do

      return
      end


