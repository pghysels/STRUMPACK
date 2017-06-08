C--- SPARSPAK-A (ANSI FORTRAN) RELEASE III --- NAME = RCM
C  (C)  UNIVERSITY OF WATERLOO   JANUARY 1984
C***************************************************************
C***************************************************************
C*******     RCM ..... REVERSE CUTHILL-MCKEE ORDERING     ******
C***************************************************************
C***************************************************************
C
C     PURPOSE - RCM NUMBERS A CONNECTED COMPONENT SPECIFIED BY
C        MASK AND ROOT, USING THE RCM ALGORITHM.  THE NUMBERING
C        IS TO BE STARTED AT THE NODE ROOT.
C
C     INPUT PARAMETERS -
C        ROOT   - IS THE NODE THAT DEFINES THE CONNECTED
C                 COMPONENT AND IT IS USED AS THE STARTING
C                 NODE FOR THE RCM ORDERING.
C        (XADJ,ADJNCY) - ADJACENCY STRUCTURE PAIR FOR THE GRAPH.
C
C     UPDATED PARAMETER -
C        MASK   - ONLY THOSE NODES WITH NONZERO INPUT MASK
C                 VALUES ARE CONSIDERED BY THE ROUTINE.  THE
C                 NODES NUMBERED BY RCM WILL HAVE THEIR MASK
C                 VALUES SET TO ZERO.
C
C     OUTPUT PARAMETERS -
C        PERM   - WILL CONTAIN THE RCM ORDERING.
C        CCSIZE - IS THE SIZE OF THE CONNECTED COMPONENT THAT
C                 HAS BEEN NUMBERED BY RCM.
C
C     WORKING PARAMETER -
C        DEG    - IS A TEMPORARY VECTOR USED TO HOLD THE DEGREE
C                 OF THE NODES IN THE SECTION GRAPH SPECIFIED BY
C                 MASK AND ROOT.
C
C     PROGRAM SUBROUTINE -
C        DEGREE.
C
C***************************************************************
C
      SUBROUTINE  RCM ( ROOT, XADJ, ADJNCY, MASK, PERM, CCSIZE,
     1                  DEG )
C
C***************************************************************
C
         INTEGER    ADJNCY(1), DEG(1)   , MASK(1)  , PERM(1)
         INTEGER    XADJ(1)
         INTEGER    CCSIZE, FNBR  , I     , J     , JSTOP ,
     1              JSTRT , K     , L     , LBEGIN, LNBR  ,
     1              LPERM , LVLEND, NBR   , NODE  , ROOT
C
C***************************************************************
C
C        -------------------------------------
C        FIND THE DEGREES OF THE NODES IN THE
C        COMPONENT SPECIFIED BY MASK AND ROOT.
C        -------------------------------------
         CALL  DEGREE ( ROOT, XADJ, ADJNCY, MASK, DEG, CCSIZE,
     1                  PERM )
         MASK(ROOT) = 0
         IF  ( CCSIZE .LE. 1 )  RETURN
         LVLEND = 0
         LNBR = 1
  100    CONTINUE
C            --------------------------------------------
C            LBEGIN AND LVLEND POINT TO THE BEGINNING AND
C            THE END OF THE CURRENT LEVEL RESPECTIVELY.
C            --------------------------------------------
             LBEGIN = LVLEND + 1
             LVLEND = LNBR
             DO  600  I = LBEGIN, LVLEND
C                ----------------------------------
C                FOR EACH NODE IN CURRENT LEVEL ...
C                ----------------------------------
                 NODE = PERM(I)
                 JSTRT = XADJ(NODE)
                 JSTOP = XADJ(NODE+1) - 1
C                -----------------------------------------
C                FIND THE UNNUMBERED NEIGHBORS OF NODE.
C                FNBR AND LNBR POINT TO THE FIRST AND LAST
C                UNNUMBERED NEIGHBORS RESPECTIVELY OF THE
C                CURRENT NODE IN PERM.
C                -----------------------------------------
                 FNBR = LNBR + 1
                 DO  200  J = JSTRT, JSTOP
                     NBR = ADJNCY(J)
                     IF  ( MASK(NBR) .EQ. 0 )  GO TO 200
                         LNBR = LNBR + 1
                         MASK(NBR) = 0
                         PERM(LNBR) = NBR
  200            CONTINUE
                 IF  ( FNBR .GE. LNBR )  GO TO 600
C                    ------------------------------------------
C                    SORT THE NEIGHBORS OF NODE IN INCREASING
C                    ORDER BY DEGREE. LINEAR INSERTION IS USED.
C                    ------------------------------------------
                     K = FNBR
  300                CONTINUE
                         L = K
                         K = K + 1
                         NBR = PERM(K)
  400                    CONTINUE
                             IF  ( L .LT. FNBR )  GO TO 500
                             LPERM = PERM(L)
                             IF  ( DEG(LPERM) .LE. DEG(NBR) )  GO TO 500
                             PERM(L+1) = LPERM
                             L = L - 1
                             GO TO 400
  500                    CONTINUE
                         PERM(L+1) = NBR
                         IF  ( K .LT. LNBR )  GO TO 300
  600        CONTINUE
             IF  ( LNBR .GT. LVLEND )  GO TO 100
C
C        ---------------------------------------
C        WE NOW HAVE THE CUTHILL MCKEE ORDERING.
C        REVERSE IT BELOW ...
C        ---------------------------------------
         K = CCSIZE/2
         L = CCSIZE
         DO  700  I = 1, K
             LPERM = PERM(L)
             PERM(L) = PERM(I)
             PERM(I) = LPERM
             L = L - 1
  700    CONTINUE
         RETURN
C
      END
