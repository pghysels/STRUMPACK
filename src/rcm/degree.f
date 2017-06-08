C--- SPARSPAK-A (ANSI FORTRAN) RELEASE III --- NAME = DEGREE
C  (C)  UNIVERSITY OF WATERLOO   JANUARY 1984
C***************************************************************
C***************************************************************
C*******     DEGREE ..... DEGREE IN MASKED COMPONENT     *******
C***************************************************************
C***************************************************************
C
C     PURPOSE - THIS ROUTINE COMPUTES THE DEGREES OF THE NODES
C        IN THE CONNECTED COMPONENT SPECIFIED BY MASK AND ROOT.
C        NODES FOR WHICH MASK IS ZERO ARE IGNORED.
C
C     INPUT PARAMETERS -
C        ROOT   - IS THE INPUT NODE THAT DEFINES THE COMPONENT.
C        (XADJ,ADJNCY) - ADJACENCY STRUCTURE PAIR.
C        MASK   - SPECIFIES A SECTION SUBGRAPH.
C
C     OUTPUT PARAMETERS -
C        DEG    - ARRAY CONTAINING THE DEGREES OF THE NODES IN
C                 THE COMPONENT.
C        CCSIZE - SIZE OF THE COMPONENT SPECIFED BY MASK AND
C                 ROOT.
C
C     WORKING PARAMETER -
C        LS     - A TEMPORARY VECTOR USED TO STORE THE NODES OF
C                 THE COMPONENT LEVEL BY LEVEL.
C
C***************************************************************
C
      SUBROUTINE  DEGREE ( ROOT, XADJ, ADJNCY, MASK, DEG,
     1                     CCSIZE, LS )
C
C***************************************************************
C
         INTEGER    ADJNCY(1), DEG(1)   , LS(1)    , MASK(1)
         INTEGER    XADJ(1)
         INTEGER    CCSIZE, I     , IDEG  , J     , JSTOP ,
     1              JSTRT , LBEGIN, LVLEND, LVSIZE, NBR   ,
     1              NODE  , ROOT
C
C***************************************************************
C
C        -------------------------------------------------
C        INITIALIZATION ...
C        THE ARRAY XADJ IS USED AS A TEMPORARY MARKER TO
C        INDICATE WHICH NODES HAVE BEEN CONSIDERED SO FAR.
C        -------------------------------------------------
         LS(1) = ROOT
         XADJ(ROOT) = - XADJ(ROOT)
         LVLEND = 0
         CCSIZE = 1
C
  100    CONTINUE
C            -----------------------------------------------------
C            LBEGIN IS THE POINTER TO THE BEGINNING OF THE CURRENT
C            LEVEL, AND LVLEND POINTS TO THE END OF THIS LEVEL.
C            -----------------------------------------------------
             LBEGIN = LVLEND + 1
             LVLEND = CCSIZE
C            -----------------------------------------------
C            FIND THE DEGREES OF NODES IN THE CURRENT LEVEL,
C            AND AT THE SAME TIME, GENERATE THE NEXT LEVEL.
C            -----------------------------------------------
             DO  400  I = LBEGIN, LVLEND
                 NODE = LS(I)
                 JSTRT = - XADJ(NODE)
                 JSTOP = IABS(XADJ(NODE+1)) - 1
                 IDEG = 0
                 IF  ( JSTOP .LT. JSTRT )  GO TO 300
                     DO  200  J = JSTRT, JSTOP
                         NBR = ADJNCY(J)
                         IF  ( MASK(NBR) .EQ. 0 )  GO TO 200
                             IDEG = IDEG + 1
                             IF  ( XADJ(NBR) .LT. 0 )  GO TO 200
                                 XADJ(NBR) = - XADJ(NBR)
                                 CCSIZE = CCSIZE + 1
                                 LS(CCSIZE) = NBR
  200                CONTINUE
  300            CONTINUE
                 DEG(NODE) = IDEG
  400        CONTINUE
C            -----------------------------------------
C            COMPUTE THE CURRENT LEVEL WIDTH.
C            IF IT IS NONZERO, GENERATE ANOTHER LEVEL.
C            -----------------------------------------
             LVSIZE = CCSIZE - LVLEND
             IF  ( LVSIZE .GT. 0 )  GO TO 100
C
C        ------------------------------------------
C        RESET XADJ TO ITS CORRECT SIGN AND RETURN.
C        ------------------------------------------
         DO  500  I = 1, CCSIZE
             NODE = LS(I)
             XADJ(NODE) = - XADJ(NODE)
  500    CONTINUE
         RETURN
C
      END
