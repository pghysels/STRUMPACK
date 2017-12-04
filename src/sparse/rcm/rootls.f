C--- SPARSPAK-A (ANSI FORTRAN) RELEASE III --- NAME = ROOTLS
C  (C)  UNIVERSITY OF WATERLOO   JANUARY 1984
C***************************************************************
C***************************************************************
C*********     ROOTLS ..... ROOTED LEVEL STRUCTURE     *********
C***************************************************************
C***************************************************************
C
C     PURPOSE - ROOTLS GENERATES THE LEVEL STRUCTURE ROOTED
C        AT THE INPUT NODE CALLED ROOT.  ONLY THOSE NODES FOR
C        WHICH MASK IS NONZERO WILL BE CONSIDERED.
C
C     INPUT PARAMETERS -
C        ROOT   - THE NODE AT WHICH THE LEVEL STRUCTURE IS TO
C                 BE ROOTED.
C        (XADJ,ADJNCY) - ADJACENCY STRUCTURE PAIR FOR THE
C                 GIVEN GRAPH.
C        MASK   - IS USED TO SPECIFY A SECTION SUBGRAPH.
C                 NODES WITH MASK(I)=0 ARE IGNORED.
C
C     OUTPUT PARAMETERS -
C        NLVL   - IS THE NUMBER OF LEVELS IN THE LEVEL
C                 STRUCTURE.
C        (XLS,LS) - ARRAY PAIR FOR THE ROOTED LEVEL STRUCTURE.
C
C***************************************************************
C
      SUBROUTINE  ROOTLS ( ROOT, XADJ, ADJNCY, MASK, NLVL,
     1                     XLS, LS )
C
C***************************************************************
C
         INTEGER    ADJNCY(1), LS(1)    , MASK(1)  , XLS(1)
         INTEGER    XADJ(1)
         INTEGER    I     , J     , JSTOP , JSTRT , LBEGIN,
     1              CCSIZE, LVLEND, LVSIZE, NBR   , NLVL  ,
     1              NODE  , ROOT
C
C***************************************************************
C
C        ------------------
C        INITIALIZATION ...
C        ------------------
         MASK(ROOT) = 0
         LS(1) = ROOT
         NLVL = 0
         LVLEND = 0
         CCSIZE = 1
C
C        -----------------------------------------------------
C        LBEGIN IS THE POINTER TO THE BEGINNING OF THE CURRENT
C        LEVEL, AND LVLEND POINTS TO THE END OF THIS LEVEL.
C        -----------------------------------------------------
  100    CONTINUE
             LBEGIN = LVLEND + 1
             LVLEND = CCSIZE
             NLVL = NLVL + 1
             XLS(NLVL) = LBEGIN
C            -------------------------------------------------
C            GENERATE THE NEXT LEVEL BY FINDING ALL THE MASKED
C            NEIGHBORS OF NODES IN THE CURRENT LEVEL.
C            -------------------------------------------------
             DO  300  I = LBEGIN, LVLEND
                 NODE = LS(I)
                 JSTRT = XADJ(NODE)
                 JSTOP = XADJ(NODE+1) - 1
                 IF  ( JSTOP .LT. JSTRT )  GO TO 300
                     DO  200  J = JSTRT, JSTOP
                         NBR = ADJNCY(J)
                         IF  ( MASK(NBR) .NE. 0 ) THEN
                             CCSIZE = CCSIZE + 1
                             LS(CCSIZE) = NBR
                             MASK(NBR) = 0
                         END IF
  200                CONTINUE
  300        CONTINUE
C            ------------------------------------------
C            COMPUTE THE CURRENT LEVEL WIDTH.
C            IF IT IS NONZERO, GENERATE THE NEXT LEVEL.
C            ------------------------------------------
             LVSIZE = CCSIZE - LVLEND
             IF  ( LVSIZE .GT. 0 )  GO TO 100
C
C        -------------------------------------------------------
C        RESET MASK TO ONE FOR THE NODES IN THE LEVEL STRUCTURE.
C        -------------------------------------------------------
         XLS(NLVL+1) = LVLEND + 1
         DO  400  I = 1, CCSIZE
             NODE = LS(I)
             MASK(NODE) = 1
  400    CONTINUE
         RETURN
C
      END
