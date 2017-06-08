      SUBROUTINE CGEQP3TOL( M, N, A, LDA, JPVT, TAU, WORK, LWORK, RWORK,
     $     INFO, RANK, RTOL, ATOL, DEPTH )
*
*  -- LAPACK computational routine (version 3.4.2) --
*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     September 2012
*
*     .. Scalar Arguments ..
      INTEGER            INFO, LDA, LWORK, M, N, RANK
*     ..
*     .. Array Arguments ..
      INTEGER            JPVT( * )
      REAL               RWORK( * ), RTOL, ATOL
      COMPLEX            A( LDA, * ), TAU( * ), WORK( * )
*     ..
*
*  =====================================================================
*
*     .. Parameters ..
      INTEGER            INB, INBMIN, IXOVER
      PARAMETER          ( INB = 1, INBMIN = 2, IXOVER = 3 )
*     ..
*     .. Local Scalars ..
      LOGICAL            LQUERY
      INTEGER            FJB, IWS, J, JB, LWKOPT, MINMN, MINWS, NA, NB,
     $                   NBMIN, NFXD, NX, SM, SMINMN, SN, TOPBMN,
     $                   C
*     ..
*     .. External Subroutines ..
      EXTERNAL           CGEQRF, CLAQP2MOD, CLAQPSMOD, 
     $     CSWAP, CUNMQR, XERBLA
*     ..
*     .. External Functions ..
      INTEGER            ILAENV
      REAL               SCNRM2
      EXTERNAL           ILAENV, SCNRM2
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          INT, MAX, MIN
*     ..
*     .. Executable Statements ..
*
*     Test input arguments
*  ====================
*
      INFO = 0
      LQUERY = ( LWORK.EQ.-1 )
      IF( M.LT.0 ) THEN
         INFO = -1
      ELSE IF( N.LT.0 ) THEN
         INFO = -2
      ELSE IF( LDA.LT.MAX( 1, M ) ) THEN
         INFO = -4
      END IF
*

      RANK=0

      IF( INFO.EQ.0 ) THEN
         MINMN = MIN( M, N )
         IF( MINMN.EQ.0 ) THEN
            IWS = 1
            LWKOPT = 1
         ELSE
            IWS = N + 1
            NB = ILAENV( INB, 'CGEQRF', ' ', M, N, -1, -1 )
            LWKOPT = ( N + 1 )*NB
         END IF
         WORK( 1 ) = LWKOPT
*
         IF( ( LWORK.LT.IWS ) .AND. .NOT.LQUERY ) THEN
            INFO = -8
         END IF
      END IF
*
      IF( INFO.NE.0 ) THEN
         CALL XERBLA( 'CGEQP3', -INFO )
         RETURN
      ELSE IF( LQUERY ) THEN
         RETURN
      END IF
*
*     Quick return if possible.
*
      IF( MINMN.EQ.0 ) THEN
         RETURN
      END IF
*
*     Move initial columns up front.
*
      NFXD = 1
      DO 10 J = 1, N
         IF( JPVT( J ).NE.0 ) THEN
            IF( J.NE.NFXD ) THEN
               CALL CSWAP( M, A( 1, J ), 1, A( 1, NFXD ), 1 )
               JPVT( J ) = JPVT( NFXD )
               JPVT( NFXD ) = J
            ELSE
               JPVT( J ) = J
            END IF
            NFXD = NFXD + 1
         ELSE
            JPVT( J ) = J
         END IF
   10 CONTINUE
      NFXD = NFXD - 1
*
*     Factorize fixed columns
*  =======================
*
*     Compute the QR factorization of fixed columns and update
*     remaining columns.
*
      IF( NFXD.GT.0 ) THEN
         NA = MIN( M, NFXD )
*CC      CALL CGEQR2( M, NA, A, LDA, TAU, WORK, INFO )
         CALL CGEQRF( M, NA, A, LDA, TAU, WORK, LWORK, INFO )
         IWS = MAX( IWS, INT( WORK( 1 ) ) )
         IF( NA.LT.N ) THEN
*CC         CALL CUNM2R( 'Left', 'Conjugate Transpose', M, N-NA,
*CC  $                   NA, A, LDA, TAU, A( 1, NA+1 ), LDA, WORK,
*CC  $                   INFO )
            CALL CUNMQR( 'Left', 'Conjugate Transpose', M, N-NA, NA, A,
     $                   LDA, TAU, A( 1, NA+1 ), LDA, WORK, LWORK,
     $                   INFO )
            IWS = MAX( IWS, INT( WORK( 1 ) ) )
         END IF
      END IF
*
*     Factorize free columns
*  ======================
*
      IF( NFXD.LT.MINMN ) THEN
*
         SM = M - NFXD
         SN = N - NFXD
         SMINMN = MINMN - NFXD
*
*        Determine the block size.
*
         NB = ILAENV( INB, 'CGEQRF', ' ', SM, SN, -1, -1 )
         NBMIN = 2
         NX = 0
*
         IF( ( NB.GT.1 ) .AND. ( NB.LT.SMINMN ) ) THEN
*
*           Determine when to cross over from blocked to unblocked code.
*
            NX = MAX( 0, ILAENV( IXOVER, 'CGEQRF', ' ', SM, SN, -1,
     $           -1 ) )
*
*
            IF( NX.LT.SMINMN ) THEN
*
*              Determine if workspace is large enough for blocked code.
*
               MINWS = ( SN+1 )*NB
               IWS = MAX( IWS, MINWS )
               IF( LWORK.LT.MINWS ) THEN
*
*                 Not enough workspace to use optimal NB: Reduce NB and
*                 determine the minimum value of NB.
*
                  NB = LWORK / ( SN+1 )
                  NBMIN = MAX( 2, ILAENV( INBMIN, 'CGEQRF', ' ', SM, SN,
     $                    -1, -1 ) )
*
*
               END IF
            END IF
         END IF
*
*        Initialize partial column norms. The first N elements of work
*        store the exact column norms.
*
         DO 20 J = NFXD + 1, N
            RWORK( J ) = SCNRM2( SM, A( NFXD+1, J ), 1 )
            RWORK( N+J ) = RWORK( J )
   20    CONTINUE
*
         IF( ( NB.GE.NBMIN ) .AND. ( NB.LT.SMINMN ) .AND.
     $       ( NX.LT.SMINMN ) ) THEN
*
*           Use blocked code initially.
*
            J = NFXD + 1
*
*           Compute factorization: while loop.
*
*
            TOPBMN = MINMN - NX
   30       CONTINUE
            IF( J.LE.TOPBMN ) THEN
               JB = MIN( NB, TOPBMN-J+1 )
*
*              Factorize JB columns among columns J:N.
*
               CALL CLAQPSMOD( M, N-J+1, J-1, JB, FJB, A( 1, J ), LDA,
     $              JPVT( J ), TAU( J ), RWORK( J ),
     $              RWORK( N+J ), WORK( 1 ), WORK( JB+1 ),
     $              N-J+1, DEPTH )
*
               DO C=J,J+FJB-1
                  IF(ABS(A(C,C))/ABS(A(1,1))<=RTOL .OR.
     $                 ABS(A(C,C))<=ATOL) THEN
                   GOTO 99
                 ELSE
                   RANK=RANK+1
                 ENDIF
               END DO
               J = J + FJB
               GO TO 30
            END IF
         ELSE
            J = NFXD + 1
         END IF
*
*        Use unblocked code to factor the last or only block.
*
*
         IF( J.LE.MINMN ) THEN
            CALL CLAQP2MOD( M, N-J+1, J-1, A( 1, J ), LDA, JPVT( J ),
     $           TAU( J ), RWORK( J ), RWORK( N+J ),
     $           WORK( 1 ), DEPTH )
*
            DO C=J,MINMN
               IF(ABS(A(C,C))/ABS(A(1,1))<=RTOL .OR.
     $                 ABS(A(C,C))<=ATOL) THEN
                GOTO 99
              ELSE
                RANK=RANK+1
              ENDIF
            END DO
         END IF
      END IF
*
      WORK( 1 ) = IWS

   99 CONTINUE
      RETURN
*
*     End of CGEQP3
*
      END
