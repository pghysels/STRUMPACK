      SUBROUTINE DLAQPSMOD( M, N, OFFSET, NB, KB, A, LDA, JPVT, TAU,
     $     VN1, VN2, AUXV, F, LDF, DEPTH )
*
*  -- LAPACK auxiliary routine (version 3.4.2) --
*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     September 2012
*
*     .. Scalar Arguments ..
      INTEGER            KB, LDA, LDF, M, N, NB, OFFSET
*     ..
*     .. Array Arguments ..
      INTEGER            JPVT( * )
      DOUBLE PRECISION   A( LDA, * ), AUXV( * ), F( LDF, * ), TAU( * ),
     $                   VN1( * ), VN2( * )
*     ..
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ZERO, ONE
      PARAMETER          ( ZERO = 0.0D+0, ONE = 1.0D+0 )
*     ..
*     .. Local Scalars ..
      INTEGER            ITEMP, J, K, LASTRK, LSTICC, PVT, RK
      DOUBLE PRECISION   AKK, TEMP, TEMP2, TOL3Z
*     ..
*     .. External Subroutines ..
      EXTERNAL           DGEMM_OMP_TASK, DGEMV_OMP_TASK, DLARFG, DSWAP
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, DBLE, MAX, MIN, NINT, SQRT
*     ..
*     .. External Functions ..
      INTEGER            IDAMAX
      DOUBLE PRECISION   DLAMCH, DNRM2
      EXTERNAL           IDAMAX, DLAMCH, DNRM2
*     ..
*     .. Executable Statements ..
*
      LASTRK = MIN( M, N+OFFSET )
      LSTICC = 0
      K = 0
      TOL3Z = SQRT(DLAMCH('Epsilon'))
*
*     Beginning of while loop.
*
   10 CONTINUE
      IF( ( K.LT.NB ) .AND. ( LSTICC.EQ.0 ) ) THEN
         K = K + 1
         RK = OFFSET + K
*
*        Determine ith pivot column and swap if necessary
*
         PVT = ( K-1 ) + IDAMAX( N-K+1, VN1( K ), 1 )
         IF( PVT.NE.K ) THEN
            CALL DSWAP( M, A( 1, PVT ), 1, A( 1, K ), 1 )
            CALL DSWAP( K-1, F( PVT, 1 ), LDF, F( K, 1 ), LDF )
            ITEMP = JPVT( PVT )
            JPVT( PVT ) = JPVT( K )
            JPVT( K ) = ITEMP
            VN1( PVT ) = VN1( K )
            VN2( PVT ) = VN2( K )
         END IF
*
*        Apply previous Householder reflectors to column K:
*        A(RK:M,K) := A(RK:M,K) - A(RK:M,1:K-1)*F(K,1:K-1)**T.
*
         IF( K.GT.1 ) THEN
            CALL DGEMV_OMP_TASK( 'No transpose', M-RK+1, K-1, -ONE,
     $           A( RK, 1 ), LDA, F( K, 1 ), LDF,
     $           ONE, A( RK, K ), 1, DEPTH )
         END IF
*
*        Generate elementary reflector H(k).
*
         IF( RK.LT.M ) THEN
            CALL DLARFG( M-RK+1, A( RK, K ), A( RK+1, K ), 1, TAU( K ) )
         ELSE
            CALL DLARFG( 1, A( RK, K ), A( RK, K ), 1, TAU( K ) )
         END IF
*
         AKK = A( RK, K )
         A( RK, K ) = ONE
*
*        Compute Kth column of F:
*
*        Compute  F(K+1:N,K) := tau(K)*A(RK:M,K+1:N)**T*A(RK:M,K).
*
         IF( K.LT.N ) THEN
            CALL DGEMV_OMP_TASK( 'Transpose', M-RK+1, N-K, TAU( K ),
     $           A( RK, K+1 ), LDA, A( RK, K ), 1, ZERO,
     $           F( K+1, K ), 1, DEPTH )
         END IF
*
*        Padding F(1:K,K) with zeros.
*
         DO 20 J = 1, K
            F( J, K ) = ZERO
   20    CONTINUE
*
*        Incremental updating of F:
*        F(1:N,K) := F(1:N,K) - tau(K)*F(1:N,1:K-1)*A(RK:M,1:K-1)**T
*                    *A(RK:M,K).
*
         IF( K.GT.1 ) THEN
            CALL DGEMV_OMP_TASK( 'Transpose', M-RK+1, K-1, -TAU( K ),
     $           A( RK, 1 ), LDA, A( RK, K ), 1, ZERO,
     $           AUXV( 1 ), 1, DEPTH )
*
            CALL DGEMV_OMP_TASK( 'No transpose', N, K-1, ONE,
     $           F( 1, 1 ), LDF, AUXV( 1 ), 1,
     $           ONE, F( 1, K ), 1, DEPTH )
         END IF
*
*        Update the current row of A:
*        A(RK,K+1:N) := A(RK,K+1:N) - A(RK,1:K)*F(K+1:N,1:K)**T.
*
         IF( K.LT.N ) THEN
            CALL DGEMV_OMP_TASK( 'No transpose', N-K, K, -ONE, 
     $           F( K+1, 1 ), LDF, A( RK, 1 ), 
     $           LDA, ONE, A( RK, K+1 ), LDA, DEPTH )
         END IF
*
*        Update partial column norms.
*
         IF( RK.LT.LASTRK ) THEN
            DO 30 J = K + 1, N
               IF( VN1( J ).NE.ZERO ) THEN
*
*                 NOTE: The following 4 lines follow from the analysis in
*                 Lapack Working Note 176.
*
                  TEMP = ABS( A( RK, J ) ) / VN1( J )
                  TEMP = MAX( ZERO, ( ONE+TEMP )*( ONE-TEMP ) )
                  TEMP2 = TEMP*( VN1( J ) / VN2( J ) )**2
                  IF( TEMP2 .LE. TOL3Z ) THEN
                     VN2( J ) = DBLE( LSTICC )
                     LSTICC = J
                  ELSE
                     VN1( J ) = VN1( J )*SQRT( TEMP )
                  END IF
               END IF
   30       CONTINUE
         END IF
*
         A( RK, K ) = AKK
*
*        End of while loop.
*
         GO TO 10
      END IF
      KB = K
      RK = OFFSET + KB
*
*     Apply the block reflector to the rest of the matrix:
*     A(OFFSET+KB+1:M,KB+1:N) := A(OFFSET+KB+1:M,KB+1:N) -
*                         A(OFFSET+KB+1:M,1:KB)*F(KB+1:N,1:KB)**T.
*
      IF( KB.LT.MIN( N, M-OFFSET ) ) THEN
         CALL DGEMM_OMP_TASK( 'No transpose', 'Transpose', M-RK, N-KB,
     $                KB, -ONE, A( RK+1, 1 ), LDA, F( KB+1, 1 ), LDF,
     $                ONE, A( RK+1, KB+1 ), LDA, DEPTH )
      END IF
*
*     Recomputation of difficult columns.
*
   40 CONTINUE
      IF( LSTICC.GT.0 ) THEN
         ITEMP = NINT( VN2( LSTICC ) )
         VN1( LSTICC ) = DNRM2( M-RK, A( RK+1, LSTICC ), 1 )
*
*        NOTE: The computation of VN1( LSTICC ) relies on the fact that 
*        SNRM2 does not fail on vectors with norm below the value of
*        SQRT(DLAMCH('S')) 
*
         VN2( LSTICC ) = VN1( LSTICC )
         LSTICC = ITEMP
         GO TO 40
      END IF
*
      RETURN
*
*     End of DLAQPS
*
      END
