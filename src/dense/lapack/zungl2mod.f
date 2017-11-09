      SUBROUTINE ZUNGL2MOD( M, N, K, A, LDA, TAU, WORK,
     $     INFO, DEPTH )
*
*  -- LAPACK computational routine (version 3.4.2) --
*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     September 2012
*
*     .. Scalar Arguments ..
      INTEGER            INFO, K, LDA, M, N
*     ..
*     .. Array Arguments ..
      COMPLEX*16         A( LDA, * ), TAU( * ), WORK( * )
*     ..
*
*  =====================================================================
*
*     .. Parameters ..
      COMPLEX*16         ONE, ZERO
      PARAMETER          ( ONE = ( 1.0D+0, 0.0D+0 ),
     $                   ZERO = ( 0.0D+0, 0.0D+0 ) )
*     ..
*     .. Local Scalars ..
      INTEGER            I, J, L
*     ..
*     .. External Subroutines ..
      EXTERNAL           XERBLA, ZLACGV, ZLARFMOD, ZSCAL
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          DCONJG, MAX
*     ..
*     .. Executable Statements ..
*
*     Test the input arguments
*
      INFO = 0
      IF( M.LT.0 ) THEN
         INFO = -1
      ELSE IF( N.LT.M ) THEN
         INFO = -2
      ELSE IF( K.LT.0 .OR. K.GT.M ) THEN
         INFO = -3
      ELSE IF( LDA.LT.MAX( 1, M ) ) THEN
         INFO = -5
      END IF
      IF( INFO.NE.0 ) THEN
         CALL XERBLA( 'ZUNGL2', -INFO )
         RETURN
      END IF
*
*     Quick return if possible
*
      IF( M.LE.0 )
     $   RETURN
*
      IF( K.LT.M ) THEN
*
*        Initialise rows k+1:m to rows of the unit matrix
*
         DO 20 J = 1, N
            DO 10 L = K + 1, M
               A( L, J ) = ZERO
   10       CONTINUE
            IF( J.GT.K .AND. J.LE.M )
     $         A( J, J ) = ONE
   20    CONTINUE
      END IF
*
      DO 40 I = K, 1, -1
*
*        Apply H(i)**H to A(i:m,i:n) from the right
*
         IF( I.LT.N ) THEN
            CALL ZLACGV( N-I, A( I, I+1 ), LDA )
            IF( I.LT.M ) THEN
               A( I, I ) = ONE
               CALL ZLARFMOD( 'Right', M-I, N-I+1, A( I, I ), LDA,
     $              DCONJG( TAU( I ) ), A( I+1, I ), LDA, WORK, DEPTH )
            END IF
            CALL ZSCAL( N-I, -TAU( I ), A( I, I+1 ), LDA )
            CALL ZLACGV( N-I, A( I, I+1 ), LDA )
         END IF
         A( I, I ) = ONE - DCONJG( TAU( I ) )
*
*        Set A(i,1:i-1) to zero
*
         DO 30 L = 1, I - 1
            A( I, L ) = ZERO
   30    CONTINUE
   40 CONTINUE
      RETURN
*
*     End of ZUNGL2
*
      END
