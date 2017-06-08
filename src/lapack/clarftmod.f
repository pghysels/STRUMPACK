      SUBROUTINE CLARFTMOD( DIRECT, STOREV, N, K, V, LDV, TAU, 
     $     T, LDT, DEPTH )
*
*  -- LAPACK auxiliary routine (version 3.4.2) --
*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     September 2012
*
*     .. Scalar Arguments ..
      CHARACTER          DIRECT, STOREV
      INTEGER            K, LDT, LDV, N
*     ..
*     .. Array Arguments ..
      COMPLEX            T( LDT, * ), TAU( * ), V( LDV, * )
*     ..
*
*  =====================================================================
*
*     .. Parameters ..
      COMPLEX            ONE, ZERO
      PARAMETER          ( ONE = ( 1.0E+0, 0.0E+0 ),
     $                   ZERO = ( 0.0E+0, 0.0E+0 ) )
*     ..
*     .. Local Scalars ..
      INTEGER            I, J, PREVLASTV, LASTV
*     ..
*     .. External Subroutines ..
      EXTERNAL           CGEMV_OMP_TASK, CLACGV, CTRMV_OMP_TASK
*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      EXTERNAL           LSAME
*     ..
*     .. Executable Statements ..
*
*     Quick return if possible
*
      IF( N.EQ.0 )
     $   RETURN
*
      IF( LSAME( DIRECT, 'F' ) ) THEN
         PREVLASTV = N
         DO I = 1, K
            PREVLASTV = MAX( PREVLASTV, I )
            IF( TAU( I ).EQ.ZERO ) THEN
*
*              H(i)  =  I
*
               DO J = 1, I
                  T( J, I ) = ZERO
               END DO
            ELSE
*
*              general case
*
               IF( LSAME( STOREV, 'C' ) ) THEN
*                 Skip any trailing zeros.
                  DO LASTV = N, I+1, -1
                     IF( V( LASTV, I ).NE.ZERO ) EXIT
                  END DO
                  DO J = 1, I-1
                     T( J, I ) = -TAU( I ) * CONJG( V( I , J ) )
                  END DO                     
                  J = MIN( LASTV, PREVLASTV )
*
*                 T(1:i-1,i) := - tau(i) * V(i:j,1:i-1)**H * V(i:j,i)
*
                  CALL CGEMV_OMP_TASK( 'Conjugate transpose', J-I, I-1,
     $                        -TAU( I ), V( I+1, 1 ), LDV, 
     $                        V( I+1, I ), 1,
     $                        ONE, T( 1, I ), 1, DEPTH )
               ELSE
*                 Skip any trailing zeros.
                  DO LASTV = N, I+1, -1
                     IF( V( I, LASTV ).NE.ZERO ) EXIT
                  END DO
                  DO J = 1, I-1
                     T( J, I ) = -TAU( I ) * V( J , I )
                  END DO                     
                  J = MIN( LASTV, PREVLASTV )
*
*                 T(1:i-1,i) := - tau(i) * V(1:i-1,i:j) * V(i,i:j)**H
*
                  CALL CGEMM_OMP_TASK( 'N', 'C', I-1, 1, J-I, -TAU( I ),
     $                        V( 1, I+1 ), LDV, V( I, I+1 ), LDV,
     $                        ONE, T( 1, I ), LDT, DEPTH )
               END IF
*
*              T(1:i-1,i) := T(1:i-1,1:i-1) * T(1:i-1,i)
*
               CALL CTRMV_OMP_TASK( 'Upper', 'No transpose',
     $              'Non-unit', I-1, T,
     $              LDT, T( 1, I ), 1, DEPTH )
               T( I, I ) = TAU( I )
               IF( I.GT.1 ) THEN
                  PREVLASTV = MAX( PREVLASTV, LASTV )
               ELSE
                  PREVLASTV = LASTV
               END IF
            END IF
         END DO
      ELSE
         PREVLASTV = 1
         DO I = K, 1, -1
            IF( TAU( I ).EQ.ZERO ) THEN
*
*              H(i)  =  I
*
               DO J = I, K
                  T( J, I ) = ZERO
               END DO
            ELSE
*
*              general case
*
               IF( I.LT.K ) THEN
                  IF( LSAME( STOREV, 'C' ) ) THEN
*                    Skip any leading zeros.
                     DO LASTV = 1, I-1
                        IF( V( LASTV, I ).NE.ZERO ) EXIT
                     END DO
                     DO J = I+1, K
                        T( J, I ) = -TAU( I ) * CONJG( V( N-K+I , J ) )
                     END DO                        
                     J = MAX( LASTV, PREVLASTV )
*
*                    T(i+1:k,i) = -tau(i) * V(j:n-k+i,i+1:k)**H * V(j:n-k+i,i)
*
                     CALL CGEMV_OMP_TASK( 'Conjugate transpose',
     $                    N-K+I-J, K-I,
     $                    -TAU( I ), V( J, I+1 ), LDV, V( J, I ),
     $                    1, ONE, T( I+1, I ), 1, DEPTH )
                  ELSE
*                    Skip any leading zeros.
                     DO LASTV = 1, I-1
                        IF( V( I, LASTV ).NE.ZERO ) EXIT
                     END DO
                     DO J = I+1, K
                        T( J, I ) = -TAU( I ) * V( J, N-K+I )
                     END DO                      
                     J = MAX( LASTV, PREVLASTV )
*
*                    T(i+1:k,i) = -tau(i) * V(i+1:k,j:n-k+i) * V(i,j:n-k+i)**H
*
                     CALL CGEMM_OMP_TASK( 'N', 'C', K-I, 1, N-K+I-J,
     $                    -TAU( I ), V( I+1, J ), LDV, V( I, J ), LDV,
     $                    ONE, T( I+1, I ), LDT, DEPTH )
                  END IF
*
*                 T(i+1:k,i) := T(i+1:k,i+1:k) * T(i+1:k,i)
*
                  CALL CTRMV_OMP_TASK( 'Lower', 'No transpose', 
     $                 'Non-unit', K-I,
     $                 T( I+1, I+1 ), LDT, T( I+1, I ), 1, DEPTH )
                  IF( I.GT.1 ) THEN
                     PREVLASTV = MIN( PREVLASTV, LASTV )
                  ELSE
                     PREVLASTV = LASTV
                  END IF
               END IF
               T( I, I ) = TAU( I )
            END IF
         END DO
      END IF
      RETURN
*
*     End of CLARFT
*
      END
