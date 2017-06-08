      SUBROUTINE DLARFBMOD( SIDE, TRANS, DIRECT, STOREV, M, N, K, V,
     $     LDV, T, LDT, C, LDC, WORK, LDWORK, DEPTH )
*
*  -- LAPACK auxiliary routine (version 3.5.0) --
*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     June 2013
*
*     .. Scalar Arguments ..
      CHARACTER          DIRECT, SIDE, STOREV, TRANS
      INTEGER            K, LDC, LDT, LDV, LDWORK, M, N
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   C( LDC, * ), T( LDT, * ), V( LDV, * ),
     $                   WORK( LDWORK, * )
*     ..
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ONE
      PARAMETER          ( ONE = 1.0D+0 )
*     ..
*     .. Local Scalars ..
      CHARACTER          TRANST
      INTEGER            I, J
*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      EXTERNAL           LSAME
*     ..
*     .. External Subroutines ..
      EXTERNAL           DCOPY, DGEMM_OMP_TASK, DTRMM_OMP_TASK
*     ..
*     .. Executable Statements ..
*
*     Quick return if possible
*
      IF( M.LE.0 .OR. N.LE.0 )
     $   RETURN
*
      IF( LSAME( TRANS, 'N' ) ) THEN
         TRANST = 'T'
      ELSE
         TRANST = 'N'
      END IF
*
      IF( LSAME( STOREV, 'C' ) ) THEN
*
         IF( LSAME( DIRECT, 'F' ) ) THEN
*
*           Let  V =  ( V1 )    (first K rows)
*                     ( V2 )
*           where  V1  is unit lower triangular.
*
            IF( LSAME( SIDE, 'L' ) ) THEN
*
*              Form  H * C  or  H**T * C  where  C = ( C1 )
*                                                    ( C2 )
*
*              W := C**T * V  =  (C1**T * V1 + C2**T * V2)  (stored in WORK)
*
*              W := C1**T
*
               DO 10 J = 1, K
                  CALL DCOPY( N, C( J, 1 ), LDC, WORK( 1, J ), 1 )
   10          CONTINUE
*
*              W := W * V1
*
               CALL DTRMM_OMP_TASK( 'Right', 'Lower',
     $              'No transpose', 'Unit', N,
     $              K, ONE, V, LDV, WORK, LDWORK, DEPTH )
               IF( M.GT.K ) THEN
*
*                 W := W + C2**T * V2
*
                  CALL DGEMM_OMP_TASK( 'Transpose', 'No transpose', N,
     $                 K, M-K,
     $                 ONE, C( K+1, 1 ), LDC, V( K+1, 1 ), LDV,
     $                 ONE, WORK, LDWORK, DEPTH )
               END IF
*
*              W := W * T**T  or  W * T
*
               CALL DTRMM_OMP_TASK( 'Right', 'Upper', TRANST,
     $              'Non-unit', N, K,
     $                     ONE, T, LDT, WORK, LDWORK, DEPTH )
*
*              C := C - V * W**T
*
               IF( M.GT.K ) THEN
*
*                 C2 := C2 - V2 * W**T
*
                  CALL DGEMM_OMP_TASK( 'No transpose', 'Transpose',
     $                 M-K, N, K,
     $                 -ONE, V( K+1, 1 ), LDV, WORK, LDWORK, ONE,
     $                 C( K+1, 1 ), LDC, DEPTH )
               END IF
*
*              W := W * V1**T
*
               CALL DTRMM_OMP_TASK( 'Right', 'Lower', 'Transpose',
     $              'Unit', N, K,
     $              ONE, V, LDV, WORK, LDWORK, DEPTH )
*
*              C1 := C1 - W**T
*
               DO 30 J = 1, K
                  DO 20 I = 1, N
                     C( J, I ) = C( J, I ) - WORK( I, J )
   20             CONTINUE
   30          CONTINUE
*
            ELSE IF( LSAME( SIDE, 'R' ) ) THEN
*
*              Form  C * H  or  C * H**T  where  C = ( C1  C2 )
*
*              W := C * V  =  (C1*V1 + C2*V2)  (stored in WORK)
*
*              W := C1
*
               DO 40 J = 1, K
                  CALL DCOPY( M, C( 1, J ), 1, WORK( 1, J ), 1 )
   40          CONTINUE
*
*              W := W * V1
*
               CALL DTRMM_OMP_TASK( 'Right', 'Lower', 'No transpose', 
     $              'Unit', M,
     $              K, ONE, V, LDV, WORK, LDWORK, DEPTH )
               IF( N.GT.K ) THEN
*
*                 W := W + C2 * V2
*
                  CALL DGEMM_OMP_TASK( 'No transpose', 'No transpose',
     $                 M, K, N-K,
     $                 ONE, C( 1, K+1 ), LDC, V( K+1, 1 ), LDV,
     $                 ONE, WORK, LDWORK, DEPTH )
               END IF
*
*              W := W * T  or  W * T**T
*
               CALL DTRMM_OMP_TASK( 'Right', 'Upper', TRANS, 
     $              'Non-unit', M, K,
     $              ONE, T, LDT, WORK, LDWORK, DEPTH )
*
*              C := C - W * V**T
*
               IF( N.GT.K ) THEN
*
*                 C2 := C2 - W * V2**T
*
                  CALL DGEMM_OMP_TASK( 'No transpose', 'Transpose', M,
     $                 N-K, K,
     $                 -ONE, WORK, LDWORK, V( K+1, 1 ), LDV, ONE,
     $                 C( 1, K+1 ), LDC, DEPTH )
               END IF
*
*              W := W * V1**T
*
               CALL DTRMM_OMP_TASK( 'Right', 'Lower', 'Transpose',
     $              'Unit', M, K,
     $              ONE, V, LDV, WORK, LDWORK, DEPTH )
*
*              C1 := C1 - W
*
               DO 60 J = 1, K
                  DO 50 I = 1, M
                     C( I, J ) = C( I, J ) - WORK( I, J )
   50             CONTINUE
   60          CONTINUE
            END IF
*
         ELSE
*
*           Let  V =  ( V1 )
*                     ( V2 )    (last K rows)
*           where  V2  is unit upper triangular.
*
            IF( LSAME( SIDE, 'L' ) ) THEN
*
*              Form  H * C  or  H**T * C  where  C = ( C1 )
*                                                    ( C2 )
*
*              W := C**T * V  =  (C1**T * V1 + C2**T * V2)  (stored in WORK)
*
*              W := C2**T
*
               DO 70 J = 1, K
                  CALL DCOPY( N, C( M-K+J, 1 ), LDC, WORK( 1, J ), 1 )
   70          CONTINUE
*
*              W := W * V2
*
               CALL DTRMM_OMP_TASK( 'Right', 'Upper', 'No transpose', 
     $              'Unit', N,
     $              K, ONE, V( M-K+1, 1 ), LDV, WORK, LDWORK, DEPTH )
               IF( M.GT.K ) THEN
*
*                 W := W + C1**T * V1
*
                  CALL DGEMM_OMP_TASK( 'Transpose', 'No transpose',
     $                 N, K, M-K,
     $                 ONE, C, LDC, V, LDV, ONE, WORK, LDWORK, DEPTH )
               END IF
*
*              W := W * T**T  or  W * T
*
               CALL DTRMM_OMP_TASK( 'Right', 'Lower', TRANST,
     $              'Non-unit', N, K,
     $              ONE, T, LDT, WORK, LDWORK, DEPTH )
*
*              C := C - V * W**T
*
               IF( M.GT.K ) THEN
*
*                 C1 := C1 - V1 * W**T
*
                  CALL DGEMM_OMP_TASK( 'No transpose', 'Transpose', 
     $                 M-K, N, K,
     $                 -ONE, V, LDV, WORK, LDWORK, ONE, C, LDC, DEPTH )
               END IF
*
*              W := W * V2**T
*
               CALL DTRMM_OMP_TASK( 'Right', 'Upper', 'Transpose',
     $              'Unit', N, K,
     $              ONE, V( M-K+1, 1 ), LDV, WORK, LDWORK, DEPTH )
*
*              C2 := C2 - W**T
*
               DO 90 J = 1, K
                  DO 80 I = 1, N
                     C( M-K+J, I ) = C( M-K+J, I ) - WORK( I, J )
   80             CONTINUE
   90          CONTINUE
*
            ELSE IF( LSAME( SIDE, 'R' ) ) THEN
*
*              Form  C * H  or  C * H**T  where  C = ( C1  C2 )
*
*              W := C * V  =  (C1*V1 + C2*V2)  (stored in WORK)
*
*              W := C2
*
               DO 100 J = 1, K
                  CALL DCOPY( M, C( 1, N-K+J ), 1, WORK( 1, J ), 1 )
  100          CONTINUE
*
*              W := W * V2
*
               CALL DTRMM_OMP_TASK( 'Right', 'Upper', 'No transpose',
     $              'Unit', M,
     $              K, ONE, V( N-K+1, 1 ), LDV, WORK, LDWORK, DEPTH )
               IF( N.GT.K ) THEN
*
*                 W := W + C1 * V1
*
                  CALL DGEMM_OMP_TASK( 'No transpose', 'No transpose', 
     $                 M, K, N-K,
     $                 ONE, C, LDC, V, LDV, ONE, WORK, LDWORK, DEPTH )
               END IF
*
*              W := W * T  or  W * T**T
*
               CALL DTRMM_OMP_TASK( 'Right', 'Lower', TRANS, 
     $              'Non-unit', M, K,
     $              ONE, T, LDT, WORK, LDWORK, DEPTH )
*
*              C := C - W * V**T
*
               IF( N.GT.K ) THEN
*
*                 C1 := C1 - W * V1**T
*
                  CALL DGEMM_OMP_TASK( 'No transpose', 'Transpose', M,
     $                 N-K, K,
     $                 -ONE, WORK, LDWORK, V, LDV, ONE, C, LDC, DEPTH )
               END IF
*
*              W := W * V2**T
*
               CALL DTRMM_OMP_TASK( 'Right', 'Upper', 'Transpose', 
     $              'Unit', M, K,
     $              ONE, V( N-K+1, 1 ), LDV, WORK, LDWORK, DEPTH )
*
*              C2 := C2 - W
*
               DO 120 J = 1, K
                  DO 110 I = 1, M
                     C( I, N-K+J ) = C( I, N-K+J ) - WORK( I, J )
  110             CONTINUE
  120          CONTINUE
            END IF
         END IF
*
      ELSE IF( LSAME( STOREV, 'R' ) ) THEN
*
         IF( LSAME( DIRECT, 'F' ) ) THEN
*
*           Let  V =  ( V1  V2 )    (V1: first K columns)
*           where  V1  is unit upper triangular.
*
            IF( LSAME( SIDE, 'L' ) ) THEN
*
*              Form  H * C  or  H**T * C  where  C = ( C1 )
*                                                    ( C2 )
*
*              W := C**T * V**T  =  (C1**T * V1**T + C2**T * V2**T) (stored in WORK)
*
*              W := C1**T
*
               DO 130 J = 1, K
                  CALL DCOPY( N, C( J, 1 ), LDC, WORK( 1, J ), 1 )
  130          CONTINUE
*
*              W := W * V1**T
*
               CALL DTRMM_OMP_TASK( 'Right', 'Upper', 'Transpose', 
     $              'Unit', N, K,
     $              ONE, V, LDV, WORK, LDWORK, DEPTH )
               IF( M.GT.K ) THEN
*
*                 W := W + C2**T * V2**T
*
                  CALL DGEMM_OMP_TASK( 'Transpose', 'Transpose', N, K, 
     $                 M-K, ONE,
     $                 C( K+1, 1 ), LDC, V( 1, K+1 ), LDV, ONE,
     $                 WORK, LDWORK, DEPTH )
               END IF
*
*              W := W * T**T  or  W * T
*
               CALL DTRMM_OMP_TASK( 'Right', 'Upper', TRANST,
     $              'Non-unit', N, K,
     $              ONE, T, LDT, WORK, LDWORK, DEPTH )
*
*              C := C - V**T * W**T
*
               IF( M.GT.K ) THEN
*
*                 C2 := C2 - V2**T * W**T
*
                  CALL DGEMM_OMP_TASK( 'Transpose', 'Transpose', M-K,
     $                 N, K, -ONE,
     $                 V( 1, K+1 ), LDV, WORK, LDWORK, ONE,
     $                 C( K+1, 1 ), LDC, DEPTH )
               END IF
*
*              W := W * V1
*
               CALL DTRMM_OMP_TASK( 'Right', 'Upper', 'No transpose',
     $              'Unit', N,
     $              K, ONE, V, LDV, WORK, LDWORK, DEPTH )
*
*              C1 := C1 - W**T
*
               DO 150 J = 1, K
                  DO 140 I = 1, N
                     C( J, I ) = C( J, I ) - WORK( I, J )
  140             CONTINUE
  150          CONTINUE
*
            ELSE IF( LSAME( SIDE, 'R' ) ) THEN
*
*              Form  C * H  or  C * H**T  where  C = ( C1  C2 )
*
*              W := C * V**T  =  (C1*V1**T + C2*V2**T)  (stored in WORK)
*
*              W := C1
*
               DO 160 J = 1, K
                  CALL DCOPY( M, C( 1, J ), 1, WORK( 1, J ), 1 )
  160          CONTINUE
*
*              W := W * V1**T
*
               CALL DTRMM_OMP_TASK( 'Right', 'Upper', 'Transpose', 
     $              'Unit', M, K,
     $              ONE, V, LDV, WORK, LDWORK, DEPTH )
               IF( N.GT.K ) THEN
*
*                 W := W + C2 * V2**T
*
                  CALL DGEMM_OMP_TASK( 'No transpose', 'Transpose', 
     $                 M, K, N-K,
     $                 ONE, C( 1, K+1 ), LDC, V( 1, K+1 ), LDV,
     $                 ONE, WORK, LDWORK, DEPTH )
               END IF
*
*              W := W * T  or  W * T**T
*
               CALL DTRMM_OMP_TASK( 'Right', 'Upper', TRANS, 
     $              'Non-unit', M, K,
     $              ONE, T, LDT, WORK, LDWORK, DEPTH )
*
*              C := C - W * V
*
               IF( N.GT.K ) THEN
*
*                 C2 := C2 - W * V2
*
                  CALL DGEMM_OMP_TASK( 'No transpose', 'No transpose', 
     $                 M, N-K, K,
     $                 -ONE, WORK, LDWORK, V( 1, K+1 ), LDV, ONE,
     $                 C( 1, K+1 ), LDC, DEPTH )
               END IF
*
*              W := W * V1
*
               CALL DTRMM_OMP_TASK( 'Right', 'Upper', 'No transpose', 
     $              'Unit', M,
     $              K, ONE, V, LDV, WORK, LDWORK, DEPTH )
*
*              C1 := C1 - W
*
               DO 180 J = 1, K
                  DO 170 I = 1, M
                     C( I, J ) = C( I, J ) - WORK( I, J )
  170             CONTINUE
  180          CONTINUE
*
            END IF
*
         ELSE
*
*           Let  V =  ( V1  V2 )    (V2: last K columns)
*           where  V2  is unit lower triangular.
*
            IF( LSAME( SIDE, 'L' ) ) THEN
*
*              Form  H * C  or  H**T * C  where  C = ( C1 )
*                                                    ( C2 )
*
*              W := C**T * V**T  =  (C1**T * V1**T + C2**T * V2**T) (stored in WORK)
*
*              W := C2**T
*
               DO 190 J = 1, K
                  CALL DCOPY( N, C( M-K+J, 1 ), LDC, WORK( 1, J ), 1 )
  190          CONTINUE
*
*              W := W * V2**T
*
               CALL DTRMM_OMP_TASK( 'Right', 'Lower', 'Transpose',
     $              'Unit', N, K,
     $              ONE, V( 1, M-K+1 ), LDV, WORK, LDWORK, DEPTH )
               IF( M.GT.K ) THEN
*
*                 W := W + C1**T * V1**T
*
                  CALL DGEMM_OMP_TASK( 'Transpose', 'Transpose', N,
     $                 K, M-K, ONE,
     $                 C, LDC, V, LDV, ONE, WORK, LDWORK, DEPTH )
               END IF
*
*              W := W * T**T  or  W * T
*
               CALL DTRMM_OMP_TASK( 'Right', 'Lower', TRANST, 
     $              'Non-unit', N, K,
     $              ONE, T, LDT, WORK, LDWORK, DEPTH )
*
*              C := C - V**T * W**T
*
               IF( M.GT.K ) THEN
*
*                 C1 := C1 - V1**T * W**T
*
                  CALL DGEMM_OMP_TASK( 'Transpose', 'Transpose', M-K,
     $                 N, K, -ONE,
     $                 V, LDV, WORK, LDWORK, ONE, C, LDC, DEPTH )
               END IF
*
*              W := W * V2
*
               CALL DTRMM_OMP_TASK( 'Right', 'Lower', 'No transpose',
     $              'Unit', N,
     $              K, ONE, V( 1, M-K+1 ), LDV, WORK, LDWORK, DEPTH )
*
*              C2 := C2 - W**T
*
               DO 210 J = 1, K
                  DO 200 I = 1, N
                     C( M-K+J, I ) = C( M-K+J, I ) - WORK( I, J )
  200             CONTINUE
  210          CONTINUE
*
            ELSE IF( LSAME( SIDE, 'R' ) ) THEN
*
*              Form  C * H  or  C * H'  where  C = ( C1  C2 )
*
*              W := C * V**T  =  (C1*V1**T + C2*V2**T)  (stored in WORK)
*
*              W := C2
*
               DO 220 J = 1, K
                  CALL DCOPY( M, C( 1, N-K+J ), 1, WORK( 1, J ), 1 )
  220          CONTINUE
*
*              W := W * V2**T
*
               CALL DTRMM_OMP_TASK( 'Right', 'Lower', 'Transpose', 
     $              'Unit', M, K,
     $              ONE, V( 1, N-K+1 ), LDV, WORK, LDWORK, DEPTH )
               IF( N.GT.K ) THEN
*
*                 W := W + C1 * V1**T
*
                  CALL DGEMM_OMP_TASK( 'No transpose', 'Transpose', 
     $                 M, K, N-K,
     $                 ONE, C, LDC, V, LDV, ONE, WORK, LDWORK, DEPTH )
               END IF
*
*              W := W * T  or  W * T**T
*
               CALL DTRMM_OMP_TASK( 'Right', 'Lower', TRANS,
     $              'Non-unit', M, K,
     $              ONE, T, LDT, WORK, LDWORK, DEPTH )
*
*              C := C - W * V
*
               IF( N.GT.K ) THEN
*
*                 C1 := C1 - W * V1
*
                  CALL DGEMM_OMP_TASK( 'No transpose', 'No transpose', 
     $                 M, N-K, K,
     $                 -ONE, WORK, LDWORK, V, LDV, ONE, C, LDC, DEPTH )
               END IF
*
*              W := W * V2
*
               CALL DTRMM_OMP_TASK( 'Right', 'Lower', 'No transpose',
     $              'Unit', M,
     $              K, ONE, V( 1, N-K+1 ), LDV, WORK, LDWORK, DEPTH )
*
*              C1 := C1 - W
*
               DO 240 J = 1, K
                  DO 230 I = 1, M
                     C( I, N-K+J ) = C( I, N-K+J ) - WORK( I, J )
  230             CONTINUE
  240          CONTINUE
*
            END IF
*
         END IF
      END IF
*
      RETURN
*
*     End of DLARFB
*
      END
