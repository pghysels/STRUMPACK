      SUBROUTINE CGETRFMOD( M, N, A, LDA, IPIV, INFO, DEPTH )
*
*  -- LAPACK computational routine (version 3.4.0) --
*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     November 2011
*
*     .. Scalar Arguments ..
      INTEGER            INFO, LDA, M, N
*     ..
*     .. Array Arguments ..
      INTEGER            IPIV( * )
      COMPLEX            A( LDA, * )
*     ..
*
*  =====================================================================
*
*     .. Parameters ..
      COMPLEX            ONE
      PARAMETER          ( ONE = ( 1.0E+0, 0.0E+0 ) )
*     ..
*     .. Local Scalars ..
      INTEGER            I, IINFO, J, JB, NB
*     ..
*     .. External Subroutines ..
      EXTERNAL           CGEMM_OMP_TASK, CGETF2MOD, CLASWP_OMP_TASK,
     $     CTRSM_OMP_TASK, XERBLA
*     ..
*     .. External Functions ..
      INTEGER            ILAENV
      EXTERNAL           ILAENV
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          MAX, MIN
*     ..
*     .. Executable Statements ..
*
*     Test the input parameters.
*
      INFO = 0
      IF( M.LT.0 ) THEN
         INFO = -1
      ELSE IF( N.LT.0 ) THEN
         INFO = -2
      ELSE IF( LDA.LT.MAX( 1, M ) ) THEN
         INFO = -4
      END IF
      IF( INFO.NE.0 ) THEN
         CALL XERBLA( 'CGETRF', -INFO )
         RETURN
      END IF
*
*     Quick return if possible
*
      IF( M.EQ.0 .OR. N.EQ.0 )
     $   RETURN
*
*     Determine the block size for this environment.
*
      NB = ILAENV( 1, 'CGETRF', ' ', M, N, -1, -1 )
      IF( NB.LE.1 .OR. NB.GE.MIN( M, N ) ) THEN
*
*        Use unblocked code.
*
         CALL CGETF2MOD( M, N, A, LDA, IPIV, INFO, DEPTH )
      ELSE
*
*        Use blocked code.
*
         DO 20 J = 1, MIN( M, N ), NB
            JB = MIN( MIN( M, N )-J+1, NB )
*
*           Factor diagonal and subdiagonal blocks and test for exact
*           singularity.
*
            CALL CGETF2MOD( M-J+1, JB, A( J, J ), LDA, IPIV( J ),
     $           IINFO, DEPTH )
*
*           Adjust INFO and the pivot indices.
*
            IF( INFO.EQ.0 .AND. IINFO.GT.0 )
     $         INFO = IINFO + J - 1
            DO 10 I = J, MIN( M, J+JB-1 )
               IPIV( I ) = J - 1 + IPIV( I )
   10       CONTINUE
*
*           Apply interchanges to columns 1:J-1.
*
            CALL CLASWP_OMP_TASK( J-1, A, LDA, J, J+JB-1, IPIV,
     $           1, DEPTH )
*
            IF( J+JB.LE.N ) THEN
*
*              Apply interchanges to columns J+JB:N.
*
               CALL CLASWP_OMP_TASK( N-J-JB+1, A( 1, J+JB ), LDA, J,
     $              J+JB-1, IPIV, 1, DEPTH )
*
*              Compute block row of U.
*
               CALL CTRSM_OMP_TASK( 'Left', 'Lower', 'No transpose',
     $              'Unit', JB, N-J-JB+1, ONE, A( J, J ),
     $              LDA, A( J, J+JB ), LDA, DEPTH )
               IF( J+JB.LE.M ) THEN
*
*                 Update trailing submatrix.
*
                  CALL CGEMM_OMP_TASK( 'No transpose', 'No transpose',
     $                         M-J-JB+1, N-J-JB+1, JB, -ONE,
     $                         A( J+JB, J ), LDA, A( J, J+JB ), LDA,
     $                         ONE, A( J+JB, J+JB ), LDA, DEPTH )
               END IF
            END IF
   20    CONTINUE
      END IF
      RETURN
*
*     End of CGETRF
*
      END
