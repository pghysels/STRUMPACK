      SUBROUTINE PDGEQPFmod( M, N, A, IA, JA, DESCA, IPIV, TAU, WORK,
     $                    LWORK, INFO, JPERM, JPIV, RANK, RTOL, ATOL )
*
*  -- ScaLAPACK routine (version 1.7) --
*     University of Tennessee, Knoxville, Oak Ridge National Laboratory,
*     and University of California, Berkeley.
*     March 14, 2000
*
*  -- Modified by F-H Rouet, Lawrence Berkeley National Lab, August 2014
*
      IMPLICIT NONE
*     .. Scalar Arguments ..
      INTEGER            IA, JA, INFO, LWORK, M, N, RANK
*     ..
*     .. Array Arguments ..
      INTEGER            DESCA( * ), IPIV( * ), JPERM( * ), JPIV ( * )
      DOUBLE PRECISION   A( * ), TAU( * ), WORK( * ), RTOL, ATOL
*     ..
*
*  Purpose
*  =======
*
*  PDGEQPF computes a QR factorization with column pivoting of a
*  M-by-N distributed matrix sub( A ) = A(IA:IA+M-1,JA:JA+N-1):
*
*                         sub( A ) * P = Q * R.
*
*  Notes
*  =====
*
*  Each global data object is described by an associated description
*  vector.  This vector stores the information required to establish
*  the mapping between an object element and its corresponding process
*  and memory location.
*
*  Let A be a generic term for any 2D block cyclicly distributed array.
*  Such a global array has an associated description vector DESCA.
*  In the following comments, the character _ should be read as
*  "of the global array".
*
*  NOTATION        STORED IN      EXPLANATION
*  --------------- -------------- --------------------------------------
*  DTYPE_A(global) DESCA( DTYPE_ )The descriptor type.  In this case,
*                                 DTYPE_A = 1.
*  CTXT_A (global) DESCA( CTXT_ ) The BLACS context handle, indicating
*                                 the BLACS process grid A is distribu-
*                                 ted over. The context itself is glo-
*                                 bal, but the handle (the integer
*                                 value) may vary.
*  M_A    (global) DESCA( M_ )    The number of rows in the global
*                                 array A.
*  N_A    (global) DESCA( N_ )    The number of columns in the global
*                                 array A.
*  MB_A   (global) DESCA( MB_ )   The blocking factor used to distribute
*                                 the rows of the array.
*  NB_A   (global) DESCA( NB_ )   The blocking factor used to distribute
*                                 the columns of the array.
*  RSRC_A (global) DESCA( RSRC_ ) The process row over which the first
*                                 row of the array A is distributed.
*  CSRC_A (global) DESCA( CSRC_ ) The process column over which the
*                                 first column of the array A is
*                                 distributed.
*  LLD_A  (local)  DESCA( LLD_ )  The leading dimension of the local
*                                 array.  LLD_A >= MAX(1,LOCr(M_A)).
*
*  Let K be the number of rows or columns of a distributed matrix,
*  and assume that its process grid has dimension p x q.
*  LOCr( K ) denotes the number of elements of K that a process
*  would receive if K were distributed over the p processes of its
*  process column.
*  Similarly, LOCc( K ) denotes the number of elements of K that a
*  process would receive if K were distributed over the q processes of
*  its process row.
*  The values of LOCr() and LOCc() may be determined via a call to the
*  ScaLAPACK tool function, NUMROC:
*          LOCr( M ) = NUMROC( M, MB_A, MYROW, RSRC_A, NPROW ),
*          LOCc( N ) = NUMROC( N, NB_A, MYCOL, CSRC_A, NPCOL ).
*  An upper bound for these quantities may be computed by:
*          LOCr( M ) <= ceil( ceil(M/MB_A)/NPROW )*MB_A
*          LOCc( N ) <= ceil( ceil(N/NB_A)/NPCOL )*NB_A
*
*  Arguments
*  =========
*
*  M       (global input) INTEGER
*          The number of rows to be operated on, i.e. the number of rows
*          of the distributed submatrix sub( A ). M >= 0.
*
*  N       (global input) INTEGER
*          The number of columns to be operated on, i.e. the number of
*          columns of the distributed submatrix sub( A ). N >= 0.
*
*  A       (local input/local output) DOUBLE PRECISION pointer into the
*          local memory to an array of dimension (LLD_A, LOCc(JA+N-1)).
*          On entry, the local pieces of the M-by-N distributed matrix
*          sub( A ) which is to be factored. On exit, the elements on
*          and above the diagonal of sub( A ) contain the min(M,N) by N
*          upper trapezoidal matrix R (R is upper triangular if M >= N);
*          the elements below the diagonal, with the array TAU, repre-
*          sent the orthogonal matrix Q as a product of elementary
*          reflectors (see Further Details).
*
*  IA      (global input) INTEGER
*          The row index in the global array A indicating the first
*          row of sub( A ).
*
*  JA      (global input) INTEGER
*          The column index in the global array A indicating the
*          first column of sub( A ).
*
*  DESCA   (global and local input) INTEGER array of dimension DLEN_.
*          The array descriptor for the distributed matrix A.
*
*  IPIV    (local output) INTEGER array, dimension LOCc(JA+N-1).
*          On exit, if IPIV(I) = K, the local i-th column of sub( A )*P
*          was the global K-th column of sub( A ). IPIV is tied to the
*          distributed matrix A.
*
*  TAU     (local output) DOUBLE PRECISION array, dimension
*          LOCc(JA+MIN(M,N)-1). This array contains the scalar factors
*          TAU of the elementary reflectors. TAU is tied to the
*          distributed matrix A.
*
*  WORK    (local workspace/local output) DOUBLE PRECISION array,
*                                                   dimension (LWORK)
*          On exit, WORK(1) returns the minimal and optimal LWORK.
*
*  LWORK   (local or global input) INTEGER
*          The dimension of the array WORK.
*          LWORK is local input and must be at least
*          LWORK >= MAX(3,Mp0 + Nq0) + LOCc(JA+N-1)+Nq0.
*
*          IROFF = MOD( IA-1, MB_A ), ICOFF = MOD( JA-1, NB_A ),
*          IAROW = INDXG2P( IA, MB_A, MYROW, RSRC_A, NPROW ),
*          IACOL = INDXG2P( JA, NB_A, MYCOL, CSRC_A, NPCOL ),
*          Mp0   = NUMROC( M+IROFF, MB_A, MYROW, IAROW, NPROW ),
*          Nq0   = NUMROC( N+ICOFF, NB_A, MYCOL, IACOL, NPCOL ),
*          LOCc(JA+N-1) = NUMROC( JA+N-1, NB_A, MYCOL, CSRC_A, NPCOL )
*
*          and NUMROC, INDXG2P are ScaLAPACK tool functions;
*          MYROW, MYCOL, NPROW and NPCOL can be determined by calling
*          the subroutine BLACS_GRIDINFO.
*
*          If LWORK = -1, then LWORK is global input and a workspace
*          query is assumed; the routine only calculates the minimum
*          and optimal size for all work arrays. Each of these
*          values is returned in the first entry of the corresponding
*          work array, and no error message is issued by PXERBLA.
*
*
*  INFO    (global output) INTEGER
*          = 0:  successful exit
*          < 0:  If the i-th argument is an array and the j-entry had
*                an illegal value, then INFO = -(i*100+j), if the i-th
*                argument is a scalar and had an illegal value, then
*                INFO = -i.
*
*  Further Details
*  ===============
*
*  The matrix Q is represented as a product of elementary reflectors
*
*     Q = H(1) H(2) . . . H(n)
*
*  Each H(i) has the form
*
*     H = I - tau * v * v'
*
*  where tau is a real scalar, and v is a real vector with v(1:i-1) = 0
*  and v(i) = 1; v(i+1:m) is stored on exit in A(ia+i-1:ia+m-1,ja+i-1).
*
*  The matrix P is represented in jpvt as follows: If
*     jpvt(j) = i
*  then the jth column of P is the ith canonical unit vector.
*
*  =====================================================================
*
*     .. Parameters ..
      INTEGER            BLOCK_CYCLIC_2D, CSRC_, CTXT_, DLEN_, DTYPE_,
     $                   LLD_, MB_, M_, NB_, N_, RSRC_
      PARAMETER          ( BLOCK_CYCLIC_2D = 1, DLEN_ = 9, DTYPE_ = 1,
     $                     CTXT_ = 2, M_ = 3, N_ = 4, MB_ = 5, NB_ = 6,
     $                     RSRC_ = 7, CSRC_ = 8, LLD_ = 9 )
      DOUBLE PRECISION   ONE, ZERO
      PARAMETER          ( ONE = 1.0D+0, ZERO = 0.0D+0 )
*     ..
*     .. Local Scalars ..
      LOGICAL            LQUERY
      INTEGER            I, IACOL, IAROW, ICOFF, ICTXT, ICURROW,
     $                   ICURCOL, II, IIA, IOFFA, IPN, IPCOL, IPW,
     $                   IROFF, ITEMP, J, JB, JJ, JJA, JJPVT, JN, KB,
     $                   K, KK, KSTART, KSTEP, LDA, LL, LWMIN, MN, MP,
     $                   MYCOL, MYROW, NPCOL, NPROW, NQ, NQ0, PVT
      DOUBLE PRECISION   AJJ, ALPHA, TEMP, TEMP2, A11
*     ..
*     .. Local Arrays ..
      INTEGER            DESCN( DLEN_ ), IDUM1( 1 ), IDUM2( 1 )
*     ..
*     .. External Subroutines ..
      EXTERNAL           BLACS_GRIDINFO, CHK1MAT, DCOPY, DESCSET,
     $                   DGEBR2D, DGEBS2D, DGERV2D,
     $                   DGESD2D, DLARFG, DSWAP, IGERV2D,
     $                   IGESD2D, INFOG1L, INFOG2L, PCHK1MAT, PDAMAX,
     $                   PDELSET, PDLARF, PDLARFG, PDNRM2,
     $                   PXERBLA
*     ..
*     .. External Functions ..
      INTEGER            ICEIL, INDXG2P, NUMROC
      EXTERNAL           ICEIL, INDXG2P, NUMROC
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, DBLE, IDINT, MAX, MIN, MOD, SQRT
*     ..
*     .. Executable Statements ..
*
*     Get grid parameters
*
      ICTXT = DESCA( CTXT_ )
      CALL BLACS_GRIDINFO( ICTXT, NPROW, NPCOL, MYROW, MYCOL )
*
*     Test the input parameters
*
      INFO = 0
      IF( NPROW.EQ.-1 ) THEN
         INFO = -(600+CTXT_)
      ELSE
         CALL CHK1MAT( M, 1, N, 2, IA, JA, DESCA, 6, INFO )
         IF( INFO.EQ.0 ) THEN
            IROFF = MOD( IA-1, DESCA( MB_ ) )
            ICOFF = MOD( JA-1, DESCA( NB_ ) )
            IAROW = INDXG2P( IA, DESCA( MB_ ), MYROW, DESCA( RSRC_ ),
     $                       NPROW )
            IACOL = INDXG2P( JA, DESCA( NB_ ), MYCOL, DESCA( CSRC_ ),
     $                       NPCOL )
            MP = NUMROC( M+IROFF, DESCA( MB_ ), MYROW, IAROW, NPROW )
            NQ = NUMROC( N+ICOFF, DESCA( NB_ ), MYCOL, IACOL, NPCOL )
            NQ0 = NUMROC( JA+N-1, DESCA( NB_ ), MYCOL, DESCA( CSRC_ ),
     $                    NPCOL )
            LWMIN = MAX( 3, MP + NQ ) + NQ0 + NQ
*
            WORK( 1 ) = DBLE( LWMIN )
            LQUERY = ( LWORK.EQ.-1 )
            IF( LWORK.LT.LWMIN .AND. .NOT.LQUERY )
     $         INFO = -10
         END IF
         IF( LWORK.EQ.-1 ) THEN
            IDUM1( 1 ) = -1
         ELSE
            IDUM1( 1 ) = 1
         END IF
         IDUM2( 1 ) = 10
         CALL PCHK1MAT( M, 1, N, 2, IA, JA, DESCA, 6, 1, IDUM1, IDUM2,
     $                  INFO )
      END IF
*
      IF( INFO.NE.0 ) THEN
         CALL PXERBLA( ICTXT, 'PDGEQPF', -INFO )
         RETURN
      ELSE IF( LQUERY ) THEN
         RETURN
      END IF
*
*     Quick return if possible
*
      IF( M.EQ.0 .OR. N.EQ.0 )
     $   RETURN
*
      CALL INFOG2L( IA, JA, DESCA, NPROW, NPCOL, MYROW, MYCOL, IIA, JJA,
     $              IAROW, IACOL )
      IF( MYROW.EQ.IAROW )
     $   MP = MP - IROFF
      IF( MYCOL.EQ.IACOL )
     $   NQ = NQ - ICOFF
      MN = MIN( M, N )
*
*     Initialize the array of pivots
*
      LDA = DESCA( LLD_ )
      JN = MIN( ICEIL( JA, DESCA( NB_ ) ) * DESCA( NB_ ), JA+N-1 )
      KSTEP  = NPCOL * DESCA( NB_ )
*
      IF( MYCOL.EQ.IACOL ) THEN
*
*        Handle first block separately
*
         JB = JN - JA + 1
         DO 10 LL = JJA, JJA+JB-1
            IPIV( LL ) = JA + LL - JJA
   10    CONTINUE
         KSTART = JN + KSTEP - DESCA( NB_ )
*
*        Loop over remaining block of columns
*
         DO 30 KK = JJA+JB, JJA+NQ-1, DESCA( NB_ )
            KB = MIN( JJA+NQ-KK, DESCA( NB_ ) )
            DO 20 LL = KK, KK+KB-1
               IPIV( LL ) = KSTART+LL-KK+1
   20       CONTINUE
            KSTART = KSTART + KSTEP
   30    CONTINUE
      ELSE
         KSTART = JN + ( MOD( MYCOL-IACOL+NPCOL, NPCOL )-1 )*
     $                        DESCA( NB_ )
         DO 50 KK = JJA, JJA+NQ-1, DESCA( NB_ )
            KB = MIN( JJA+NQ-KK, DESCA( NB_ ) )
            DO 40 LL = KK, KK+KB-1
               IPIV( LL ) = KSTART+LL-KK+1
   40       CONTINUE
            KSTART = KSTART + KSTEP
   50    CONTINUE
      END IF
*
*     Initialize partial column norms, handle first block separately
*
      CALL DESCSET( DESCN, 1, DESCA( N_ ), 1, DESCA( NB_ ), MYROW,
     $              DESCA( CSRC_ ), ICTXT, 1 )
*
      IPN = 1
      IPW = IPN + NQ0 + NQ
      JJ = IPN + JJA - 1
      IF( MYCOL.EQ.IACOL ) THEN
         DO 60 KK = 0, JB-1
            CALL PDNRM2( M, WORK( JJ+KK ), A, IA, JA+KK, DESCA, 1 )
            WORK( NQ+JJ+KK ) = WORK( JJ+KK )
   60    CONTINUE
         JJ = JJ + JB
      END IF
      ICURCOL = MOD( IACOL+1, NPCOL )
*
*     Loop over the remaining blocks of columns
*
      DO 80 J = JN+1, JA+N-1, DESCA( NB_ )
         JB = MIN( JA+N-J, DESCA( NB_ ) )
*
         IF( MYCOL.EQ.ICURCOL ) THEN
            DO 70 KK = 0, JB-1
               CALL PDNRM2( M, WORK( JJ+KK ), A, IA, J+KK, DESCA, 1 )
               WORK( NQ+JJ+KK ) = WORK( JJ+KK )
   70       CONTINUE
            JJ = JJ + JB
         END IF
         ICURCOL = MOD( ICURCOL+1, NPCOL )
   80 CONTINUE
*
*     Compute factorization
*
      RANK = 0
      DO 120 J = JA, JA+MN-1
         I = IA + J - JA
*
         CALL INFOG1L( J, DESCA( NB_ ), NPCOL, MYCOL, DESCA( CSRC_ ),
     $                 JJ, ICURCOL )
         K = JA + N - J
         IF( K.GT.1 ) THEN
            CALL PDAMAX( K, TEMP, PVT, WORK( IPN ), 1, J, DESCN,
     $                   DESCN( M_ ) )
            CALL PDNRM2( M-J+1, TEMP, A, J, PVT, DESCA, 1 )
            CALL DGAMX2D( ICTXT, 'A', ' ', 1, 1, TEMP, 1, 1, 1,
     $                    -1, -1, -1)
            IF(J.EQ.JA) THEN
              IF(ABS(TEMP)<ATOL) THEN
                GOTO 99
              END IF
              A11=ABS(TEMP)
            ELSE
              IF(ABS(TEMP)/A11<RTOL .OR. ABS(TEMP)<ATOL) THEN
                GOTO 99
              END IF
            END IF
         ELSE
            PVT = J
         END IF
         RANK = RANK +1
         JPIV(J)=PVT
         ITEMP=JPERM(J)
         JPERM(J)=JPERM(PVT)
         JPERM(PVT)=ITEMP
         IF( J.NE.PVT ) THEN
            CALL INFOG1L( PVT, DESCA( NB_ ), NPCOL, MYCOL,
     $                    DESCA( CSRC_ ), JJPVT, IPCOL )
            IF( ICURCOL.EQ.IPCOL ) THEN
               IF( MYCOL.EQ.ICURCOL ) THEN
                  CALL DSWAP( MP, A( IIA+(JJ-1)*LDA ), 1,
     $                        A( IIA+(JJPVT-1)*LDA ), 1 )
                  ITEMP = IPIV( JJPVT )
                  IPIV( JJPVT ) = IPIV( JJ )
                  IPIV( JJ ) = ITEMP
                  WORK( IPN+JJPVT-1 ) = WORK( IPN+JJ-1 )
                  WORK( IPN+NQ+JJPVT-1 ) = WORK( IPN+NQ+JJ-1 )
               END IF
            ELSE
               IF( MYCOL.EQ.ICURCOL ) THEN
*
                  CALL DGESD2D( ICTXT, MP, 1, A( IIA+(JJ-1)*LDA ), LDA,
     $                          MYROW, IPCOL )
                  WORK( IPW )   = DBLE( IPIV( JJ ) )
                  WORK( IPW+1 ) = WORK( IPN + JJ - 1 )
                  WORK( IPW+2 ) = WORK( IPN + NQ + JJ - 1 )
                  CALL DGESD2D( ICTXT, 3, 1, WORK( IPW ), 3, MYROW,
     $                          IPCOL )
*
                  CALL DGERV2D( ICTXT, MP, 1, A( IIA+(JJ-1)*LDA ), LDA,
     $                          MYROW, IPCOL )
                  CALL IGERV2D( ICTXT, 1, 1, IPIV( JJ ), 1, MYROW,
     $                          IPCOL )
*
               ELSE IF( MYCOL.EQ.IPCOL ) THEN
*
                  CALL DGESD2D( ICTXT, MP, 1, A( IIA+(JJPVT-1)*LDA ),
     $                          LDA, MYROW, ICURCOL )
                  CALL IGESD2D( ICTXT, 1, 1, IPIV( JJPVT ), 1, MYROW,
     $                          ICURCOL )
*
                  CALL DGERV2D( ICTXT, MP, 1, A( IIA+(JJPVT-1)*LDA ),
     $                          LDA, MYROW, ICURCOL )
                  CALL DGERV2D( ICTXT, 3, 1, WORK( IPW ), 3, MYROW,
     $                          ICURCOL )
                  IPIV( JJPVT ) = IDINT( WORK( IPW ) )
                  WORK( IPN+JJPVT-1 ) = WORK( IPW+1 )
                  WORK( IPN+NQ+JJPVT-1 ) = WORK( IPW+2 )
*
               END IF
*
            END IF
*
         END IF
*
*        Generate elementary reflector H(i)
*
         CALL INFOG1L( I, DESCA( MB_ ), NPROW, MYROW, DESCA( RSRC_ ),
     $                 II, ICURROW )
         IF( DESCA( M_ ).EQ.1 ) THEN
            IF( MYROW.EQ.ICURROW ) THEN
               IF( MYCOL.EQ.ICURCOL ) THEN
                  IOFFA = II+(JJ-1)*DESCA( LLD_ )
                  AJJ = A( IOFFA )
                  CALL DLARFG( 1, AJJ, A( IOFFA ), 1, TAU( JJ ) )
                  IF( N.GT.1 ) THEN
                     ALPHA = ONE - TAU( JJ )
                     CALL DGEBS2D( ICTXT, 'Rowwise', ' ', 1, 1, ALPHA,
     $                                  1 )
                     CALL DSCAL( NQ-JJ, ALPHA, A( IOFFA+DESCA( LLD_ ) ),
     $                           DESCA( LLD_ ) )
                  END IF
                  CALL DGEBS2D( ICTXT, 'Columnwise', ' ', 1, 1,
     $                          TAU( JJ ), 1 )
                  A( IOFFA ) = AJJ
               ELSE
                  IF( N.GT.1 ) THEN
                     CALL DGEBR2D( ICTXT, 'Rowwise', ' ', 1, 1, ALPHA,
     $                             1, ICURROW, ICURCOL )
                     CALL DSCAL( NQ-JJ+1, ALPHA, A( I ), DESCA( LLD_ ) )
                  END IF
               END IF
            ELSE IF( MYCOL.EQ.ICURCOL ) THEN
               CALL DGEBR2D( ICTXT, 'Columnwise', ' ', 1, 1, TAU( JJ ),
     $                       1, ICURROW, ICURCOL )
            END IF
*
         ELSE
*
            CALL PDLARFG( M-J+JA, AJJ, I, J, A, MIN( I+1, IA+M-1 ), J,
     $                    DESCA, 1, TAU )
            IF( J.LT.JA+N-1 ) THEN
*
*              Apply H(i) to A(ia+j-ja:ia+m-1,j+1:ja+n-1) from the left
*
               CALL PDELSET( A, I, J, DESCA, ONE )
               CALL PDLARF( 'Left', M-J+JA, JA+N-1-J, A, I, J, DESCA,
     $                      1, TAU, A, I, J+1, DESCA, WORK( IPW ) )
            END IF
            CALL PDELSET( A, I, J, DESCA, AJJ )
*
         END IF
*
*        Update partial columns norms
*
         IF( MYCOL.EQ.ICURCOL )
     $      JJ = JJ + 1
         IF( MOD( J, DESCA( NB_ ) ).EQ.0 )
     $      ICURCOL = MOD( ICURCOL+1, NPCOL )
         IF( (JJA+NQ-JJ).GT.0 ) THEN
            IF( MYROW.EQ.ICURROW ) THEN
               CALL DGEBS2D( ICTXT, 'Columnwise', ' ', 1, JJA+NQ-JJ,
     $                       A( II+( MIN( JJA+NQ-1, JJ )-1 )*LDA ),
     $                       LDA )
               CALL DCOPY( JJA+NQ-JJ, A( II+( MIN( JJA+NQ-1, JJ )
     $                     -1)*LDA ), LDA, WORK( IPW+MIN( JJA+NQ-1,
     $                    JJ )-1 ), 1 )
            ELSE
               CALL DGEBR2D( ICTXT, 'Columnwise', ' ', JJA+NQ-JJ, 1,
     $                       WORK( IPW+MIN( JJA+NQ-1, JJ )-1 ),
     $                       MAX( 1, NQ ), ICURROW, MYCOL )
            END IF
         END IF
*
         JN = MIN( ICEIL( J+1, DESCA( NB_ ) ) * DESCA( NB_ ),
     $                    JA + N - 1 )
         IF( MYCOL.EQ.ICURCOL ) THEN
            DO 90 LL = JJ-1, JJ + JN - J - 2
               IF( WORK( IPN+LL ).NE.ZERO ) THEN
                  TEMP = ONE-( ABS( WORK( IPW+LL ) ) /
     $                         WORK( IPN+LL ) )**2
                  TEMP = MAX( TEMP, ZERO )
                  TEMP2 = ONE + 0.05D+0*TEMP*
     $                    ( WORK( IPN+LL ) / WORK( IPN+NQ+LL ) )**2
                  IF( TEMP2.EQ.ONE ) THEN
                     IF( IA+M-1.GT.I ) THEN
                        CALL PDNRM2( IA+M-I-1, WORK( IPN+LL ), A, I+1,
     $                               J+LL-JJ+2, DESCA, 1 )
                        WORK( IPN+NQ+LL ) = WORK( IPN+LL )
                     ELSE
                        WORK( IPN+LL ) = ZERO
                        WORK( IPN+NQ+LL ) = ZERO
                     END IF
                  ELSE
                     WORK( IPN+LL ) = WORK( IPN+LL ) * SQRT( TEMP )
                  END IF
               END IF
   90       CONTINUE
            JJ = JJ + JN - J
         END IF
         ICURCOL = MOD( ICURCOL+1, NPCOL )
*
         DO 110 K = JN+1, JA+N-1, DESCA( NB_ )
            KB = MIN( JA+N-K, DESCA( NB_ ) )
*
            IF( MYCOL.EQ.ICURCOL ) THEN
               DO 100 LL = JJ-1, JJ+KB-2
                  IF( WORK( IPN+LL ).NE.ZERO ) THEN
                     TEMP = ONE-( ABS( WORK( IPW+LL ) ) /
     $                            WORK( IPN+LL ) )**2
                     TEMP = MAX( TEMP, ZERO )
                     TEMP2 = ONE + 0.05D+0*TEMP*
     $                     ( WORK( IPN+LL ) / WORK( IPN+NQ+LL ) )**2
                     IF( TEMP2.EQ.ONE ) THEN
                        IF( IA+M-1.GT.I ) THEN
                           CALL PDNRM2( IA+M-I-1, WORK( IPN+LL ), A,
     $                                  I+1, K+LL-JJ+1, DESCA, 1 )
                           WORK( IPN+NQ+LL ) = WORK( IPN+LL )
                        ELSE
                           WORK( IPN+LL ) = ZERO
                           WORK( IPN+NQ+LL ) = ZERO
                        END IF
                     ELSE
                        WORK( IPN+LL ) = WORK( IPN+LL ) * SQRT( TEMP )
                     END IF
                  END IF
  100          CONTINUE
               JJ = JJ + KB
            END IF
            ICURCOL = MOD( ICURCOL+1, NPCOL )
*
  110    CONTINUE
*
  120 CONTINUE
*
      WORK( 1 ) = DBLE( LWMIN )
*
   99 CONTINUE
      RETURN
*
*     End of PDGEQPF
*
      END
