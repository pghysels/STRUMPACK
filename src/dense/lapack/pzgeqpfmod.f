      SUBROUTINE PZGEQPFmod( M, N, A, IA, JA, DESCA, IPIV, TAU, WORK,
     $                    LWORK, RWORK, LRWORK, INFO, JPERM, JPIV, RANK,
     $                    RTOL, ATOL )
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
      INTEGER            IA, JA, INFO, LRWORK, LWORK, M, N, RANK
*     ..
*     .. Array Arguments ..
      INTEGER            DESCA( * ), IPIV( * ), JPERM( * ), JPIV ( * )
      DOUBLE PRECISION   RWORK( * ), RTOL, ATOL
      COMPLEX*16         A( * ), TAU( * ), WORK( * )
*     ..
*
*  Purpose
*  =======
*
*  PZGEQPF computes a QR factorization with column pivoting of a
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
*  A       (local input/local output) COMPLEX*16 pointer into the
*          local memory to an array of dimension (LLD_A, LOCc(JA+N-1)).
*          On entry, the local pieces of the M-by-N distributed matrix
*          sub( A ) which is to be factored. On exit, the elements on
*          and above the diagonal of sub( A ) contain the min(M,N) by N
*          upper trapezoidal matrix R (R is upper triangular if M >= N);
*          the elements below the diagonal, with the array TAU, repre-
*          sent the unitary matrix Q as a product of elementary
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
*  TAU     (local output) COMPLEX*16, array, dimension
*          LOCc(JA+MIN(M,N)-1). This array contains the scalar factors
*          TAU of the elementary reflectors. TAU is tied to the
*          distributed matrix A.
*
*  WORK    (local workspace/local output) COMPLEX*16 array,
*                                                    dimension (LWORK)
*          On exit, WORK(1) returns the minimal and optimal LWORK.
*
*  LWORK   (local or global input) INTEGER
*          The dimension of the array WORK.
*          LWORK is local input and must be at least
*          LWORK >= MAX(3,Mp0 + Nq0).
*
*          If LWORK = -1, then LWORK is global input and a workspace
*          query is assumed; the routine only calculates the minimum
*          and optimal size for all work arrays. Each of these
*          values is returned in the first entry of the corresponding
*          work array, and no error message is issued by PXERBLA.
*
*  RWORK   (local workspace/local output) DOUBLE PRECISION array,
*                                                 dimension (LRWORK)
*          On exit, RWORK(1) returns the minimal and optimal LRWORK.
*
*  LRWORK  (local or global input) INTEGER
*          The dimension of the array RWORK.
*          LRWORK is local input and must be at least
*          LRWORK >= LOCc(JA+N-1)+Nq0.
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
*          If LRWORK = -1, then LRWORK is global input and a workspace
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
*  where tau is a complex scalar, and v is a complex vector with
*  v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in
*  A(ia+i-1:ia+m-1,ja+i-1).
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
     $                   ICURCOL, II, IIA, IOFFA, IPCOL, IROFF, ITEMP,
     $                   J, JB, JJ, JJA, JJPVT, JN, KB, K, KK, KSTART,
     $                   KSTEP, LDA, LL, LRWMIN, LWMIN, MN, MP, MYCOL,
     $                   MYROW, NPCOL, NPROW, NQ, NQ0, PVT
      DOUBLE PRECISION   TEMP, TEMP2, A11
      COMPLEX*16         AJJ, ALPHA
*     ..
*     .. Local Arrays ..
      INTEGER            DESCN( DLEN_ ), IDUM1( 2 ), IDUM2( 2 )
*     ..
*     .. External Subroutines ..
      EXTERNAL           BLACS_GRIDINFO, CHK1MAT, DESCSET, IGERV2D,
     $                   IGESD2D, INFOG1L, INFOG2L, PCHK1MAT, PDAMAX,
     $                   PDZNRM2, PXERBLA, PZELSET,
     $                   PZLARFC, PZLARFG, ZCOPY, ZGEBR2D,
     $                   ZGEBS2D, ZGERV2D, ZGESD2D, ZLARFG,
     $                   ZSWAP
*     ..
*     .. External Functions ..
      INTEGER            ICEIL, INDXG2P, NUMROC
      EXTERNAL           ICEIL, INDXG2P, NUMROC
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, DCMPLX, DCONJG, IDINT, MAX, MIN, MOD, SQRT
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
            LWMIN = MAX( 3, MP + NQ )
            LRWMIN = NQ0 + NQ
*
            WORK( 1 ) = DCMPLX( DBLE( LWMIN ) )
            RWORK( 1 ) = DBLE( LRWMIN )
            LQUERY = ( LWORK.EQ.-1 .OR. LRWORK.EQ.-1 )
            IF( LWORK.LT.LWMIN .AND. .NOT.LQUERY ) THEN
               INFO = -10
            ELSE IF( LRWORK.LT.LRWMIN .AND. .NOT.LQUERY ) THEN
               INFO = -12
            END IF
         END IF
         IF( LWORK.EQ.-1 ) THEN
            IDUM1( 1 ) = -1
         ELSE
            IDUM1( 1 ) = 1
         END IF
         IDUM2( 1 ) = 10
         IF( LRWORK.EQ.-1 ) THEN
            IDUM1( 2 ) = -1
         ELSE
            IDUM1( 2 ) = 1
         END IF
         IDUM2( 2 ) = 12
         CALL PCHK1MAT( M, 1, N, 2, IA, JA, DESCA, 6, 2, IDUM1, IDUM2,
     $                  INFO )
      END IF
*
      IF( INFO.NE.0 ) THEN
         CALL PXERBLA( ICTXT, 'PZGEQPF', -INFO )
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
      JJ = JJA
      IF( MYCOL.EQ.IACOL ) THEN
         DO 60 KK = 0, JB-1
            CALL PDZNRM2( M, RWORK( JJ+KK ), A, IA, JA+KK, DESCA, 1 )
            RWORK( NQ+JJ+KK ) = RWORK( JJ+KK )
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
               CALL PDZNRM2( M, RWORK( JJ+KK ), A, IA, J+KK, DESCA, 1 )
               RWORK( NQ+JJ+KK ) = RWORK( JJ+KK )
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
            CALL PDAMAX( K, TEMP, PVT, RWORK, 1, J, DESCN,
     $                   DESCN( M_ ) )
            CALL PDZNRM2( M-J+1, TEMP, A, J, PVT, DESCA, 1 )
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
                  CALL ZSWAP( MP, A( IIA+(JJ-1)*LDA ), 1,
     $                        A( IIA+(JJPVT-1)*LDA ), 1 )
                  ITEMP = IPIV( JJPVT )
                  IPIV( JJPVT ) = IPIV( JJ )
                  IPIV( JJ ) = ITEMP
                  RWORK( JJPVT ) = RWORK( JJ )
                  RWORK( NQ+JJPVT ) = RWORK( NQ+JJ )
               END IF
            ELSE
               IF( MYCOL.EQ.ICURCOL ) THEN
*
                  CALL ZGESD2D( ICTXT, MP, 1, A( IIA+(JJ-1)*LDA ), LDA,
     $                          MYROW, IPCOL )
                  WORK( 1 ) = DCMPLX( DBLE( IPIV( JJ ) ) )
                  WORK( 2 ) = DCMPLX( RWORK( JJ ) )
                  WORK( 3 ) = DCMPLX( RWORK( JJ + NQ ) )
                  CALL ZGESD2D( ICTXT, 3, 1, WORK, 3, MYROW, IPCOL )
*
                  CALL ZGERV2D( ICTXT, MP, 1, A( IIA+(JJ-1)*LDA ), LDA,
     $                          MYROW, IPCOL )
                  CALL IGERV2D( ICTXT, 1, 1, IPIV( JJ ), 1, MYROW,
     $                          IPCOL )
*
               ELSE IF( MYCOL.EQ.IPCOL ) THEN
*
                  CALL ZGESD2D( ICTXT, MP, 1, A( IIA+(JJPVT-1)*LDA ),
     $                          LDA, MYROW, ICURCOL )
                  CALL IGESD2D( ICTXT, 1, 1, IPIV( JJPVT ), 1, MYROW,
     $                          ICURCOL )
*
                  CALL ZGERV2D( ICTXT, MP, 1, A( IIA+(JJPVT-1)*LDA ),
     $                          LDA, MYROW, ICURCOL )
                  CALL ZGERV2D( ICTXT, 3, 1, WORK, 3, MYROW, ICURCOL )
                  IPIV( JJPVT ) = IDINT( DBLE( WORK( 1 ) ) )
                  RWORK( JJPVT ) = DBLE( WORK( 2 ) )
                  RWORK( JJPVT+NQ ) = DBLE( WORK( 3 ) )
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
                  CALL ZLARFG( 1, AJJ, A( IOFFA ), 1, TAU( JJ ) )
                  IF( N.GT.1 ) THEN
                     ALPHA = DCMPLX( ONE ) - DCONJG( TAU( JJ ) )
                     CALL ZGEBS2D( ICTXT, 'Rowwise', ' ', 1, 1, ALPHA,
     $                                  1 )
                     CALL ZSCAL( NQ-JJ, ALPHA, A( IOFFA+DESCA( LLD_ ) ),
     $                           DESCA( LLD_ ) )
                  END IF
                  CALL ZGEBS2D( ICTXT, 'Columnwise', ' ', 1, 1,
     $                          TAU( JJ ), 1 )
                  A( IOFFA ) = AJJ
               ELSE
                  IF( N.GT.1 ) THEN
                     CALL ZGEBR2D( ICTXT, 'Rowwise', ' ', 1, 1, ALPHA,
     $                             1, ICURROW, ICURCOL )
                     CALL ZSCAL( NQ-JJ+1, ALPHA, A( I ), DESCA( LLD_ ) )
                  END IF
               END IF
            ELSE IF( MYCOL.EQ.ICURCOL ) THEN
               CALL ZGEBR2D( ICTXT, 'Columnwise', ' ', 1, 1, TAU( JJ ),
     $                       1, ICURROW, ICURCOL )
            END IF
*
         ELSE
*
            CALL PZLARFG( M-J+JA, AJJ, I, J, A, MIN( I+1, IA+M-1 ), J,
     $                    DESCA, 1, TAU )
            IF( J.LT.JA+N-1 ) THEN
*
*              Apply H(i) to A(ia+j-ja:ia+m-1,j+1:ja+n-1) from the left
*
               CALL PZELSET( A, I, J, DESCA, DCMPLX( ONE ) )
               CALL PZLARFC( 'Left', M-J+JA, JA+N-1-J, A, I, J, DESCA,
     $                       1, TAU, A, I, J+1, DESCA, WORK )
            END IF
            CALL PZELSET( A, I, J, DESCA, AJJ )
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
               CALL ZGEBS2D( ICTXT, 'Columnwise', ' ', 1, JJA+NQ-JJ,
     $                       A( II+( MIN( JJA+NQ-1, JJ )-1 )*LDA ),
     $                       LDA )
               CALL ZCOPY( JJA+NQ-JJ, A( II+( MIN( JJA+NQ-1, JJ )
     $                     -1)*LDA ), LDA, WORK( MIN( JJA+NQ-1, JJ ) ),
     $                     1 )
            ELSE
               CALL ZGEBR2D( ICTXT, 'Columnwise', ' ', JJA+NQ-JJ, 1,
     $                       WORK( MIN( JJA+NQ-1, JJ ) ), MAX( 1, NQ ),
     $                       ICURROW, MYCOL )
            END IF
         END IF
*
         JN = MIN( ICEIL( J+1, DESCA( NB_ ) ) * DESCA( NB_ ),
     $                    JA + N - 1 )
         IF( MYCOL.EQ.ICURCOL ) THEN
            DO 90 LL = JJ, JJ + JN - J - 1
               IF( RWORK( LL ).NE.ZERO ) THEN
                  TEMP = ONE-( ABS( WORK( LL ) ) / RWORK( LL ) )**2
                  TEMP = MAX( TEMP, ZERO )
                  TEMP2 = ONE + 0.05D+0*TEMP*
     $                    ( RWORK( LL ) / RWORK( NQ+LL ) )**2
                  IF( TEMP2.EQ.ONE ) THEN
                     IF( IA+M-1.GT.I ) THEN
                        CALL PDZNRM2( IA+M-I-1, RWORK( LL ), A,
     $                                I+1, J+LL-JJ, DESCA, 1 )
                        RWORK( NQ+LL ) = RWORK( LL )
                     ELSE
                        RWORK( LL ) = ZERO
                        RWORK( NQ+LL ) = ZERO
                     END IF
                  ELSE
                     RWORK( LL ) = RWORK( LL ) * SQRT( TEMP )
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
               DO 100 LL = JJ, JJ+KB-1
                  IF( RWORK(LL).NE.ZERO ) THEN
                     TEMP = ONE-( ABS( WORK( LL ) ) / RWORK( LL ) )**2
                     TEMP = MAX( TEMP, ZERO )
                     TEMP2 = ONE + 0.05D+0*TEMP*
     $                       ( RWORK( LL ) / RWORK( NQ+LL ) )**2
                     IF( TEMP2.EQ.ONE ) THEN
                        IF( IA+M-1.GT.I ) THEN
                           CALL PDZNRM2( IA+M-I-1, RWORK( LL ), A,
     $                                   I+1, K+LL-JJ, DESCA, 1 )
                           RWORK( NQ+LL ) = RWORK( LL )
                        ELSE
                           RWORK( LL ) = ZERO
                           RWORK( NQ+LL ) = ZERO
                        END IF
                     ELSE
                        RWORK( LL ) = RWORK( LL ) * SQRT( TEMP )
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
      WORK( 1 ) = DCMPLX( DBLE( LWMIN ) )
      RWORK( 1 ) = DBLE( LRWMIN )
*
   99 CONTINUE
      RETURN
*
*     End of PZGEQPF
*
      END
