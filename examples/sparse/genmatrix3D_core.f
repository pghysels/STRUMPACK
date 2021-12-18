! Artem, this is the generator for Operto's matrices. There is
! everything you need in comments below. Pay attention : the actual size
! of the mesh is (n1+16)x(n2+16)x(n3+16). Just look below for more info.
! This is a COMPLEX code. Their application are single precision.
!
!      

!     #############################################################################
!     
!     PROGRAM GENMATRIX3D
!     
!     GeoSciences Azur / Seiscope 3D matrix generator.
!     
!     Program to build finite-difference frequency-domain impedance matrix
!     using a 27-point stencil. This matrix results from the discretization
!     of the 3D acoustic wave equation in the frequency domain.
!     
!     Input parameters: 
!     Dimension of the 3D grid: n1 n2 n3. n1 is the fast index. n3 is the slow index
!     set these parameters in the input ascii file fwm.par
!     
!     Output files (direct access binary files):
!     - fmatrix.bin(ne): single-precision complex coefficients
!     - fjcn.bin(ne): column indices of complex coefficients
!     - firn.bin(ne): row indices of complex coefficients
!     
!     Number of unknowns (nn): n1 x n2 x n3
!     Number of non zero complex coefficients (ne): 27 x n1 x n2 x n3
!     
!     #############################################################################
!     Modified version by E.A. on 12 March 2007.
!     Initially the main part of a-self contained binary ;
!     this part is a driver to generate a 3D matrix from GeoSciences Azur.
!     
!     The API is simple:
!     A call to SUBMATANAL allows the user to compute the
!     number of nonzeros and then to allocate irn, jcn and A.
!     Then a call to GENMATRIX3D fills irn, jcn and A.
!     
!     #############################################################################
      integer function kkindex(i1,i2,i3,n1,n2)
      integer i1,i2,i3,n1,n2
      kkindex=(i3-1)*n1*n2+(i2-1)*n1+i1
      return
      end
    
      subroutine GENMATRIX3D_ANAL(n1,n2,n3,npml,nne,ne,FROMFILE
     &     ,DATAFILE)
      integer,intent(in):: n1,n2,n3,npml
      integer,intent(out):: nne,ne
      integer :: n1e,n2e,n3e
      LOGICAL,intent(in) :: FROMFILE
      CHARACTER(len=300),intent(in)::DATAFILE
c     
      IF(FROMFILE) THEN
         CALL subreadpar3D1_0(n1,n2,n3,npml,DATAFILE)
      ENDIF
      n1e=n1+2*npml
      n2e=n2+2*npml
      n3e=n3+2*npml
      nne=n1e*n2e*n3e
      ne=27*n1e*n2e*n3e         !estimation of the number of non zero coefficients in A

      write(*,*) "N = ", nne
      write(*,*) "NNZ = ", ne, " estimated"

      return
      end subroutine GENMATRIX3D_ANAL
      
      SUBROUTINE GENMATRIX3D(irn,jcn,mat,n1,n2,n3,npml,nnz
     &     ,FROMFILE,DATAFILE)
      
! #   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
! #
! #                                                         PURPOSE
! #
! #   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
! #
! # Build the complex-valued impedance matrix resulting A from the finite-difference discretization of the heterogeneous Helmholtz equation
! # that is the second-order visco-acoustic time-harmonic wave equation for pressure p. The aim is the modeling of visco-acoustic
! # wave propagation in a 3D visco-acoustic medium parametrized by wavespeed, density and quality factor (the inverse of the attenuation).
! # This amounts to solve
! #                                                        A . p  = s
! # where p is the monochromatic complex-valued wavefield and A is the impedance matrix the coefficients of which depends on the chosen frequency and
! # medium properties. The matrix A is built for an infinite medium. This implies that the input grid is augmented with PML absorbing layers.
! # If the input grid provided by the user is of dimension (n1,n2,n3), the computational grid will be of dimension (n1 + 2*npml, n2+2*npml, n3+2*npml) where
! # npml is the number of grid points in the PML along one direction. The number of rows / columns in the matrix A (i.e., the number of unknowns of the system)
! # is (n1 + 2*npml)x(n2+2*npml)x(n3+2*npml).
! # The numerical bandwidth of the matrix is O(n1 x n2). The matrix has a symmetric pattern but is not symmetric because of the absorbing boundary conditions.
! # The number of non zero coefficients per row is 27. The Helmholtz equation is discretized with the so-called 27-point mixed-grid stencil which is a combination
! # of second-order accurate stencils. This implies that the spatial support of the stencil spans over 2 grid interval and the numerical bandwidth of the matrix
! # is O(n1 x n2).
! # The medium is homogeneous, is parametrized by wave speed (4000 m/s), density (1 kg/m^3) and quality factor (10000, no attenuation) and the medium properties
! # are set in the program (vel0, rho0, qf0). Frequency is 4 Hz. The grid interval dz is computed in the program such that dz corresponds to 4 grid point per wavelength.
! # Use program genmatrix.V1 for more generic (homogeneous / heterogeneous) input media.
! # A complex valued bulk modulus is computed from velocity, density and attenuation: mu = rho x velc**2 where velc is a complex-valued velocity given by
! #                                           velc = vel x (1 - i 1/2Q) where vel is the real-valued velocity and Q is the quality factor.
! # The complex-valued bulk modulus and the density are the two medium parameters inputed in subroutine subevalmatrix.f90 which builds the impedance matrix.
! # A rhs vector is built for a point source located in the middle of the grid (only one non zero coefficient). This rhs vector is written on disk in dense format.
! # If the user wants to compute and check the solution of system A . p = s, he can use this rhs vector.
! # The solution p (the Green function for pressure) should show a monochromatric spherical wavefield expanding from the source position with a decreasing amplitude.
! # The wavelelength (the spatial period of the wavelfield) should correspond to c0 / freq where c0 is the homogeneous velocity and freq is the frequency.
! # The amplitude decay should scale with 1/r where r is the propagation distance if no attenuation is used for modeling (Q = 10000).
! # This solution can be validated against an analytical solution.
! #
! #
! #   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
! #
! #                                                       INPUTS/OUTPUTS
! #
! #   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
! #
! # INPUT PARAMETERS / FILES: input parameters are provided on 1 line of ascii file fwm.par
! #
! # The list of input parameters are:
! #
! # Line 1: n1[int] n2[int] n3[int]
! #
! # n1 n2 n3 [INT INT INT]: dimension of the 3D FD grid (n1 is the fast dimension, n3 is the slow dimension in the fortran sense).
! # npml [INT]: the number of grid points in the x, y, z directions in the PML layers. PML are absorbing layers added
! # to the 6 faces of the input 3D grid. They are used to mimick an infinite medium from a finite computational grid.
! # npml=8 is a good pragmatical value if a discretization rule of 4 grid point per wavelength is used.
! # The real size of the FD grid with PML layers will be (n1 + 2xnpml) x (n2 + 2xnpml) x (n3 + 2xnpml)
! #
! # OUTPUT PARAMETERS / FILES
! # - NNZ [INT]: Number of non zero coefficients in the matrix
! # - NNE [INT]: Numner of rows/columns in the matrix
! #
! #      @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
! #      @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
! #      @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
! #      @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
!
      IMPLICIT NONE
      REAL,ALLOCATABLE :: rho(:)
      COMPLEX,ALLOCATABLE :: mu(:)

!     DAMPING FUNCTIONS FOR PML

      COMPLEX,ALLOCATABLE :: d1(:),d2(:),d3(:)
      COMPLEX,ALLOCATABLE :: d1b(:),d2b(:),d3b(:)

      INTEGER :: nn,n1,n2,n3,nne,n1e,n2e,n3e,ne,npml,nnz,nne2
      INTEGER :: ivel,irho,iqf
      REAL :: pi,dz,dz2,freq,omega,apml,vel0,rho0,qf0
      REAL ::  c,d,e,f,w1,w2,w3,wavelength

      COMPLEX :: omegac,ci,velc
      LOGICAL,intent(in) :: FROMFILE
      CHARACTER(len=300),intent(in)::DATAFILE
      CHARACTER(LEN=160) :: namevel,namerho,nameqf
      
      INTEGER :: irn(nnz)
      INTEGER :: jcn(nnz)
      COMPLEX :: mat(nnz)

! #####################################################################################
!     MEDIUM PROPERTIES AND MODELED FREQUENCIES (vel0, rho0, qf0, freq can be modified; the grid interval will
!     be adapted accordingly to satisfy the discretization rule of 4 grid points per wavelength)
! #####################################################################################

      IF(FROMFILE) THEN
         CALL subreadpar3D1_1(ivel,irho,iqf,n1,n2,n3,npml,dz,freq,
     &        vel0,rho0,qf0,namevel,namerho,nameqf,DATAFILE)
      ELSE
         ivel=-999
         irho=-999
         iqf=-999
         vel0=1500.             !Vp
         rho0=1.                !rho
         qf0=10000              !Q
         freq=10.                !Frequency (Hz)
      ENDIF
! #####################################################################################
!     MIXED-GRID STENCIL WEIGHTS (OPERTO ET AL., GEOPHYSICS, 2007)
!     the following coefficients were optimized for 4 grid points per wavelength.
!     Therefore, they differ from those derived in Operto et al. (2007)

      c=0.5915900
      d=4.9653493E-02
      e=5.1085097E-03
      f=6.1483691E-03
      w1=8.8075437E-02
      w2=0.8266806
      w3=8.5243940E-02

! #####################################################################################
!     SET THE GRID INTERVAL ACCORDING TO THE ACCURACY OF THE MIXED-GRID STENCIL (4 GRID POINTS PER WAVELENGTH)

      IF((.NOT.FROMFILE).OR.(FROMFILE.AND.(ivel.eq.0))) THEN
         wavelength=vel0/freq
         dz=wavelength/15.
      ENDIF

      WRITE(*,*) 'GRID INTERVAL (M) = ',dz

!####################################################################
!     SET CONSTANTS
!     omega: angular frequency
!     omegac: complex-valued frequency used to damp seismic wavefield in time in seismic imaging (not used here; imaginaty part niil)
!     apml: damping coefficient for PML

      pi=3.1415926536
      ci=cmplx(0.,1.)
      dz2=dz*dz
      omega=2.*pi*freq
      omegac=cmplx(omega,0.)
      
      apml=90.

!####################################################################
!     SET GRID DIMENSIONS

      n1e=n1+2*npml
      n2e=n2+2*npml
      n3e=n3+2*npml
      nne=n1e*n2e*n3e
      nne2=(n1e+2)*(n2e+2)*(n3e+2)
      nn=n1*n2*n3
      ne=27*n1e*n2e*n3e         !estimation of the number of non zero coefficients in A

      WRITE(*,*) "NUMBER OF UNKNOWNS IN THE INPUT"
     &     //" GRID (nn = n1 x n2 x n3): ",nn
      WRITE(*,*) "NUMBER OF UNKNOWNS IN THE AUGMENTED"
     &     //"GRID WITH PML (nne = n1e x n2e x n3e): ",nne
      WRITE(*,*) "N1E N2E N3E = ",n1e,n2e,n3e

!     NNE2 IS THE SIZE OF THE VECTOR BUFFER FOR RHO AND MU USED TO BUILD A (ONE EXTRA POINT
!     IS ADDED AT EACH END OF EACH DIMENSION TO FACILITATE THE COMPUTATION OF THE DERIVATIVE
!     IN SUBEVALMATRIX. THE NUMBER OF UNKNOWNS CORRESPONDING TO A IS NNE NOT NNE2.

!####################################################################
!     BUILD DENSITY AND BULK MODULUS BUFFER VECTORS REQUIRED TO BUILD A

      ALLOCATE ( mu (nne2) )
      ALLOCATE ( rho (nne2) )

      IF(FROMFILE) THEN
         CALL subrhomu(ivel,irho,iqf,vel0,rho0,
     &        qf0,namevel,namerho,nameqf,n1,n2,n3,npml,nn,nne2,rho,mu)
      ELSE
         velc=vel0*(1.-0.5*ci/qf0) !COMPLEX-VALUED WAVESPEED
         rho(:)=rho0            !DENSITY
         mu(:)=rho0*velc*velc   !BULK MODULUS
      ENDIF

!####################################################################
!     COMPUTE DAMPING FUNCTIONS FOR PML ABSORBING BOUNDARY CONDITIONS

      ALLOCATE (d1(n1e+2))
      ALLOCATE (d2(n2e+2))
      ALLOCATE (d3(n3e+2))
      ALLOCATE (d1b(n1e+2))
      ALLOCATE (d2b(n2e+2))
      ALLOCATE (d3b(n3e+2))

      CALL subdamp(d1,d1b,n1e,npml,dz,apml,omega)
      CALL subdamp(d2,d2b,n2e,npml,dz,apml,omega)
      CALL subdamp(d3,d3b,n3e,npml,dz,apml,omega)

!####################################################################
! BUILD IMPEDANCE MATRIX A
             
      CALL subevalmatrix0(omegac,mu,rho,ne,irn,
     &     jcn, mat,
     &     n1e,n2e,n3e,d1,d2,d3,d1b,d2b,d3b,dz,c,d,e,f,w1,w2,w3,nnz)

!     NNZ: REAL NUMBER OF NON ZERO COEFFICIENTS IN THE IMPEDANCE MATRIX
      WRITE(*,*) 'NUMBER OF NON ZERO COEFFICIENTS IN A (NNZ) = ',nnz

      DEALLOCATE(mu)
      DEALLOCATE(rho)
      DEALLOCATE(d1)
      DEALLOCATE(d2)
      DEALLOCATE(d3)
      DEALLOCATE(d1b)
      DEALLOCATE(d2b)
      DEALLOCATE(d3b)

      WRITE(*,*) 'END OF GENMATRIX3D'
      
      END SUBROUTINE GENMATRIX3D
c     
      subroutine subdamp(damp,dampb,n,npml,dz,a,omega)
      implicit none
      real :: pi,pi2,xpml,xmax,x,xb,eps,epsb,a,dz,omega
      integer :: i,n,npml
      complex damp(0:n+1),dampb(0:n+1)
      complex ci

      pi=3.14159265
      pi2=pi/2
      ci=cmplx(0.,1.)

      do i=0,n+1
         damp(i)=1.
         dampb(i)=1.
      end do

      xpml=float(npml)*dz
      xmax=float(n-1)*dz

      do i=1,npml
         x=float(i-1)*dz
         xb=float(i-1)*dz+0.5*dz
         eps=a*(1.-cos((xpml-x)*pi2/xpml))
         epsb=a*(1.-cos((xpml-xb)*pi2/xpml))
         damp(i)=1./(1.+ci*eps/omega)
         dampb(i)=1./(1.+ci*epsb/omega)
         damp(n-i+1)=damp(i)
      end do

      damp(0)=damp(1)
      damp(n+1)=damp(n)

      do i=1,npml+1
         xb=xmax+0.5*dz-float(i-1)*dz
         epsb=a*(1.-cos((xb-(xmax-xpml))*pi2/xpml))
         dampb(n-i+1)=1./(1.+ci*epsb/omega)
      end do

      dampb(0)=dampb(1)
      dampb(n+1)=dampb(n)

!     ----------------------------------------------------------
!     DEBUG
!     write(*,*) "******************"
!     do i=0,n+1
!     write(*,*) "damp(i)",i,damp(i)
!     end do
!     write(*,*) "**************"
!     do i=0,n+1
!     write(*,*) "dampb(i)",i,dampb(i)
!     end do
!     ----------------------------------------------------------

      return
      end subroutine subdamp
      

      subroutine subevalmatrix0(omegac,mu,rho,ne,irn,icn,mat,n1,n2,n3,
     &     d1,d2,d3,d1b,d2b,d3b,h,c,d,e,f,w1,w2,w3,nnz)
      implicit none

      real c,d,e,f,w1,w2,w3

      integer ne,i1,i2,i3,n1,n2,n3
      integer l,l1,l2,l3,k,nn,nnz
      integer irn(ne),icn(ne)

      complex omegac,mu(0:n1+1,0:n2+1,0:n3+1)
      complex mat(ne)
      
      real rho(0:n1+1,0:n2+1,0:n3+1),omega2,h,h2
      real rm0m,r0m0,r00m,r00p,r0pp,r0mm,r0pm,r0p0,r0mp
      real rm00,rpp0,rp00,rmm0,rm0p,rmpm
      real rmmm,rmp0,rmmp,rmpp
      real rpm0,rp0m,rp0p,rpmm,rpmp,rppm,rppp,r000,w2u,w3u
      complex d1(0:n1+1),d2(0:n2+1),d3(0:n3+1)
      complex d1b(0:n1+1),d2b(0:n2+1),d3b(0:n3+1)

      external kkindex
      integer kkindex

      omega2=real(omegac)*real(omegac)
      h2=1./(h*h)

!     Allocate memory for impedance matrix

      irn(:)=0
      icn(:)=0
      mat(:)=cmplx(0.,0.)
      
      write(*,*) 'N1 N2 N3 = ',n1,n2,n3
      write(*,*) 'c d e f w1 w2 w3 = ',c,d,e,f,w1,w2,w3

      w2u=w2/3.
      w3u=w3/4.
      write(*,*) 'w2u w3u = ',w2u,w3u

      nnz=0
      nn=n1*n2*n3

!     Loop over rows of impedance matrix

      do i3=1,n3
         do i2=1,n2
            do i1=1,n1


               r000=1./rho(i1,i2,i3)  
               
               rp00=0.5*(1./rho(i1,i2,i3)+1./rho(i1+1,i2,i3))
               r0p0=0.5*(1./rho(i1,i2,i3)+1./rho(i1,i2+1,i3))
               r00p=0.5*(1./rho(i1,i2,i3)+1./rho(i1,i2,i3+1))
               rm00=0.5*(1./rho(i1,i2,i3)+1./rho(i1-1,i2,i3))
               r0m0=0.5*(1./rho(i1,i2,i3)+1./rho(i1,i2-1,i3))
               r00m=0.5*(1./rho(i1,i2,i3)+1./rho(i1,i2,i3-1))

               rpp0=0.25*(1./rho(i1,i2,i3)+1./rho(i1+1,i2+1,i3)+1./
     &              rho(i1+1,i2,i3)+1./rho(i1,i2+1,i3))
               r0pp=0.25*(1./rho(i1,i2,i3)+1./rho(i1,i2+1,i3+1)+1./
     &              rho(i1,i2+1,i3)+1./rho(i1,i2,i3+1))
               rp0p=0.25*(1./rho(i1,i2,i3)+1./rho(i1+1,i2,i3+1)+1./
     &              rho(i1+1,i2,i3)+1./rho(i1,i2,i3+1))
               rmm0=0.25*(1./rho(i1,i2,i3)+1./rho(i1-1,i2-1,i3)+1./
     &              rho(i1-1,i2,i3)+1./rho(i1,i2-1,i3))
               r0mm=0.25*(1./rho(i1,i2,i3)+1./rho(i1,i2-1,i3-1)+1./
     &              rho(i1,i2-1,i3)+1./rho(i1,i2,i3-1))
               rm0m=0.25*(1./rho(i1,i2,i3)+1./rho(i1-1,i2,i3-1)+1./
     &              rho(i1-1,i2,i3)+1./rho(i1,i2,i3-1))
               rpm0=0.25*(1./rho(i1,i2,i3)+1./rho(i1+1,i2-1,i3)+1./
     &              rho(i1+1,i2,i3)+1./rho(i1,i2-1,i3))
               r0pm=0.25*(1./rho(i1,i2,i3)+1./rho(i1,i2+1,i3-1)+1./
     &              rho(i1,i2+1,i3)+1./rho(i1,i2,i3-1))
               rp0m=0.25*(1./rho(i1,i2,i3)+1./rho(i1+1,i2,i3-1)+1./
     &              rho(i1+1,i2,i3)+1./rho(i1,i2,i3-1))
               rmp0=0.25*(1./rho(i1,i2,i3)+1./rho(i1-1,i2+1,i3)+1./
     &              rho(i1-1,i2,i3)+1./rho(i1,i2+1,i3))
               r0mp=0.25*(1./rho(i1,i2,i3)+1./rho(i1,i2-1,i3+1)+1./
     &              rho(i1,i2-1,i3)+1./rho(i1,i2,i3+1))
               rm0p=0.25*(1./rho(i1,i2,i3)+1./rho(i1-1,i2,i3+1)+1./
     &              rho(i1-1,i2,i3)+1./rho(i1,i2,i3+1))

               rppp=0.125*(1./rho(i1,i2,i3)+1./rho(i1+1,i2+1,i3+1)+1./
     &              rho(i1+1,i2,i3)+1./rho(i1,i2+1,i3)+1./
     &              rho(i1,i2,i3+1) +1./rho(i1+1,i2+1,i3)+1./
     &              rho(i1,i2+1,i3+1)+1./rho(i1+1,i2,i3+1))
               rmmm=0.125*(1./rho(i1,i2,i3)+1./rho(i1-1,i2-1,i3-1)+1./
     &              rho(i1-1,i2,i3)+1./rho(i1,i2-1,i3)+1./
     &              rho(i1,i2,i3-1) +1./rho(i1-1,i2-1,i3)+1./
     &              rho(i1,i2-1,i3-1)+1./rho(i1-1,i2,i3-1))
               rmpp=0.125*(1./rho(i1,i2,i3)+1./rho(i1-1,i2+1,i3+1)+1./
     &              rho(i1-1,i2,i3)+1./rho(i1,i2+1,i3)+1./
     &              rho(i1,i2,i3+1) +1./rho(i1-1,i2+1,i3)+1./
     &              rho(i1,i2+1,i3+1)+1./rho(i1-1,i2,i3+1))
               rpmp=0.125*(1./rho(i1,i2,i3)+1./rho(i1+1,i2-1,i3+1)+1./
     &              rho(i1+1,i2,i3)+1./rho(i1,i2-1,i3)+1./
     &              rho(i1,i2,i3+1) +1./rho(i1+1,i2-1,i3)+1./
     &              rho(i1,i2-1,i3+1)+1./rho(i1+1,i2,i3+1))
               rppm=0.125*(1./rho(i1,i2,i3)+1./rho(i1+1,i2+1,i3-1)+1./
     &              rho(i1+1,i2,i3)+1./rho(i1,i2+1,i3)+1./
     &              rho(i1,i2,i3-1) +1./rho(i1+1,i2+1,i3)+1./
     &              rho(i1,i2+1,i3-1)+1./rho(i1+1,i2,i3-1))
               rpmm=0.125*(1./rho(i1,i2,i3)+1./rho(i1+1,i2-1,i3-1)+1./
     &              rho(i1+1,i2,i3)+1./rho(i1,i2-1,i3)+1./
     &              rho(i1,i2,i3-1) +1./rho(i1+1,i2-1,i3)+1./
     &              rho(i1,i2-1,i3-1)+1./rho(i1+1,i2,i3-1))
               rmpm=0.125*(1./rho(i1,i2,i3)+1./rho(i1-1,i2+1,i3-1)+1./
     &              rho(i1-1,i2,i3)+1./rho(i1,i2+1,i3)+1./
     &              rho(i1,i2,i3-1) +1./rho(i1-1,i2+1,i3)+1./
     &              rho(i1,i2+1,i3-1)+1./rho(i1-1,i2,i3-1))
               rmmp=0.125*(1./rho(i1,i2,i3)+1./rho(i1-1,i2-1,i3+1)+1./
     &              rho(i1-1,i2,i3)+1./rho(i1,i2-1,i3)+1./
     &              rho(i1,i2,i3+1) +1./rho(i1-1,i2-1,i3)+1./
     &              rho(i1,i2-1,i3+1)+1./rho(i1-1,i2,i3+1))


!     k is the row number; l is the column number

               k=kkindex(i1,i2,i3,n1,n2)

!     ---------------------------------------------------------

!     1 node

               l=k

!     
!     Node 000
!     
               nnz=nnz+1
               irn(nnz)=k
               icn(nnz)=l
!     
               mat(nnz)=
!     
!     Mass term
!     
     &              c*omega2/mu(i1,i2,i3)
!     R1
     &              -w1*h2*(
     &              d2(i2)*(d2b(i2)*r0p0+d2b(i2-1)*r0m0)
     &              +d3(i3)*(d3b(i3)*r00p+d3b(i3-1)*r00m)
     &              +d1(i1)*(d1b(i1)*rp00+d1b(i1-1)*rm00))
!     
!     R2, R3 and R4
!     
!     R2
     &              -w2u*h2*(
!     
     &              0.25*(
     &              d2(i2)*(d2b(i2)*r0pp+d2b(i2-1)*r0mm 
     &              +d2b(i2-1)*r0mp+d2b(i2)*r0pm)
     &              +d3(i3)*(d3b(i3)*r0pp+d3b(i3-1)*r0mm
     &              +d3b(i3-1)*r0pm+d3b(i3)*r0mp))
     &              +d1(i1)*(d1b(i1)*rp00+d1b(i1-1)*rm00)
!     R3
     &              +0.25*(
     &              d1(i1)*(d1b(i1)*rp0p+d1b(i1-1)*rm0m
     &              +d1b(i1-1)*rm0p+d1b(i1)*rp0m)
     &              +d3(i3)*(d3b(i3)*rp0p+d3b(i3-1)*rm0m
     &              +d3b(i3-1)*rp0m+d3b(i3)*rm0p))
     &              +d2(i2)*(d2b(i2)*r0p0+d2b(i2-1)*r0m0)
!     R4
     &              +0.25*(
     &              d2(i2)*(d2b(i2)*rpp0+d2b(i2-1)*rmm0
     &              +d2b(i2-1)*rpm0+d2b(i2)*rmp0)
     &              +d1(i1)*(d1b(i1)*rpp0+d1b(i1-1)*rmm0
     &              +d1b(i1-1)*rmp0+d1b(i1)*rpm0))
     &              +d3(i3)*(d3b(i3)*r00p+d3b(i3-1)*r00m)
     &              )
!     
!     B1, B2, B3 and B4
!     
     &              -w3u*0.5*h2*(
!     
     &              d2(i2)*(d2b(i2)*rmpp+d2b(i2-1)*rpmm+d2b(i2)*rppm+
     &              d2b(i2-1)*rmmp)
     &              +d3(i3)*(d3b(i3)*rppp+d3b(i3-1)*rmmm+d3b(i3)*rmmp+
     &              d3b(i3-1)*rppm)
     &              +d1(i1)*(d1b(i1)*rppp+d1b(i1-1)*rmmm+d1b(i1)*rpmm+
     &              d1b(i1-1)*rmpp)
!     
     &              +d2(i2)*(d2b(i2)*rppp+d2b(i2-1)*rmmm+d2b(i2)*rmpm+
     &              d2b(i2-1)*rpmp)
     &              +d3(i3)*(d3b(i3)*rmpp+d3b(i3-1)*rpmm+d3b(i3)*rpmp+
     &              d3b(i3-1)*rmpm)
     &              +d1(i1)*(d1b(i1)*rppm+d1b(i1-1)*rmmp+d1b(i1)*rpmp+
     &              d1b(i1-1)*rmpm)
!     
     &          ) 


!     ---------------------------------------------------------

!     6 nodes
!     
!     Node 100
!     
               l=kkindex(i1+1,i2,i3,n1,n2)
!     if (l.lt.1.or.l.gt.nn) go to 200
               l1=i1+1
               l2=i2
               l3=i3
               if (l1.lt.1.or.l1.gt.n1.or.l2.lt.1.or.l2.gt.n2.or.l3.lt.1
     &              .or.l3.gt.n3) go to 200
!     
               nnz=nnz+1
               irn(nnz)=k
               icn(nnz)=l
!     
               mat(nnz)=                                                            
!     
!     Mass term
!     
     &              d*omega2/mu(i1+1,i2,i3)                                          
!     
!     R1
!     
     &              +w1*h2*d1(i1)*d1b(i1)*r00p                                   
!     
!     R2, R3, R4
!     
     &              +w2u*0.25*h2*(                                                    
     &              d1(i1)*d1b(i1)*(rp0m+rp0p+rpp0+rpm0)                        
     &              -d3(i3)*(d3b(i3)*rp0p+d3b(i3-1)*rp0m)                         
     &              -d2(i2)*(d2b(i2)*rpp0+d2b(i2-1)*rpm0))                           
     &              +w2u*h2*d1(i1)*d1b(i1)*rp00                                     
!     
!     B1, B2, B3, B4
!     
     &              +w3u*0.5*h2*d1(i1)*d1b(i1)*(rppp+rpmm+rppm+rpmp)                    
!     
!     ---------------------------------------------------------
!     
!     Node 010
!     
 200           l=kkindex(i1,i2+1,i3,n1,n2)
               l1=i1
               l2=i2+1
               l3=i3
               if (l1.lt.1.or.l1.gt.n1.or.l2.lt.1.or.l2.gt.n2.or.l3.lt.1
     &              .or.l3.gt.n3) go to 201
!     if (l.lt.1.or.l.gt.nn) go to 201
!     
               nnz=nnz+1
               irn(nnz)=k
               icn(nnz)=l
!     
               mat(nnz)=
!     
!     Mass term
!     
     &              d*omega2/mu(i1,i2+1,i3)                                     
!     
!     R1
!     
     &              +w1*h2*d2(i2)*d2b(i2)*r0p0                                    
!     
!     R2, R3, R4
!     
     &              +w2u*0.25*h2*(                                                   
     &              d2(i2)*d2b(i2)*(r0pp+r0pm+rmp0+rpp0)                           
     &              -d3(i3)*(d3b(i3)*r0pp+d3b(i3-1)*r0pm)                          
     &              -d1(i1)*(d1b(i1)*rpp0+d1b(i1-1)*rmp0))                          
     &              +w2u*h2*d2(i2)*d2b(i2)*r0p0                                    
!     
!     B1, B2, B3, B4
!     
     &              +w3u*0.5*h2*d2(i2)*d2b(i2)*(rppp+rmpm+rmpp+rppm)                  
!     
!     ---------------------------------------------------------
!     
!     Node 001
!     
 201           l=kkindex(i1,i2,i3+1,n1,n2)
               l1=i1
               l2=i2
               l3=i3+1
               if (l1.lt.1.or.l1.gt.n1.or.l2.lt.1.or.l2.gt.n2.or.l3.lt.1
     &              .or.l3.gt.n3) go to 202
!     if (l.lt.1.or.l.gt.nn) go to 202
               nnz=nnz+1
               irn(nnz)=k
               icn(nnz)=l
!     
               mat(nnz)=                                                          
!     
!     Mass term
!     
     &              d*omega2/mu(i1,i2,i3+1)                                           
!     
     &              +w1*h2*d3(i3)*d3b(i3)*r00p                                        
!     
!     R2, R3, R4
!     
     &              +w2u*0.25*h2*(                                                 
     &              -d2(i2)*(d2b(i2)*r0pp+d2b(i2-1)*r0mp)                             
     &              -d1(i1)*(d1b(i1)*rp0p+d1b(i1-1)*rm0p)                             
     &              +d3(i3)*d3b(i3)*(r0pp+r0mp+rp0p+rm0p))                            
     &              +w2u*h2*d3(i3)*d3b(i3)*r00p                                        
!     
!     B1, B2, B3, B4
!     
     &              +w3u*0.5*h2*d3(i3)*d3b(i3)*(rppp+rmmp+rmpp+rpmp)

!     
!     ---------------------------------------------------------
!     
!     Node -100
!     
 202           l=kkindex(i1-1,i2,i3,n1,n2) 
               l1=i1-1
               l2=i2
               l3=i3
               if (l1.lt.1.or.l1.gt.n1.or.l2.lt.1.or.l2.gt.n2.or.l3.lt.1
     &              .or.l3.gt.n3) go to 203
!   if (l.lt.1.or.l.gt.nn) go to 203
               nnz=nnz+1
               irn(nnz)=k
               icn(nnz)=l
!     
               mat(nnz)=                                                         
!     
!     Mass term
!     
     &              d*omega2/mu(i1-1,i2,i3)                                             
     &
!     
!     R1
!     
     &              +w1*h2*d1(i1)*d1b(i1-1)*rm00                                        
     &
!     
!     R2, R3, R4
!     
     &              +w2u*0.25*h2*(                                                    
     &              d1(i1)*d1b(i1-1)*(rm0m+rm0p+rmp0+rmm0)                           
     &              -d3(i3)*(d3b(i3)*rm0p+d3b(i3-1)*rm0m)                             
     &              -d2(i2)*(d2b(i2)*rmp0+d2b(i2-1)*rmm0))                              
     &              +w2u*h2*d1(i1)*d1b(i1-1)*rm00                                       
!     
!     B1, B2, B3, B4
!     
     &              +w3u*0.5*h2*d1(i1)*d1b(i1-1)*(rmmm+rmpp+rmmp+rmpm)
!     
!     ---------------------------------------------------------
!     
!     Node 0-10
!     
 203           l=kkindex(i1,i2-1,i3,n1,n2)
               l1=i1
               l2=i2-1
               l3=i3
               if (l1.lt.1.or.l1.gt.n1.or.l2.lt.1.or.l2.gt.n2.or.l3.lt.1
     &              .or.l3.gt.n3) go to 204
!     if (l.lt.1.or.l.gt.nn) go to 204
               nnz=nnz+1
               irn(nnz)=k
               icn(nnz)=l
!     
               mat(nnz)=                                                           
!     
!     Mass term
!     
     &              d*omega2/mu(i1,i2-1,i3)                                              
!     
!     R1
!     
     &              +w1*h2*d2(i2)*d2b(i2-1)*r0m0                                         
!     
!     R2, R3, R4
!     
     &              +w2u*0.25*h2*(                                                        
     &              d2(i2)*d2b(i2-1)*(r0mp+r0mm+rmm0+rpm0)                              
     &              -d3(i3)*(d3b(i3)*r0mp+d3b(i3-1)*r0mm)                             
     &              -d1(i1)*(d1b(i1)*rpm0+d1b(i1-1)*rmm0))                               
     &              +w2u*h2*d2(i2)*d2b(i2-1)*r0m0                                       
!     
!     B1, B2, B3, B4
!     
     &              +w3u*0.5*h2*d2(i2)*d2b(i2-1)*(rpmp+rmmm+rmmp+rpmm)
!     
!     ---------------------------------------------------------
!     
!     Node 00-1
!     
 204           l=kkindex(i1,i2,i3-1,n1,n2)
               l1=i1
               l2=i2
               l3=i3-1
               if (l1.lt.1.or.l1.gt.n1.or.l2.lt.1.or.l2.gt.n2.or.l3.lt.1
     &              .or.l3.gt.n3) go to 205
!     if (l.lt.1.or.l.gt.nn) go to 205
               nnz=nnz+1
               irn(nnz)=k
               icn(nnz)=l
!     
               mat(nnz)=                                                             
!     
!     Mass term
!     
     &              d*omega2/mu(i1,i2,i3-1)                                               
!     
!     R1
!     
     &              +w1*h2*d3(i3)*d3b(i3-1)*r00m                                        
!     
!     R2, R3, R4
!     
     &              +w2u*0.25*h2*(                                                         
     &              -d2(i2)*(d2b(i2)*r0pm+d2b(i2-1)*r0mm)                               
     &              -d1(i1)*(d1b(i1)*rp0m+d1b(i1-1)*rm0m)                               
     &              +d3(i3)*d3b(i3-1)*(r0pm+r0mm+rp0m+rm0m))                             
     &              +w2u*h2*d3(i3)*d3b(i3-1)*r00m                                       
!     
!     B1, B2, B3, B4
!     
     &              +w3u*0.5*h2*d3(i3)*d3b(i3-1)*(rppm+rmmm+rmpm+rpmm)                      

!     ---------------------------------------------------------
!     
!     12 nodes
!     
!     Node 110
!     
 205           l=kkindex(i1+1,i2+1,i3,n1,n2)
               l1=i1+1
               l2=i2+1
               l3=i3
               if (l1.lt.1.or.l1.gt.n1.or.l2.lt.1.or.l2.gt.n2.or.l3.lt.1
     &              .or.l3.gt.n3) go to 206
!     if (l.lt.1.or.l.gt.nn) go to 206
               nnz=nnz+1
               irn(nnz)=k
               icn(nnz)=l
!     
               mat(nnz)=                                                              
!     
!     Mass term
!     
     &              e*omega2/mu(i1+1,i2+1,i3)
!     
!     R2, R3 and R4
!     
     &              +w2u*0.25*h2*rpp0*(d1(i1)*d1b(i1)+d2(i2)*d2b(i2)) 
!     
!     B1,B2,B3 and B4
!     
     &              -w3u*0.5*h2*d3(i3)*(d3b(i3)*rppp+d3b(i3-1)*rppm)                       
!     
!     ---------------------------------------------------------
!     
!     Node 011
!     
 206           l=kkindex(i1,i2+1,i3+1,n1,n2)
               l1=i1
               l2=i2+1
               l3=i3+1
               if (l1.lt.1.or.l1.gt.n1.or.l2.lt.1.or.l2.gt.n2.or.l3.lt.1
     &              .or.l3.gt.n3) go to 207
!     if (l.lt.1.or.l.gt.nn) go to 207
               nnz=nnz+1
               irn(nnz)=k
               icn(nnz)=l
!     
               mat(nnz)=                                                               
!     
!     Mass term
!     
     &              e*omega2/mu(i1,i2+1,i3+1)                                             
!     
!     R2, R3 and R4
!     
     &              +w2u*0.25*h2*r0pp*(d2(i2)*d2b(i2)+d3(i3)*d3b(i3))                    
!     
!     B1,B2,B3 and B4
!     
     &              -w3u*0.5*h2*d1(i1)*(d1b(i1)*rppp+d1b(i1-1)*rmpp)                        
!     
!     ---------------------------------------------------------
!     
!     Node 101
!     
 207           l=kkindex(i1+1,i2,i3+1,n1,n2)
               l1=i1+1
               l2=i2
               l3=i3+1
               if (l1.lt.1.or.l1.gt.n1.or.l2.lt.1.or.l2.gt.n2.or.l3.lt.1
     &              .or.l3.gt.n3) go to 208
! if (l.lt.1.or.l.gt.nn) go to 208
               nnz=nnz+1
               irn(nnz)=k
               icn(nnz)=l
!     
               mat(nnz)=                                                             
!     
!     Mass term
!     
     &              e*omega2/mu(i1+1,i2,i3+1)                                          
!     
!     R2, R3 and R4
!     
     &              +w2u*0.25*h2*rp0p*(d1(i1)*d1b(i1)+d3(i3)*d3b(i3))                  
!     
!     B1,B2,B3 and B4
!     
     &              -w3u*0.5*h2*d2(i2)*(d2b(i2)*rppp+d2b(i2-1)*rpmp)        
!     
!     ---------------------------------------------------------
!     
!     Node -1-10
!     
 208           l=kkindex(i1-1,i2-1,i3,n1,n2)
               l1=i1-1
               l2=i2-1
               l3=i3
               if (l1.lt.1.or.l1.gt.n1.or.l2.lt.1.or.l2.gt.n2.or.l3.lt.1
     &              .or.l3.gt.n3) go to 209
!     if (l.lt.1.or.l.gt.nn) go to 209
               nnz=nnz+1
               irn(nnz)=k
               icn(nnz)=l
!     
               mat(nnz)=                                                               
!     
!     Mass term
!     
     &              e*omega2/mu(i1-1,i2-1,i3)                                             
!     
!     R2, R3 and R4
!     
     &             +w2u*0.25*h2*rmm0*(d1(i1)*d1b(i1-1)+d2(i2)*d2b(i2-1))                  
!     
!     B1,B2,B3 and B4
!     
     &              -w3u*0.5*h2*d3(i3)*(d3b(i3-1)*rmmm+d3b(i3)*rmmp)                        
!     
!     ---------------------------------------------------------
!     
!     Node 0-1-1
!     
 209           l=kkindex(i1,i2-1,i3-1,n1,n2)
               l1=i1
               l2=i2-1
               l3=i3-1
               if (l1.lt.1.or.l1.gt.n1.or.l2.lt.1.or.l2.gt.n2.or.l3.lt.1
     &              .or.l3.gt.n3) go to 210
!     if (l.lt.1.or.l.gt.nn) go to 210
               nnz=nnz+1
               irn(nnz)=k
               icn(nnz)=l
!     
               mat(nnz)=                                                              
!     
!     Mass term
!     
     &              e*omega2/mu(i1,i2-1,i3-1)                                            
!     
!     R2, R3 and R4
!     
     &             +w2u*0.25*h2*r0mm*(d2(i2)*d2b(i2-1)+d3(i3)*d3b(i3-1))                
!     
!     B1,B2,B3 and B4
!     
     &              -w3u*0.5*h2*d1(i1)*(d1b(i1-1)*rmmm+d1b(i1)*rpmm)                         
!     
!     ---------------------------------------------------------
!     
!     Node -10-1
!     
 210           l=kkindex(i1-1,i2,i3-1,n1,n2)
               l1=i1-1
               l2=i2
               l3=i3-1
               if (l1.lt.1.or.l1.gt.n1.or.l2.lt.1.or.l2.gt.n2.or.l3.lt.1
     &              .or.l3.gt.n3) go to 211
! if (l.lt.1.or.l.gt.nn) go to 211
               nnz=nnz+1
               irn(nnz)=k
               icn(nnz)=l
!     
               mat(nnz)=                                                               
!     
!     Mass term
!     
     &              e*omega2/mu(i1-1,i2,i3-1)                                             
!     
!     R2, R3 and R4
!     
     &             +w2u*0.25*h2*rm0m*(d1(i1)*d1b(i1-1)+d3(i3)*d3b(i3-1))                  
!     
!     B1,B2,B3 and B4
!     
     &              -w3u*0.5*h2*d2(i2)*(d2b(i2-1)*rmmm+d2b(i2)*rmpm)                        
!     
!     ---------------------------------------------------------
!     
!     Node -110
!     
 211           l=kkindex(i1-1,i2+1,i3,n1,n2)
               l1=i1-1
               l2=i2+1
               l3=i3
               if (l1.lt.1.or.l1.gt.n1.or.l2.lt.1.or.l2.gt.n2.or.l3.lt.1
     &              .or.l3.gt.n3) go to 212
!     if (l.lt.1.or.l.gt.nn) go to 212
               nnz=nnz+1
               irn(nnz)=k
               icn(nnz)=l
!     
               mat(nnz)=                                                             
!     
!     Mass term
!     
     &              e*omega2/mu(i1-1,i2+1,i3)                                              
!     
!     R2, R3 and R4
!     
     &              +w2u*0.25*h2*rmp0*(d1(i1)*d1b(i1-1)+d2(i2)*d2b(i2))                    
!     
!     B1,B2,B3 and B4
!     
     &              -w3u*0.5*h2*d3(i3)*(d3b(i3-1)*rmpm+d3b(i3)*rmpp)                        
!     
!     ---------------------------------------------------------
!     
!     Node 1-10
!     
 212           l=kkindex(i1+1,i2-1,i3,n1,n2)
               l1=i1+1
               l2=i2-1
               l3=i3
               if (l1.lt.1.or.l1.gt.n1.or.l2.lt.1.or.l2.gt.n2.or.l3.lt.1
     &              .or.l3.gt.n3) go to 213
!     if (l.lt.1.or.l.gt.nn) go to 213
               nnz=nnz+1
               irn(nnz)=k
               icn(nnz)=l
!     
               mat(nnz)=                                                              
!     
!     Mass term
!     
     &              e*omega2/mu(i1+1,i2-1,i3)                                           
!     
!     R2, R3 and R4
!     
     &              +w2u*0.25*h2*rpm0*(d1(i1)*d1b(i1)+d2(i2)*d2b(i2-1))                    
!     
!     B1,B2,B3 and B4
!     
     &              -w3u*0.5*h2*d3(i3)*(d3b(i3-1)*rpmm+d3b(i3)*rpmp)                      
!     
!     ---------------------------------------------------------
!     
!     Node 0-11
!     
 213           l=kkindex(i1,i2-1,i3+1,n1,n2)
               l1=i1
               l2=i2-1
               l3=i3+1
               if (l1.lt.1.or.l1.gt.n1.or.l2.lt.1.or.l2.gt.n2.or.l3.lt.1
     &              .or.l3.gt.n3) go to 214
!     if (l.lt.1.or.l.gt.nn) go to 214
               nnz=nnz+1
               irn(nnz)=k
               icn(nnz)=l
!     
               mat(nnz)=                                                             
!     
!     Mass term
!     
     &              e*omega2/mu(i1,i2-1,i3+1)                                           
!     
!     R2, R3 and R4
!     
     &              +w2u*0.25*h2*r0mp*(d2(i2)*d2b(i2-1)+d3(i3)*d3b(i3))                 
!     
!     B1,B2,B3 and B4
!     
     &              -w3u*0.5*h2*d1(i1)*(d1b(i1-1)*rmmp+d1b(i1)*rpmp)
!     
!     ---------------------------------------------------------
!     
!     Node 01-1
!     
 214           l=kkindex(i1,i2+1,i3-1,n1,n2)
               l1=i1
               l2=i2+1
               l3=i3-1
               if (l1.lt.1.or.l1.gt.n1.or.l2.lt.1.or.l2.gt.n2.or.l3.lt.1
     &              .or.l3.gt.n3) go to 215
!     if (l.lt.1.or.l.gt.nn) go to 215
               nnz=nnz+1
               irn(nnz)=k
               icn(nnz)=l
!     
               mat(nnz)=                                                             
!     
!     Mass term
!     
     &              e*omega2/mu(i1,i2+1,i3-1)                                              
!     
!     R2, R3 and R4
!     
     &              +w2u*0.25*h2*r0pm*(d2(i2)*d2b(i2)+d3(i3)*d3b(i3-1))                    
!     
!     B1,B2,B3 and B4
!     
     &              -w3u*0.5*h2*d1(i1)*(d1b(i1-1)*rmpm+d1b(i1)*rppm)
!     
!     ---------------------------------------------------------
!     
!     Node 10-1
!     
 215           l=kkindex(i1+1,i2,i3-1,n1,n2)
               l1=i1+1
               l2=i2
               l3=i3-1
               if (l1.lt.1.or.l1.gt.n1.or.l2.lt.1.or.l2.gt.n2.or.l3.lt.1
     &              .or.l3.gt.n3) go to 216
!     if (l.lt.1.or.l.gt.nn) go to 216
               nnz=nnz+1
               irn(nnz)=k
               icn(nnz)=l
!     
               mat(nnz)=                                                               
!     
!     Mass term
!     
     &              e*omega2/mu(i1+1,i2,i3-1)                                              
!     
!     R2, R3 and R4
!     
     &              +w2u*0.25*h2*rp0m*(d1(i1)*d1b(i1)+d3(i3)*d3b(i3-1))                     
!     
!     B1,B2,B3 and B4
!     
     &              -w3u*0.5*h2*d2(i2)*(d2b(i2-1)*rpmm+d2b(i2)*rppm)
!     
!     ---------------------------------------------------------
!     
!     Node -101
 216           l=kkindex(i1-1,i2,i3+1,n1,n2)
               l1=i1-1
               l2=i2
               l3=i3+1
               if (l1.lt.1.or.l1.gt.n1.or.l2.lt.1.or.l2.gt.n2.or.l3.lt.1
     &              .or.l3.gt.n3) go to 217
!     if (l.lt.1.or.l.gt.nn) go to 217
               nnz=nnz+1
               irn(nnz)=k
               icn(nnz)=l
!     
               mat(nnz)=                                                            
!     
!     Mass term
!     
     &          e*omega2/mu(i1-1,i2,i3+1)                                              
!     
!     R2, R3 and R4
!     
     &          +w2u*0.25*h2*rm0p*(d1(i1)*d1b(i1-1)+d3(i3)*d3b(i3))                  
!     
!     B1,B2,B3 and B4
!     
     &          -w3u*0.5*h2*d2(i2)*(d2b(i2-1)*rmmp+d2b(i2)*rmpp)

!     ---------------------------------------------------------


!     8 nodes
!     
!     Node 111
!     
 217           l=kkindex(i1+1,i2+1,i3+1,n1,n2)
               l1=i1+1
               l2=i2+1
               l3=i3+1
               if (l1.lt.1.or.l1.gt.n1.or.l2.lt.1.or.l2.gt.n2.or.l3.lt.1
     &              .or.l3.gt.n3) go to 218
!     if (l.lt.1.or.l.gt.nn) go to 218
               nnz=nnz+1
               irn(nnz)=k
               icn(nnz)=l
!     
               mat(nnz)=                                                              
!     
!     Mass term
!     
     &              f*omega2/mu(i1+1,i2+1,i3+1)                                               
!     
!     B1,B2,B3 and B4
!     
     &              +w3u*0.5*h2*rppp*(d2(i2)*d2b(i2)+d3(i3)*d3b(i3)
     &              +d1(i1)*d1b(i1))
!     ---------------------------------------------------------
!     
!     Node -1-1-1
!     
 218           l=kkindex(i1-1,i2-1,i3-1,n1,n2)
               l1=i1-1
               l2=i2-1
               l3=i3-1
               if (l1.lt.1.or.l1.gt.n1.or.l2.lt.1.or.l2.gt.n2.or.l3.lt.1
     &              .or.l3.gt.n3) go to 219
!     if (l.lt.1.or.l.gt.nn) go to 219
               nnz=nnz+1
               irn(nnz)=k
               icn(nnz)=l
!     
               mat(nnz)=                                                             
!     
!     Mass term
!     
     &              f*omega2/mu(i1-1,i2-1,i3-1)                                             
!     
!     B1,B2,B3 and B4
!     
     &              +w3u*0.5*h2*rmmm*(d2(i2)*d2b(i2-1)+d3(i3)*d3b(i3-1)+
     &              d1(i1)*d1b(i1-1))
!     
!     ---------------------------------------------------------
!     
!     Node -111
!     
 219           l=kkindex(i1-1,i2+1,i3+1,n1,n2)
               l1=i1-1
               l2=i2+1
               l3=i3+1
               if (l1.lt.1.or.l1.gt.n1.or.l2.lt.1.or.l2.gt.n2.or.l3.lt.1
     &              .or.l3.gt.n3) go to 220
!     if (l.lt.1.or.l.gt.nn) go to 220
               nnz=nnz+1
               irn(nnz)=k
               icn(nnz)=l
!     
               mat(nnz)=                                                              
!     
!     Mass term
!     
     &              f*omega2/mu(i1-1,i2+1,i3+1)                                             
!     
!     B1,B2,B3 and B4
!     
     &              +w3u*0.5*h2*rmpp*(d2(i2)*d2b(i2)+d3(i3)*d3b(i3)
     &              +d1(i1)*d1b(i1-1))
!     
!     ---------------------------------------------------------
!     
!     Node 1-1-1
!     
 220           l=kkindex(i1+1,i2-1,i3-1,n1,n2)
               l1=i1+1
               l2=i2-1
               l3=i3-1
               if (l1.lt.1.or.l1.gt.n1.or.l2.lt.1.or.l2.gt.n2.or.l3.lt.1
     &              .or.l3.gt.n3) go to 221
! if (l.lt.1.or.l.gt.nn) go to 221
               nnz=nnz+1
               irn(nnz)=k
               icn(nnz)=l
!     
               mat(nnz)=                                                              
!     
!     Mass term
!     
     &              f*omega2/mu(i1+1,i2-1,i3-1)                                         
!     
!     B1,B2,B3 and B4
!     
     &              +w3u*0.5*h2*rpmm*(d2(i2)*d2b(i2-1)+d3(i3)*d3b(i3-1)+
     &              d1(i1)*d1b(i1))
!     
!     ---------------------------------------------------------
!     
!     Node 1-11
!     
 221           l=kkindex(i1+1,i2-1,i3+1,n1,n2)
               l1=i1+1
               l2=i2-1
               l3=i3+1
               if (l1.lt.1.or.l1.gt.n1.or.l2.lt.1.or.l2.gt.n2.or.l3.lt.1
     &              .or.l3.gt.n3) go to 222
!   if (l.lt.1.or.l.gt.nn) go to 222
               nnz=nnz+1
               irn(nnz)=k
               icn(nnz)=l
!     
               mat(nnz)=                                                            
!     
!     Mass term
!     
     &          f*omega2/mu(i1+1,i2-1,i3+1)                                            
!     
!     B1,B2,B3 and B4
!     
     &          +w3u*0.5*h2*rpmp*(d2(i2)*d2b(i2-1)+d3(i3)*d3b(i3)
     &              +d1(i1)*d1b(i1))
!     
!     ---------------------------------------------------------
!     
!     Node -11-1
!     
 222           l=kkindex(i1-1,i2+1,i3-1,n1,n2)
               l1=i1-1
               l2=i2+1
               l3=i3-1
               if (l1.lt.1.or.l1.gt.n1.or.l2.lt.1.or.l2.gt.n2.or.l3.lt.1
     &              .or.l3.gt.n3) go to 223
!     if (l.lt.1.or.l.gt.nn) go to 223
               nnz=nnz+1
               irn(nnz)=k
               icn(nnz)=l
!     
               mat(nnz)=                                                             
!     
!     Mass term
!     
     &              f*omega2/mu(i1-1,i2+1,i3-1)                                            
!     
!     B1,B2,B3 and B4
!     
     &              +w3u*0.5*h2*rmpm*(d2(i2)*d2b(i2)+d3(i3)*d3b(i3-1)
     &              +d1(i1)*d1b(i1-1))
!     
!     ---------------------------------------------------------
!     
!     Node 11-1
!     
 223           l=kkindex(i1+1,i2+1,i3-1,n1,n2)
               l1=i1+1
               l2=i2+1
               l3=i3-1
               if (l1.lt.1.or.l1.gt.n1.or.l2.lt.1.or.l2.gt.n2.or.l3.lt.1
     &              .or.l3.gt.n3) go to 224
! if (l.lt.1.or.l.gt.nn) go to 224
               nnz=nnz+1
               irn(nnz)=k
               icn(nnz)=l
!     
               mat(nnz)=                                                               
!     
!     Mass term
!     
     &              f*omega2/mu(i1+1,i2+1,i3-1)                                            
!     
!     B1,B2,B3 and B4
!     
     &              +w3u*0.5*h2*rppm*(d2(i2)*d2b(i2)+d3(i3)*d3b(i3-1)
     &              +d1(i1)*d1b(i1))
!     
!     ---------------------------------------------------------
!     
!     Node -1-11
!     
 224           l=kkindex(i1-1,i2-1,i3+1,n1,n2)
               l1=i1-1
               l2=i2-1
               l3=i3+1
               if (l1.lt.1.or.l1.gt.n1.or.l2.lt.1.or.l2.gt.n2.or.l3.lt.1
     &              .or.l3.gt.n3) go to 225
!     if (l.lt.1.or.l.gt.nn) go to 225
               nnz=nnz+1
               irn(nnz)=k
               icn(nnz)=l
!     
               mat(nnz)=                                                               
!     
!     Mass term
!     
     &              f*omega2/mu(i1-1,i2-1,i3+1)                                             
!     
!     B1,B2,B3 and B4
!     
     &              +w3u*0.5*h2*rmmp*(d2(i2)*d2b(i2-1)+d3(i3)*d3b(i3)
     &              +d1(i1)*d1b(i1-1))

!     
 225           continue

!     ---------------------------------------------------------

            end do
         end do
      end do
      
      write(*,*) 'NNZ = ' ,nnz
      write(*,*) 'NE = ',ne


      return
      end subroutine subevalmatrix0


      SUBROUTINE subreadpar3D1_0(n1,n2,n3,npml,DATAFILE)
      IMPLICIT NONE
      INTEGER :: unitfwm,ivel,irho,iqf,n1,n2,n3,npml
      REAL :: dz
      CHARACTER(len=300)::DATAFILE

      unitfwm=42

      OPEN(unitfwm,file=trim(adjustl(DATAFILE)))

      READ(unitfwm,*) ivel,irho,iqf
      READ(unitfwm,*) n1,n2,n3,npml,dz

      CLOSE(unitfwm)
      
      RETURN
      END SUBROUTINE subreadpar3D1_0

      SUBROUTINE subreadpar3D1_1(ivel,irho,iqf,n1,n2,n3,npml,dz,freq
     &     ,vel0,rho0,qf0,namevel,namerho,nameqf,DATAFILE)
      IMPLICIT NONE
      INTEGER :: unitfwm,ivel,irho,iqf,n1,n2,n3,npml
      REAL :: dz,freq,vel0,rho0,qf0
      CHARACTER(LEN=160) :: namevel,namerho,nameqf
      CHARACTER(len=300)::DATAFILE

      unitfwm=42

      OPEN(unitfwm,file=trim(adjustl(DATAFILE)))

      READ(unitfwm,*) ivel,irho,iqf
      READ(unitfwm,*) n1,n2,n3,npml,dz
      READ(unitfwm,*) freq
      READ(unitfwm,*) vel0,rho0,qf0
      READ(unitfwm,*) namevel,namerho,nameqf

      CLOSE(unitfwm)

      RETURN
      END SUBROUTINE subreadpar3D1_1

!     --------------------------------------------------------------------------------------
!     SUBROUTINE SUBRHOMU: BUILD RHO AND MU VECTOR TO BUILD IMPEDANCE MATRIX
!     INPUT:
!     ivel(0/1),irho(0/1),iqf(0/1): homogeneous media (0) or heterogeneous media read from disk (1)
!     vel0,rho0,qf0: value of the velocity, density and quality factor if ivel=0, irho=0, iqf=0, respectively
!     namec,namerho,nameq: input file name for velocity, density and quality factor if ivel=1, irho=1, iqf=1, respectively
!     nn: number of unknowns in the input grid
!     nne2: dimension of the buffer vectors used to build the impedance matrix
!     OUTPUT:
!     rho(nne2): density vector used to build the impedance matrix
!     mu(nne2): bulk modulus vector used to build the impedance matrix
!     
!
      SUBROUTINE subrhomu(ivel,irho,iqf,vel0,rho0
     &     ,qf0,namevel,namerho,nameqf,n1,n2,n3,npml,nn,nne2,rho,mu)

      IMPLICIT NONE

      CHARACTER(LEN=160) :: namevel,namerho,nameqf
      
      INTEGER :: ivel,irho,iqf,nn,nne2,n1,n2,n3,npml
      INTEGER :: unitvel,unitrho,unitqf
      
      REAL :: vel0,rho0,qf0

      COMPLEX :: velc,ci

      REAL,ALLOCATABLE :: veltemp(:),vel(:),rhotemp(:)
      REAL,ALLOCATABLE :: qf(:),qftemp(:)

      REAL,DIMENSION(nne2) :: rho
      COMPLEX,DIMENSION(nne2) :: mu

      ci=cmplx(0.,1.)

      unitvel=43
      unitrho=44
      unitqf=45
      
      IF (ivel.eq.1) THEN
         ALLOCATE (veltemp(nn))
         ALLOCATE (vel(nne2))
         OPEN(unitvel,file=namevel,access='direct',recl=nn*4)
         READ(unitvel,rec=1) veltemp
         CLOSE(unitvel)
      END IF
      
      IF (irho.eq.1) THEN
         ALLOCATE (rhotemp(nn))
         OPEN(unitrho,file=namerho,access='direct',recl=nn*4)
         READ(unitrho,rec=1) rhotemp
         CLOSE(unitrho)
      END IF
      
      IF (iqf.eq.1) THEN
         ALLOCATE (qftemp(nn))
         ALLOCATE (qf(nne2))
         OPEN(unitqf,file=nameqf,access='direct',recl=nn*4)
         READ(unitqf,rec=1) qf
         CLOSE(unitqf)
      END IF

      IF ( ivel.eq.0 .and. irho.eq.0 .and. iqf.eq.0) THEN !IVEL=0/IRHO=0/IQF=0
         velc=vel0*(1.-0.5*ci/qf0)
         rho(:)=rho0
         mu(:)=rho0*velc*velc
      ELSE IF ( ivel.eq.1 .and. irho.eq.0 .and. iqf.eq.0) THEN !IVEL=1/IRHO=0/IQF=0
         rho(:)=rho0
         CALL subaugment(veltemp,n1,n2,n3,npml+1,npml+1,npml+1,vel)
         CALL submu100(vel,rho0,qf0,nne2,mu)
      ELSE IF ( ivel.eq.0 .and. irho.eq.1 .and. iqf.eq.0) THEN !IVEL=0/IRHO=1/IQF=0
         CALL subaugment(rhotemp,n1,n2,n3,npml+1,npml+1,npml+1,rho)
         CALL submu010(vel0,rho,qf0,nne2,mu)
      ELSE IF ( ivel.eq.0 .and. irho.eq.0 .and. iqf.eq.1) THEN !IVEL=0/IRHO=0/IQF=1
         rho(:)=rho0
         CALL subaugment(qftemp,n1,n2,n3,npml+1,npml+1,npml+1,qf)
         CALL submu001(vel0,rho0,qf,nne2,mu)
      ELSE IF ( ivel.eq.1 .and. irho.eq.1 .and. iqf.eq.0) THEN !IVEL=1/IRHO=1/IQF=0
         CALL subaugment(veltemp,n1,n2,n3,npml+1,npml+1,npml+1,vel)
         CALL subaugment(rhotemp,n1,n2,n3,npml+1,npml+1,npml+1,rho)
         CALL submu110(vel,rho,qf0,nne2,mu)
      ELSE IF ( ivel.eq.1 .and. irho.eq.0 .and. iqf.eq.1) THEN !IVEL=1/IRHO=0/IQF=1
         rho(:)=rho0
         CALL subaugment(veltemp,n1,n2,n3,npml+1,npml+1,npml+1,vel)
         CALL subaugment(qftemp,n1,n2,n3,npml+1,npml+1,npml+1,qf)
         CALL submu101(vel,rho0,qf,nne2,mu)
      ELSE IF ( ivel.eq.0 .and. irho.eq.1 .and. iqf.eq.1) THEN !IC=0/IRHO=1/IQ=1
         CALL subaugment(rhotemp,n1,n2,n3,npml+1,npml+1,npml+1,rho)
         CALL subaugment(qftemp,n1,n2,n3,npml+1,npml+1,npml+1,qf)
         CALL submu011(vel0,rho,qf,nne2,mu)
      ELSE IF ( ivel.eq.1 .and. irho.eq.1 .and. iqf.eq.1) THEN !IC=1/IRHO=1/IQ=1
         CALL subaugment(veltemp,n1,n2,n3,npml+1,npml+1,npml+1,vel)
         CALL subaugment(rhotemp,n1,n2,n3,npml+1,npml+1,npml+1,rho)
         CALL subaugment(qftemp,n1,n2,n3,npml+1,npml+1,npml+1,qf)
         CALL submu111(vel,rho,qf,nne2,mu)
      ELSE
         WRITE(*,*) 'BUG: NO OPTION FOUND TO BUILD '
     &        //'BULK MODULUS MODEL!!!'
      END IF

      IF (ivel.eq.1) THEN
         DEALLOCATE(veltemp)
         DEALLOCATE(vel)
      END IF

      IF (irho.eq.1) THEN
         DEALLOCATE(rhotemp)
      END IF

      IF (iqf.eq.1) THEN
         DEALLOCATE(qf)
         DEALLOCATE(qftemp)
      END IF
      
      RETURN
      END SUBROUTINE subrhomu
!     
! ---------------------------------------------------------------------------------------------
!     
!     SUBROUTINES TO COMPUTE BULK MODULUS FROM VELOCITY, DENSITY AND ATTENUATION MODELS
!     
!     MU = VEL * ( 1 - 0.5 * CMPLX(0.,1.) / QF )
!     
!      
! ---------------------------------------------------------------------------------------------

      SUBROUTINE submu100(vel,rho0,qf0,nn,mu)
      IMPLICIT NONE
      INTEGER :: nn
      REAL,DIMENSION(nn) :: vel
      REAL :: rho0,qf0
      COMPLEX,DIMENSION(nn) :: mu
      COMPLEX :: ci
      
      ci = cmplx(0.,1.)

      mu(:) = rho0 * vel(:) * (1. - 0.5 * ci / qf0 ) * vel(:) * 
     &     (1. - 0.5 * ci / qf0 )

      RETURN
      END SUBROUTINE submu100

! ---------------------------------------------------------------------------------------------

      SUBROUTINE submu010(vel0,rho,qf0,nn,mu)
      IMPLICIT NONE
      INTEGER :: nn
      REAL,DIMENSION(nn) :: rho
      REAL :: vel0,qf0
      COMPLEX,DIMENSION(nn) :: mu
      COMPLEX :: ci

      ci = cmplx(0.,1.)

      mu(:) = rho(:) * vel0 * (1. - 0.5 * ci / qf0 ) * vel0 * 
     &     (1. - 0.5 * ci / qf0 )

      RETURN
      END SUBROUTINE submu010
      
! ---------------------------------------------------------------------------------------------

      SUBROUTINE submu001(vel0,rho0,qf,nn,mu)
      IMPLICIT NONE
      INTEGER :: nn
      REAL,DIMENSION(nn) :: qf
      REAL :: vel0,rho0
      COMPLEX,DIMENSION(nn) :: mu
      COMPLEX :: ci

      ci = cmplx(0.,1.)

      mu(:) = rho0 * vel0 * (1. - 0.5 * ci / qf(:) ) * vel0 * 
     &     (1. - 0.5 * ci / qf(:) )
      
      RETURN
      END SUBROUTINE submu001

! ---------------------------------------------------------------------------------------------

      SUBROUTINE submu110(vel,rho,qf0,nn,mu)
      IMPLICIT NONE
      INTEGER :: nn
      REAL,DIMENSION(nn) :: vel,rho
      REAL :: qf0
      COMPLEX,DIMENSION(nn) :: mu
      COMPLEX :: ci

      ci = cmplx(0.,1.)

      mu(:) = rho(:) * vel(:) * (1. - 0.5 * ci / qf0 ) * vel(:) * 
     &     (1. - 0.5 * ci / qf0 )

      RETURN
      END SUBROUTINE submu110

! ---------------------------------------------------------------------------------------------

      SUBROUTINE submu101(vel,rho0,qf,nn,mu)
      IMPLICIT NONE
      INTEGER :: nn
      REAL,DIMENSION(nn) :: vel,qf
      REAL :: rho0
      COMPLEX,DIMENSION(nn) :: mu
      COMPLEX :: ci

      ci = cmplx(0.,1.)

      mu(:) = rho0 * vel(:) * (1. - 0.5 * ci / qf(:) ) * vel(:) * 
     &     (1. - 0.5 * ci / qf(:) )

      RETURN
      END SUBROUTINE submu101

! ---------------------------------------------------------------------------------------------
      
      SUBROUTINE submu011(vel0,rho,qf,nn,mu)
      IMPLICIT NONE
      INTEGER :: nn
      REAL,DIMENSION(nn) :: rho,qf
      REAL :: vel0
      COMPLEX,DIMENSION(nn) :: mu
      COMPLEX :: ci

      ci = cmplx(0.,1.)

      mu(:) = rho(:) * vel0 * (1. - 0.5 * ci / qf(:) ) * vel0 * 
     &     (1. - 0.5 * ci / qf(:) )
      
      RETURN
      END SUBROUTINE submu011

!     ---------------------------------------------------------------------------------------------

      SUBROUTINE submu111(vel,rho,qf,nn,mu)
      IMPLICIT NONE
      INTEGER :: nn
      REAL,DIMENSION(nn) :: vel,rho,qf
      COMPLEX,DIMENSION(nn) :: mu
      COMPLEX :: ci

      ci = cmplx(0.,1.)

      mu(:) = rho(:) * vel(:) * (1. - 0.5 * ci / qf(:) ) * vel(:) * 
     &     (1. - 0.5 * ci / qf(:) )
      
      RETURN
      END SUBROUTINE submu111

! -------------------------------------------------------------------------------------------------
!     SUBROUTINE SUBAUGMENT: AUGMENT A 3D FD GRID WITH PML LAYERS ALONG ALL THE SIDES OF THE CUBE
!     Inputs:
!     x(n1,n2,n3): input model
!     n1, n2, n2: size of the original model
!     npt1 npt2 npt3: number of PML points in the 3 directions.
!     Output:
!     x1(n1+2*npt1,n2+2*npt2,n3+2*npt3)
!     ------------------------------------------------------------------------------------------------
      subroutine subaugment(x,n1,n2,n3,npt1,npt2,npt3,x1)

      integer :: n1,n2,n3,npt1,npt2,npt3,i1,i2,i3
      real :: x(n1,n2,n3)
      real :: x1(-npt1+1:n1+npt1,-npt2+1:n2+npt2
     &     ,-npt3+1:n3+npt3)

      do i3=1,n3
         do i2=1,n2
            do i1=1,n1
               x1(i1,i2,i3)=x(i1,i2,i3)
            end do
         end do
      end do

!     Front and back faces
      
      do i2=1,n2
         do i1=1,n1
            do i3=-npt3+1,0
               x1(i1,i2,i3)=x(i1,i2,1)
            end do
            do i3=n3+1,n3+npt3
               x1(i1,i2,i3)=x(i1,i2,n3)
            end do
         end do
      end do
      
!     Top and bottom faces
      
      do i3=1,n3
         do i2=1,n2
            do i1=-npt1+1,0
               x1(i1,i2,i3)=x(1,i2,i3)
            end do
            do i1=n1+1,n1+npt1
               x1(i1,i2,i3)=x(n1,i2,i3)
            end do
         end do
      end do

!     Right and left faces

      do i3=1,n3
         do i1=1,n1
            do i2=-npt2+1,0
               x1(i1,i2,i3)=x(i1,1,i3)
            end do
            do i2=n2+1,n2+npt2
               x1(i1,i2,i3)=x(i1,n2,i3)
            end do
         end do
      end do
 
!     

      do i1=1,n1

         do i2=-npt2+1,0
            do i3=-npt3+1,0
               x1(i1,i2,i3)=x(i1,1,1)
            end do
         end do

         do i2=n2+1,n2+npt2
            do i3=n3+1,n3+npt3
               x1(i1,i2,i3)=x(i1,n2,n3)
            end do
         end do

         do i2=-npt2+1,0
            do i3=n3+1,n3+npt3
               x1(i1,i2,i3)=x(i1,1,n3)
            end do
         end do

         do i2=n2+1,n2+npt2
            do i3=-npt3+1,0
               x1(i1,i2,i3)=x(i1,n2,1)
            end do
         end do
         
      end do

!     
      do i2=1,n2

         do i1=-npt1+1,0
            do i3=-npt3+1,0
               x1(i1,i2,i3)=x(1,i2,1)
            end do
         end do

         do i1=n1+1,n1+npt1
            do i3=n3+1,n3+npt3
               x1(i1,i2,i3)=x(n1,i2,n3)
            end do
         end do

         do i1=-npt1+1,0
            do i3=n3+1,n3+npt3
               x1(i1,i2,i3)=x(1,i2,n3)
            end do
         end do

         do i1=n1+1,n1+npt1
            do i3=-npt3+1,0
               x1(i1,i2,i3)=x(n1,i2,1)
            end do
         end do

      end do

!     

      do i3=1,n3
         
         do i1=-npt1+1,0
            do i2=-npt2+1,0
               x1(i1,i2,i3)=x(1,1,i3)
            end do
         end do

         do i1=n1+1,n1+npt1
            do i2=n2+1,n2+npt2
               x1(i1,i2,i3)=x(n1,n2,i3)
            end do
         end do

         do i1=-npt1+1,0
            do i2=n2+1,n2+npt2
               x1(i1,i2,i3)=x(1,n2,i3)
            end do
         end do

         do i1=n1+1,n1+npt1
            do i2=-npt2+1,0
               x1(i1,i2,i3)=x(n1,1,i3)
            end do
         end do

      end do

!     eight corners

      do i3=-npt3+1,0
         do i2=-npt2+1,0
            do i1=-npt1+1,0
               x1(i1,i2,i3)=x(1,1,1)
            end do
         end do
      end do

      do i3=-npt3+1,0
         do i2=-npt2+1,0
            do i1=n1+1,n1+npt1
               x1(i1,i2,i3)=x(n1,1,1)
            end do
         end do
      end do

      do i3=-npt3+1,0
         do i2=n2+1,n2+npt2
            do i1=-npt1+1,0
               x1(i1,i2,i3)=x(1,n2,1)
            end do
         end do
      end do

      do i3=n3+1,n3+npt3
         do i2=-npt2+1,0
            do i1=-npt1+1,0
               x1(i1,i2,i3)=x(1,1,n3)
            end do
         end do
      end do
 
      do i3=-npt3+1,0
         do i2=n2+1,n2+npt2
            do i1=n1+1,n1+npt1
               x1(i1,i2,i3)=x(n1,n2,1)
            end do
         end do
      end do

      do i3=n3+1,n3+npt3
         do i2=-npt2+1,0
            do i1=n1+1,n1+npt1
               x1(i1,i2,i3)=x(n1,1,n3)
            end do
         end do
      end do

      do i3=n3+1,n3+npt3
         do i2=n2+1,n2+npt2
            do i1=-npt1+1,0
               x1(i1,i2,i3)=x(1,n2,n3)
            end do
         end do
      end do

      do i3=n3+1,n3+npt3
         do i2=n2+1,n2+npt2
            do i1=n1+1,n1+npt1
               x1(i1,i2,i3)=x(n1,n2,n3)
            end do
         end do
      end do
 
      return
      end subroutine subaugment

