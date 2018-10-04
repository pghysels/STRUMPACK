# Manual makefile for STRUMPACK. If the CMake installation procedure
# does not work for you, you can simply run make in this folder.  You
# will have to create/modify the make.inc file in the STRUMPACK root
# folder.

include ../make.inc

VPATH = dense/:dense/lapack/:misc/:sparse/:sparse/rcm/:
SPSRC = \
	StrumpackSparseSolver.o \
	StrumpackParameters.o strumpack_mc64ad.o \
	rcm.o degree.o fnroot.o genrcm.o rootls.o \
	TaskTimer.o BLASLAPACKOpenMPTask.o \
	sgeqp3tol.o dgeqp3tol.o cgeqp3tol.o zgeqp3tol.o \
	sgeqrfmod.o dgeqrfmod.o cgeqrfmod.o zgeqrfmod.o \
	sgeqr2mod.o dgeqr2mod.o cgeqr2mod.o zgeqr2mod.o \
	slaqpsmod.o dlaqpsmod.o claqpsmod.o zlaqpsmod.o \
	slaqp2mod.o dlaqp2mod.o claqp2mod.o zlaqp2mod.o \
	sgetrfmod.o dgetrfmod.o cgetrfmod.o zgetrfmod.o \
	sgetf2mod.o dgetf2mod.o cgetf2mod.o zgetf2mod.o \
	sorglqmod.o dorglqmod.o cunglqmod.o zunglqmod.o \
	sorgl2mod.o dorgl2mod.o cungl2mod.o zungl2mod.o \
	slarfbmod.o dlarfbmod.o clarfbmod.o zlarfbmod.o \
	slarftmod.o dlarftmod.o clarftmod.o zlarftmod.o \
	slarfmod.o  dlarfmod.o  clarfmod.o  zlarfmod.o \
	sgelqfmod.o dgelqfmod.o cgelqfmod.o zgelqfmod.o \
	sgelq2mod.o dgelq2mod.o cgelq2mod.o zgelq2mod.o \
	cgeqp3mod.o dgeqp3mod.o sgeqp3mod.o zgeqp3mod.o

.c.o:
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< $(VERBOSE)

.f.o:
	$(FORTRAN) $(FFLAGS) $(CPPFLAGS) -c $< $(VERBOSE)

.cpp.o:
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< $(VERBOSE)

all: $(SPSRC)
	$(ARCH) $(ARCHFLAGS) $(STRUMPACKLIB) $(SPSRC)
	$(RANLIB) $(STRUMPACKLIB)

clean:
	rm -f $(SPSRC) $(STRUMPACKLIB) *.o
