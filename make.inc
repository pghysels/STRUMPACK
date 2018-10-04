# STRUMPACKroot = ${CMAKE_INSTALL_PREFIX}

STRUMPACKROOT = $(shell pwd)/../
STRUMPACKLIB  = $(STRUMPACKROOT)/src/libstrumpack.a

METIS_DIR     = $(HOME)/local/parmetis-4.0.3/metis

## optional
PARMETIS_DIR  = $(HOME)/local/parmetis-4.0.3
SCOTCH_DIR    = $(HOME)/local/scotch_6.0.4

ARCH          = ar
ARCHFLAGS     = cr
RANLIB        = ranlib

CC            = mpicc
CXX           = mpic++
FORTRAN       = mpif90

OPTFLAGS      = -O3 -march=native -fopenmp
CPPFLAGS      = -I./ -I$(METIS_DIR)/include \
                -I$(PARMETIS_DIR)/include \
                -I$(SCOTCH_DIR)/include

CFLAGS        = $(OPTFLAGS)
CXXFLAGS      = $(OPTFLAGS)
FFLAGS        = $(OPTFLAGS)

