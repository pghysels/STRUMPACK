if (NOT (TPL_SCALAPACK_LIBRARIES))
  find_library(SCALAPACK_LIBRARY
    NAMES scalapack scalapack-pvm scalapack-mpi scalapack-mpich scalapack-mpich2 scalapack-openmpi scalapack-lam
    PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib /sw/lib /usr/lib64/openmpi/lib /usr/lib64/mpich/lib
    ENV LD_LIBRARY_PATH ENV SCALAPACKDIR ENV BLACSDIR)
  find_library(BLACS_LIBRARY
    NAMES blacs blacs-pvm blacs-mpi blacs-mpich blacs-mpich2 blacs-openmpi blacs-lam mpiblacs scalapack ${SCALAPACK_LIBRARY}
    PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib /sw/lib /usr/lib64/openmpi/lib /usr/lib64/mpich/lib
    ENV LD_LIBRARY_PATH ENV SCALAPACKDIR ENV BLACSDIR)
  set(SCALAPACK_LIBRARIES ${SCALAPACK_LIBRARY} ${BLACS_LIBRARY})
else()
  set(SCALAPACK_LIBRARIES ${TPL_SCALAPACK_LIBRARIES})
endif()

# Report the found libraries, quit with fatal error if any required library has not been found.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SCALAPACK DEFAULT_MSG SCALAPACK_LIBRARIES)
