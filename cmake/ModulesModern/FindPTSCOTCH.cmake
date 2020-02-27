#   FindPTSCOTCH.cmake
#
#   Finds the PTSCOTCH library.
#
#   This module will define the following variables:
#   
#     PTSCOTCH_FOUND         - System has found PTSCOTCH installation
#     PTSCOTCH_INCLUDE_DIR   - Location of PTSCOTCH headers
#     PTSCOTCH_LIBRARIES     - PTSCOTCH libraries
#     PTSCOTCH_USES_ILP64    - Whether PTSCOTCH was configured with ILP64
#     PTSCOTCH_USES_PTHREADS - Whether PTSCOTCH was configured with PThreads
#
#   This module can handle the following COMPONENTS
#
#     ilp64    - 64-bit index integers
#     pthreads - SMP parallelism via PThreads
#     parmetis - Has ParMETIS compatibility layer
#
#   This module will export the following targets if PTSCOTCH_FOUND
#
#     PTSCOTCH::ptscotch
#
#
#
#
#   Proper usage:
#
#     project( TEST_FIND_PTSCOTCH C )
#     find_package( PTSCOTCH )
#
#     if( PTSCOTCH_FOUND )
#       add_executable( test test.cxx )
#       target_link_libraries( test PTSCOTCH::ptscotch )
#     endif()
#
#
#
#
#   This module will use the following variables to change
#   default behaviour if set
#
#     ptscotch_PREFIX
#     ptscotch_INCLUDE_DIR
#     ptscotch_LIBRARY_DIR
#     ptscotch_LIBRARIES

#==================================================================
#   Copyright (c) 2018 The Regents of the University of California,
#   through Lawrence Berkeley National Laboratory.  
#
#   Author: David Williams-Young
#   
#   This file is part of cmake-modules. All rights reserved.
#   
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#   
#   (1) Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#   (2) Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#   (3) Neither the name of the University of California, Lawrence Berkeley
#   National Laboratory, U.S. Dept. of Energy nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
#   
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#   ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
#   ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#   LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
#   ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#   
#   You are under no obligation whatsoever to provide any bug fixes, patches, or
#   upgrades to the features, functionality or performance of the source code
#   ("Enhancements") to anyone; however, if you choose to make your Enhancements
#   available either publicly, or directly to Lawrence Berkeley National
#   Laboratory, without imposing a separate written license agreement for such
#   Enhancements, then you hereby grant the following license: a non-exclusive,
#   royalty-free perpetual license to install, use, modify, prepare derivative
#   works, incorporate into other computer software, distribute, and sublicense
#   such enhancements or derivative works thereof, in binary and source code form.
#
#==================================================================

cmake_minimum_required( VERSION 3.11 ) # Require CMake 3.11+
# Set up some auxillary vars if hints have been set

if( ptscotch_PREFIX AND NOT ptscotch_INCLUDE_DIR )
  set( ptscotch_INCLUDE_DIR ${ptscotch_PREFIX}/include )
endif()


if( ptscotch_PREFIX AND NOT ptscotch_LIBRARY_DIR )
  set( ptscotch_LIBRARY_DIR 
    ${ptscotch_PREFIX}/lib 
    ${ptscotch_PREFIX}/lib32 
    ${ptscotch_PREFIX}/lib64 
  )
endif()

# Pass ptscotch vars as scotch vars if they exist
if( ptscotch_PREFIX AND NOT scotch_PREFIX )
  set( scotch_PREFIX ${ptscotch_PREFIX} )
endif()

if( ptscotch_INCLUDE_DIR AND NOT scotch_INCLUDE_DIR )
  set( scotch_INCLUDE_DIR ${ptscotch_INCLUDE_DIR} )
endif()

if( ptscotch_LIBRARY_DIR AND NOT scotch_LIBRARY_DIR )
  set( scotch_LIBRARY_DIR ${ptscotch_LIBRARY_DIR} )
endif()


# DEPENDENCIES
# Make sure C is enabled
get_property( PTSCOTCH_languages GLOBAL PROPERTY ENABLED_LANGUAGES )
if( NOT "C" IN_LIST PTSCOTCH_languages )
  message( FATAL_ERROR "C Language Must Be Enabled for PTSCOTCH Linkage" )
endif()


# SCOTCH
if( NOT TARGET SCOTCH::scotch )
  find_dependency( SCOTCH REQUIRED )
endif()


# MPI
if( NOT TARGET MPI::MPI_C )
  find_dependency( MPI REQUIRED )
endif()


# Try to find the header
find_path( PTSCOTCH_INCLUDE_DIR 
  NAMES ptscotch.h
  HINTS ${ptscotch_PREFIX}
  PATHS ${ptscotch_INCLUDE_DIR}
  PATH_SUFFIXES include
  DOC "Location of PTSCOTCH header"
)

# Try to find libraries if not already set
if( NOT ptscotch_LIBRARIES )

  find_library( PTSCOTCH_LIBRARY
    NAMES ptscotch 
    HINTS ${ptscotch_PREFIX}
    PATHS ${ptscotch_LIBRARY_DIR}
    PATH_SUFFIXES lib lib64 lib32
    DOC "PTSCOTCH Library"
  )

  find_library( PTSCOTCH_ERR_LIBRARY
    NAMES ptscotcherr
    HINTS ${ptscotch_PREFIX}
    PATHS ${ptscotch_LIBRARY_DIR}
    PATH_SUFFIXES lib lib64 lib32
    DOC "PTSCOTCH Error Libraries"
  )

  find_library( PTSCOTCH_ERREXIT_LIBRARY
    NAMES ptscotcherrexit
    HINTS ${ptscotch_PREFIX}
    PATHS ${ptscotch_LIBRARY_DIR}
    PATH_SUFFIXES lib lib64 lib32
    DOC "PTSCOTCH Error-Exit Libraries"
  )

  set( PTSCOTCH_LIBRARIES 
       ${PTSCOTCH_LIBRARY} 
       ${PTSCOTCH_ERR_LIBRARY}
       ${PTSCOTCH_ERREXIT_LIBRARY} )


  if( "parmetis" IN_LIST PTSCOTCH_FIND_COMPONENTS )

    find_library( PTSCOTCH_PARMETIS_LIBRARY
      NAMES ptscotchparmetis
      HINTS ${ptscotch_PREFIX}
      PATHS ${ptscotch_LIBRARY_DIR}
      PATH_SUFFIXES lib lib64 lib32
      DOC "PTSCOTCH-ParMETIS compatibility Libraries"
    )

    if( PTSCOTCH_PARMETIS_LIBRARY )
      list( APPEND PTSCOTCH_LIBRARIES ${PTSCOTCH_PARMETIS_LIBRARY} )
      set( PTSCOTCH_parmetis_FOUND TRUE )
    endif()

  endif()

else()

  # FIXME: Check if files exists at least?
  set( PTSCOTCH_LIBRARIES ${ptscotch_LIBRARIES} )

endif()

# Check version
if( SCOTCH_FOUND )

  set( PTSCOTCH_VERSION_STRING ${SCOTCH_VERSION_STRING} )
  set( PTSCOTCH_USES_ILP64     ${SCOTCH_USES_ILP64} )
  set( PTSCOTCH_USES_PTHREADS  ${SCOTCH_USES_PTHREADS} )
  
endif()



# Handle components
if( PTSCOTCH_USES_ILP64 )
  set( PTSCOTCH_ilp64_FOUND TRUE )
endif()

if( PTSCOTCH_USES_PTHREADS )
  set( PTSCOTCH_pthreads_FOUND TRUE )
endif()

# Determine if we've found PTSCOTCH
mark_as_advanced( PTSCOTCH_FOUND PTSCOTCH_INCLUDE_DIR PTSCOTCH_LIBRARIES )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args( PTSCOTCH
  REQUIRED_VARS PTSCOTCH_LIBRARIES PTSCOTCH_INCLUDE_DIR
  VERSION_VAR PTSCOTCH_VERSION_STRING
  HANDLE_COMPONENTS
)

# Export target
if( PTSCOTCH_FOUND AND NOT TARGET PTSCOTCH::ptscotch )

  add_library( PTSCOTCH::ptscotch INTERFACE IMPORTED )
  set_target_properties( PTSCOTCH::ptscotch PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${PTSCOTCH_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES      "${PTSCOTCH_LIBRARIES};SCOTCH::scotch;MPI::MPI_C" 
  )

endif()
