#   FindSCOTCH.cmake
#
#   Finds the SCOTCH library.
#
#   This module will define the following variables:
#   
#     SCOTCH_FOUND         - System has found SCOTCH installation
#     SCOTCH_INCLUDE_DIR   - Location of SCOTCH headers
#     SCOTCH_LIBRARIES     - SCOTCH libraries
#     SCOTCH_USES_ILP64    - Whether SCOTCH was configured with ILP64
#     SCOTCH_USES_PTHREADS - Whether SCOTCH was configured with PThreads
#
#   This module can handle the following COMPONENTS
#
#     ilp64    - 64-bit index integers
#     pthreads - SMP parallelism via PThreads
#     metis    - Has METIS compatibility layer
#
#   This module will export the following targets if SCOTCH_FOUND
#
#     SCOTCH::scotch
#
#
#
#
#   Proper usage:
#
#     project( TEST_FIND_SCOTCH C )
#     find_package( SCOTCH )
#
#     if( SCOTCH_FOUND )
#       add_executable( test test.cxx )
#       target_link_libraries( test SCOTCH::scotch )
#     endif()
#
#
#
#
#   This module will use the following variables to change
#   default behaviour if set
#
#     scotch_PREFIX
#     scotch_INCLUDE_DIR
#     scotch_LIBRARY_DIR
#     scotch_LIBRARIES

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

if( scotch_PREFIX AND NOT scotch_INCLUDE_DIR )
  set( scotch_INCLUDE_DIR ${scotch_PREFIX}/include )
endif()


if( scotch_PREFIX AND NOT scotch_LIBRARY_DIR )
  set( scotch_LIBRARY_DIR 
    ${scotch_PREFIX}/lib 
    ${scotch_PREFIX}/lib32 
    ${scotch_PREFIX}/lib64 
  )
endif()


# Try to find the header
find_path( SCOTCH_INCLUDE_DIR 
  NAMES scotch.h
  HINTS ${scotch_PREFIX}
  PATHS ${scotch_INCLUDE_DIR}
  PATH_SUFFIXES include
  DOC "Location of SCOTCH header"
)

# Try to find libraries if not already set
if( NOT scotch_LIBRARIES )

  find_library( SCOTCH_LIBRARY
    NAMES scotch 
    HINTS ${scotch_PREFIX}
    PATHS ${scotch_LIBRARY_DIR}
    PATH_SUFFIXES lib lib64 lib32
    DOC "SCOTCH Library"
  )

  find_library( SCOTCH_ERR_LIBRARY
    NAMES scotcherr
    HINTS ${scotch_PREFIX}
    PATHS ${scotch_LIBRARY_DIR}
    PATH_SUFFIXES lib lib64 lib32
    DOC "SCOTCH Error Libraries"
  )

  find_library( SCOTCH_ERREXIT_LIBRARY
    NAMES scotcherrexit
    HINTS ${scotch_PREFIX}
    PATHS ${scotch_LIBRARY_DIR}
    PATH_SUFFIXES lib lib64 lib32
    DOC "SCOTCH Error-Exit Libraries"
  )


  set( SCOTCH_LIBRARIES 
       ${SCOTCH_LIBRARY} 
       ${SCOTCH_ERR_LIBRARY}
       ${SCOTCH_ERREXIT_LIBRARY} )

  if( "metis" IN_LIST SCOTCH_FIND_COMPONENTS )

    find_library( SCOTCH_METIS_LIBRARY
      NAMES scotchmetis
      HINTS ${scotch_PREFIX}
      PATHS ${scotch_LIBRARY_DIR}
      PATH_SUFFIXES lib lib64 lib32
      DOC "SCOTCH-METIS compatibility Libraries"
    )

    if( SCOTCH_METIS_LIBRARY )
      list( APPEND SCOTCH_LIBRARIES ${SCOTCH_METIS_LIBRARY} )
      set( SCOTCH_metis_FOUND TRUE )
    endif()

  endif()


else()

  # FIXME: Check if files exists at least?
  set( SCOTCH_LIBRARIES ${scotch_LIBRARIES} )

endif()

# Check version
if( EXISTS ${SCOTCH_INCLUDE_DIR}/scotch.h )
  set( version_pattern 
  "^#define[\t ]+SCOTCH_(VERSION|RELEASE|PATCHLEVEL)[\t ]+([0-9\\.]+)$"
  )
  file( STRINGS ${SCOTCH_INCLUDE_DIR}/scotch.h scotch_version
        REGEX ${version_pattern} )
  
  foreach( match ${scotch_version} )
  
    if(SCOTCH_VERSION_STRING)
      set(SCOTCH_VERSION_STRING "${SCOTCH_VERSION_STRING}.")
    endif()
  
    string(REGEX REPLACE ${version_pattern} 
      "${SCOTCH_VERSION_STRING}\\2" 
      SCOTCH_VERSION_STRING ${match}
    )
  
    set(SCOTCH_VERSION_${CMAKE_MATCH_1} ${CMAKE_MATCH_2})
  
  endforeach()
  
  unset( scotch_version )
  unset( version_pattern )
endif()

# Check ILP64
if( EXISTS ${SCOTCH_INCLUDE_DIR}/scotch.h )

  set( idxwidth_pattern
  "^typedef[\t ]+(int64_t|int32_t)[\t ]SCOTCH_Idx\\;$"
  )
  file( STRINGS ${SCOTCH_INCLUDE_DIR}/scotch.h scotch_idxwidth
        REGEX ${idxwidth_pattern} )

  string( REGEX REPLACE ${idxwidth_pattern} 
          "${SCOTCH_IDXWIDTH_STRING}\\1"
          SCOTCH_IDXWIDTH_STRING "${scotch_idxwidth}" )

  if( ${SCOTCH_IDXWIDTH_STRING} MATCHES "int64_t" )
    set( SCOTCH_USES_ILP64 TRUE )
  else()
    set( SCOTCH_USES_ILP64 FALSE )
  endif()

  unset( idxwidth_pattern      )
  unset( scotch_idxwidth        )
  unset( SCOTCH_IDXWIDTH_STRING )

endif()


# Check Threads
if( SCOTCH_LIBRARIES )

  # FIXME: This assumes that threads are even installed
  set( CMAKE_THREAD_PREFER_PTHREAD ON )
  find_package( Threads QUIET )

  include( CMakePushCheckState )

  cmake_push_check_state( RESET )

  set( CMAKE_REQUIRED_LIBRARIES Threads::Threads ${SCOTCH_LIBRARIES} )
  set( CMAKE_REQUIRED_QUIET ON )

  include( CheckLibraryExists )
  #  check_library_exists( "" threadReduce ""
  #    SCOTCH_USES_PTHREADS )
  check_library_exists( "" pthread_create ""
    SCOTCH_USES_PTHREADS )

  cmake_pop_check_state()

endif()


# Handle components
if( SCOTCH_USES_ILP64 )
  set( SCOTCH_ilp64_FOUND TRUE )
endif()

if( SCOTCH_USES_PTHREADS )
  set( SCOTCH_pthreads_FOUND TRUE )
endif()

# Determine if we've found SCOTCH
mark_as_advanced( SCOTCH_FOUND SCOTCH_INCLUDE_DIR SCOTCH_LIBRARIES )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args( SCOTCH
  REQUIRED_VARS SCOTCH_LIBRARIES SCOTCH_INCLUDE_DIR
  VERSION_VAR SCOTCH_VERSION_STRING
  HANDLE_COMPONENTS
)

# Export target
if( SCOTCH_FOUND AND NOT TARGET SCOTCH::scotch )

  add_library( SCOTCH::scotch INTERFACE IMPORTED )
  set_target_properties( SCOTCH::scotch PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${SCOTCH_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES      "${SCOTCH_LIBRARIES}" 
  )

endif()
