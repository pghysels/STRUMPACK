#   FindMETIS.cmake
#
#   Finds the METIS library.
#
#   This module will define the following variables:
#   
#     METIS_FOUND        - System has found METIS installation
#     METIS_INCLUDE_DIR  - Location of METIS headers
#     METIS_LIBRARIES    - METIS libraries
#     METIS_USES_ILP64   - Whether METIS was configured with ILP64
#
#   This module can handle the following COMPONENTS
#
#     ilp64 - 64-bit index integers
#
#   This module will export the following targets if METIS_FOUND
#
#     METIS::metis
#
#
#
#
#   Proper usage:
#
#     project( TEST_FIND_METIS C )
#     find_package( METIS )
#
#     if( METIS_FOUND )
#       add_executable( test test.cxx )
#       target_link_libraries( test METIS::metis )
#     endif()
#
#
#
#
#   This module will use the following variables to change
#   default behaviour if set
#
#     metis_PREFIX
#     metis_INCLUDE_DIR
#     metis_LIBRARY_DIR
#     metis_LIBRARIES

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

if( metis_PREFIX AND NOT metis_INCLUDE_DIR )
  set( metis_INCLUDE_DIR ${metis_PREFIX}/include )
endif()


if( metis_PREFIX AND NOT metis_LIBRARY_DIR )
  set( metis_LIBRARY_DIR 
    ${metis_PREFIX}/lib 
    ${metis_PREFIX}/lib32 
    ${metis_PREFIX}/lib64 
  )
endif()


# Try to find the header
find_path( METIS_INCLUDE_DIR 
  NAMES metis.h
  HINTS ${metis_PREFIX}
  PATHS ${metis_INCLUDE_DIR}
  PATH_SUFFIXES include
  DOC "Location of METIS header"
)

# Try to find libraries if not already set
if( NOT metis_LIBRARIES )

  find_library( METIS_LIBRARIES
    NAMES metis
    HINTS ${metis_PREFIX}
    PATHS ${metis_LIBRARY_DIR}
    PATH_SUFFIXES lib lib64 lib32
    DOC "METIS Libraries"
  )

else()

  # FIXME: Check if files exists at least?
  set( METIS_LIBRARIES ${metis_LIBRARIES} )

endif()

# Check version
if( EXISTS ${METIS_INCLUDE_DIR}/metis.h )
  set( version_pattern 
  "^#define[\t ]+METIS_VER_(MAJOR|MINOR|SUBMINOR)[\t ]+([0-9\\.]+)$"
  )
  file( STRINGS ${METIS_INCLUDE_DIR}/metis.h metis_version
        REGEX ${version_pattern} )
  
  foreach( match ${metis_version} )
  
    if(METIS_VERSION_STRING)
      set(METIS_VERSION_STRING "${METIS_VERSION_STRING}.")
    endif()
  
    string(REGEX REPLACE ${version_pattern} 
      "${METIS_VERSION_STRING}\\2" 
      METIS_VERSION_STRING ${match}
    )
  
    set(METIS_VERSION_${CMAKE_MATCH_1} ${CMAKE_MATCH_2})
  
  endforeach()
  
  unset( metis_version )
  unset( version_pattern )
endif()

# Check ILP64
if( EXISTS ${METIS_INCLUDE_DIR}/metis.h )

  set( idxwidth_pattern
  "^#define[\t ]+IDXTYPEWIDTH[\t ]+([0-9\\.]+[0-9\\.]+)$"
  )
  file( STRINGS ${METIS_INCLUDE_DIR}/metis.h metis_idxwidth
        REGEX ${idxwidth_pattern} )

  string( REGEX REPLACE ${idxwidth_pattern} 
          "${METIS_IDXWIDTH_STRING}\\1"
          METIS_IDXWIDTH_STRING ${metis_idxwidth} )

  if( ${METIS_IDXWIDTH_STRING} MATCHES "64" )
    set( METIS_USES_ILP64 TRUE )
  else()
    set( METIS_USES_ILP64 FALSE )
  endif()

  unset( idxwidth_pattern      )
  unset( metis_idxwidth        )
  unset( METIS_IDXWIDTH_STRING )

endif()



# Handle components
if( METIS_USES_ILP64 )
  set( METIS_ilp64_FOUND TRUE )
endif()

# Determine if we've found METIS
mark_as_advanced( METIS_FOUND METIS_INCLUDE_DIR METIS_LIBRARIES )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args( METIS
  REQUIRED_VARS METIS_LIBRARIES METIS_INCLUDE_DIR
  VERSION_VAR METIS_VERSION_STRING
  HANDLE_COMPONENTS
)

# Export target
if( METIS_FOUND AND NOT TARGET METIS::metis )

  add_library( METIS::metis INTERFACE IMPORTED )
  set_target_properties( METIS::metis PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${METIS_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES      "${METIS_LIBRARIES}" 
  )

endif()
