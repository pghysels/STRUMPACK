#   FindZFP.cmake
#
#   Finds the ZFP library.
# FIXME: check for CUDA support
#
#   This module will define the following variables:
#
#     ZFP_FOUND         - System has found ZFP installation
#     ZFP_INCLUDE_DIR   - Location of ZFP headers
#     ZFP_LIBRARIES     - ZFP libraries
#
#   This module will export the following targets if ZFP_FOUND
#
#     ZFP::zfp
#
#
#
#
#   Proper usage:
#
#     project( TEST_FIND_ZFP C )
#     find_package( ZFP )
#
#     if( ZFP_FOUND )
#       add_executable( test test.cxx )
#       target_link_libraries( test ZFP::zfp )
#     endif()
#
#
#
#
#   This module will use the following variables to change
#   default behaviour if set
#
#     zfp_PREFIX
#     zfp_INCLUDE_DIR
#     zfp_LIBRARY_DIR
#     zfp_LIBRARIES

#==================================================================
#   Copyright (c) 2018 The Regents of the University of California,
#   through Lawrence Berkeley National Laboratory.  
#
#   Author: Pieter Ghysels
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

if( zfp_PREFIX AND NOT zfp_INCLUDE_DIR )
  set( zfp_INCLUDE_DIR ${zfp_PREFIX}/include )
endif()


if( zfp_PREFIX AND NOT zfp_LIBRARY_DIR )
  set( zfp_LIBRARY_DIR ${zfp_PREFIX}/lib )
endif()


# Try to find the header
find_path( ZFP_INCLUDE_DIR 
  NAMES zfp.h
  HINTS ${zfp_PREFIX}
  PATHS ${zfp_INCLUDE_DIR}
  PATH_SUFFIXES include
  DOC "Location of ZFP header"
)

# Try to find libraries if not already set
if( NOT zfp_LIBRARIES )

  find_library( ZFP_LIBRARY
    NAMES zfp
    HINTS ${zfp_PREFIX}
    PATHS ${zfp_LIBRARY_DIR}
    PATH_SUFFIXES lib lib64 lib32
    DOC "ZFP Library"
  )

  set( ZFP_LIBRARIES ${ZFP_LIBRARY} )

else()

  # FIXME: Check if files exists at least?
  set( ZFP_LIBRARIES ${zfp_LIBRARIES} )

endif()

# Determine if we've found ZFP
mark_as_advanced( ZFP_FOUND ZFP_INCLUDE_DIR ZFP_LIBRARIES )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args( ZFP
  REQUIRED_VARS ZFP_LIBRARIES ZFP_INCLUDE_DIR
  )

# Export target
if( ZFP_FOUND AND NOT TARGET ZFP::zfp )

  add_library( ZFP::zfp INTERFACE IMPORTED )
  set_target_properties( ZFP::zfp PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ZFP_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES      "${ZFP_LIBRARIES}"
    )

endif()
