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

function( fill_out_prefix name )

  if( ${name}_PREFIX AND NOT ${name}_INCLUDE_DIR )
    set( ${name}_INCLUDE_DIR ${${name}_PREFIX}/include PARENT_SCOPE )
  endif()

  if( ${name}_PREFIX AND NOT ${name}_LIBRARY_DIR )
    set( ${name}_LIBRARY_DIR 
         "${${name}_PREFIX}/lib;${${name}_PREFIX}/lib32;${${name}_PREFIX}/lib64"
         PARENT_SCOPE
    )
  endif()

endfunction()

function( copy_meta_data _src _dest )

  if( ${_src}_LIBRARIES AND NOT ${_dest}_LIBRARIES )
    set( ${_dest}_LIBRARIES ${${_src}_LIBRARIES} PARENT_SCOPE )
  endif()

  if( ${_src}_PREFIX AND NOT ${_dest}_PREFIX )
    set( ${_dest}_PREFIX ${${_src}_PREFIX} PARENT_SCOPE )
  endif()

  if( ${_src}_INCLUDE_DIR AND NOT ${_dest}_INCLUDE_DIR )
    set( ${_dest}_INCLUDE_DIR ${${_src}_INCLUDE_DIR} PARENT_SCOPE )
  endif()

  if( ${_src}_LIBRARY_DIR AND NOT ${_dest}_LIBRARY_DIR )
    set( ${_dest}_LIBRARY_DIR ${${_src}_LIBRARY_DIR} PARENT_SCOPE )
  endif()

endfunction()


function( get_true_target_property _out _target _property )

  if( TARGET ${_target} )
    get_property( _${_target}_imported TARGET ${_target} PROPERTY IMPORTED )

    if( NOT ${_property} MATCHES "INTERFACE_LINK_LIBRARIES" )
      get_property( _${_target}_property TARGET ${_target} PROPERTY ${_property} )
    endif()

    if( _${_target}_imported )

      #message( STATUS "${_target} is IMPORTED" )

      get_property( _${_target}_link TARGET ${_target} PROPERTY INTERFACE_LINK_LIBRARIES )
      foreach( _lib ${_${_target}_link} )
        #message( STATUS "Checking ${_lib}")
        if( TARGET ${_lib} )
          get_true_target_property( _${_lib}_property ${_lib} ${_property} )
          #message( STATUS "${_lib} is a TARGET with ${_${_lib}_property}" )
          if( _${_lib}_property )
            list( APPEND _${_target}_property_imported ${_${_lib}_property} )
          endif()
        elseif( ${_property} MATCHES "INTERFACE_LINK_LIBRARIES" )
          list( APPEND _${_target}_property_imported ${_lib} )
        endif()
      endforeach()
      if(_${_target}_property_imported)
        list(APPEND _${_target}_property ${_${_target}_property_imported} )
      endif()
      set( ${_out} ${_${_target}_property} PARENT_SCOPE )
    else()
      #message( STATUS "${_target} is NOT IMPORTED" )
      #message( STATUS "Setting ${_out} to ${_${_target}_property} " )
      set( ${_out} ${_${_target}_property} PARENT_SCOPE )
    endif()
  endif()

endfunction()
