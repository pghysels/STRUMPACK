cmake_minimum_required( VERSION 3.11 )

include( CMakePushCheckState )
include( CheckLibraryExists )
include( CheckSymbolExists )
include( CMakeFindDependencyMacro )
include( FindPackageHandleStandardArgs )

include( ${CMAKE_CURRENT_LIST_DIR}/CommonFunctions.cmake )

fill_out_prefix( magma )

find_path( MAGMA_INCLUDE_DIR
  NAMES magma.h
  HINTS ${magma_PREFIX}
  PATHS ${magma_INCLUDE_DIR}
  PATH_SUFFIXES include
  DOS "local of MAGMA header"
)

if( NOT magma_LIBRARIES )

  find_library( MAGMA_LIBRARIES
    NAMES magma
    HINTS ${magma_PREFIX}
    PATHS ${magma_LIBRARY_DIR}
    PATH_SUFFIXES lib lib64 lib32
    DOC "MAGMA Libraries"
  )

else()

  # FIXME: Check if files exists at least?
  set( MAGMA_LIBRARIES ${magma_LIBRARIES} )

endif()



# Check version
if( EXISTS ${MAGMA_INCLUDE_DIR}/magma_types.h )
  set( version_pattern 
  "^#define[\t ]+MAGMA_VERSION_(MAJOR|MINOR|MICRO)[\t ]+([0-9\\.]+)$"
  )
  file( STRINGS ${MAGMA_INCLUDE_DIR}/magma_types.h magma_version
        REGEX ${version_pattern} )
  
  foreach( match ${magma_version} )
  
    if(MAGMA_VERSION_STRING)
      set(MAGMA_VERSION_STRING "${MAGMA_VERSION_STRING}.")
    endif()
  
    string(REGEX REPLACE ${version_pattern} 
      "${MAGMA_VERSION_STRING}\\2" 
      MAGMA_VERSION_STRING ${match}
    )
  
    set(MAGMA_VERSION_${CMAKE_MATCH_1} ${CMAKE_MATCH_2})
  
  endforeach()
  
  unset( magma_version )
  unset( version_pattern )
endif()


# Determine if we've found MAGMA
mark_as_advanced( MAGMA_FOUND MAGMA_INCLUDE_DIR MAGMA_LIBRARIES )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args( MAGMA
  REQUIRED_VARS MAGMA_LIBRARIES MAGMA_INCLUDE_DIR
  VERSION_VAR MAGMA_VERSION_STRING
  HANDLE_COMPONENTS
)

# Export target
if( MAGMA_FOUND AND NOT TARGET MAGMA::magma )

  add_library( MAGMA::magma INTERFACE IMPORTED )
  set_target_properties( MAGMA::magma PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${MAGMA_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES      "${MAGMA_LIBRARIES}" 
  )

endif()
