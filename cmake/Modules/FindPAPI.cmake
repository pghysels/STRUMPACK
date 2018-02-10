if (NOT (PAPI_INCLUDES AND PAPI_LIBRARIES))
  find_path(PAPI_INCLUDES NAMES papi.h PATHS $ENV{PAPIDIR} PATH_SUFFIXES include)
  find_library(PAPI_LIBRARY papi PATHS $ENV{PAPIDIR} PATH_SUFFIXES lib)
  set(PAPI_LIBRARIES ${METIS_LIBRARY})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PAPI DEFAULT_MSG PAPI_INCLUDES PAPI_LIBRARIES)
