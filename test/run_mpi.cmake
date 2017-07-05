# execute the test command that was added earlier.

set(ENV{OMP_NUM_THREADS} ${TH})
execute_process(
  COMMAND mpirun -n ${MPIPROCS} ${TEST}
  ${ARG1} ${ARG2} ${ARG3} ${ARG4}
  ${ARG5} ${ARG6} ${ARG7} ${ARG8} ${ARG9} ${ARG10}
  ${ARG11} ${ARG12} ${ARG13} ${ARG14} ${ARG15}
  TIMEOUT 60
  OUTPUT_FILE ${OUTPUT}
  RESULT_VARIABLE RET
  )

file(APPEND ${ALL_OUTPUT} ${HEADING})
file(APPEND ${ALL_OUTPUT} "\n")
file(READ   ${OUTPUT} single_output)
file(APPEND ${ALL_OUTPUT} ${single_output})
file(APPEND ${ALL_OUTPUT} "\n")
file(APPEND ${ALL_OUTPUT} "\n")
file(REMOVE ${OUTPUT})   # remove the individual output file.

if (NOT "${RET}" STREQUAL "0")
  message(FATAL_ERROR "TEST FAILED!!!")
endif()
