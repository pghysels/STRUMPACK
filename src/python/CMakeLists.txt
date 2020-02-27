configure_file(STRUMPACKKernel.py.in
  ${PROJECT_BINARY_DIR}/STRUMPACKKernel.py)

install(FILES
  ${PROJECT_BINARY_DIR}/STRUMPACKKernel.py
  DESTINATION include/python)
