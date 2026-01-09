# ============================================================================
# Subsetix Kokkos Library Targets
# ============================================================================

# Create the main library target as an INTERFACE library
add_library(subsetix_core INTERFACE)

# Set include directories
target_include_directories(subsetix_core
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link Kokkos
target_link_libraries(subsetix_core
  INTERFACE
    Kokkos::kokkos
)

# Require C++20
target_compile_features(subsetix_core INTERFACE cxx_std_20)

# Alias for convenience
add_library(subsetix::core ALIAS subsetix_core)

# Installation rules (optional, for future packaging)
include(GNUInstallDirs)

install(TARGETS subsetix_core
  EXPORT subsetix_targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT subsetix_targets
  FILE subsetix_targets.cmake
  NAMESPACE subsetix::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/subsetix
)
