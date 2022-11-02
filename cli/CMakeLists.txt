add_executable(ppocra ppocra.cpp helper.cpp)
target_link_libraries(ppocra PRIVATE paddleocr)
set_target_properties(ppocra PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
)
install(TARGETS ppocra)
install(FILES config.ini DESTINATION bin)

if(MSVC)
install(FILES $<TARGET_PDB_FILE:ppocra> DESTINATION bin OPTIONAL)
endif()