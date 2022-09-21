add_executable(ppocra ppocra.cpp)
target_link_libraries(ppocra PRIVATE paddleocr)
set_target_properties(ppocra PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
)
install(TARGETS ppocra)
