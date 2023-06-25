add_library(CMSIS
    	    CMSIS.cpp
	    CMSISCompiledFunction.cpp
	    CMSISFactory.cpp)

target_link_libraries(CMSIS
    		      PRIVATE
			Backend
			Base
			Graph
			CodeGen
			IR
			LLVMCore
			QuantizationBase
			Runtime
			Support)

set(linked_backends ${linked_backends} CMSIS PARENT_SCOPE)
