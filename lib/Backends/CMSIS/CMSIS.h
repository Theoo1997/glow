/*
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GLOW_BACKENDS_CMSISBackend
#define GLOW_BACKENDS_CMSISBackend

#include <memory>
#include <vector>
#include <string>

#include <glow/Backend/Backend.h>
#include <glow/Backend/CompiledFunction.h>

namespace glow {

struct CMSISBackend: public Backend {
    Expected<std::unique_ptr<CompiledFunction>> compile(Function *F) const override;

    Expected<std::unique_ptr<CompiledFunction>> compile(Function *F, const BackendOptions &opt) const override;

    Expected<llvm::StringMap<std::unique_ptr<CompiledFunction>>>
    compileFunctions(std::vector<Function *> &functions,
	    	     llvm::StringMap<BackendOptions> &optsMap) const override;

    bool isOpSupported(const NodeInfo &NI) const override;

    std::string getBackendName() const override { return getName(); }
    static std::string getName() { return "CMSIS"; }

    static unsigned numDevices();
    static std::vector<unsigned> scanDeviceIDs();
};

} // namespace glow

#endif
