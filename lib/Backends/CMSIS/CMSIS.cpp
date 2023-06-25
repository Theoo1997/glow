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

#include "CMSIS.h"

namespace glow {

Expected<std::unique_ptr<CompiledFunction>> CMSISBackend::compile(Function *F) const {
    return nullptr;
}

Expected<std::unique_ptr<CompiledFunction>> CMSISBackend::compile(Function *F, const BackendOptions &opt) const {
    return nullptr;
}

Expected<llvm::StringMap<std::unique_ptr<CompiledFunction>>> CMSISBackend::compileFunctions(std::vector<Function *> &functions,
	llvm::StringMap<BackendOptions> &optsMap) const {
    return llvm::StringMap<std::unique_ptr<CompiledFunction>>();
}

bool CMSISBackend::isOpSupported(const NodeInfo&NI) const {
    return false;
}

unsigned CMSISBackend::numDevices() {
    return 1;
}

std::vector<unsigned> CMSISBackend::scanDeviceIDs() {
    return {0};
}

} // namespace glow
