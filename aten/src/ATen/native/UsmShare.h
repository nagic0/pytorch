#pragma once

#include <ATen/native/DispatchStub.h>
#include <c10/core/Device.h>
#include <c10/core/Storage.h>


namespace at::native {

using usm_share_fn = c10::Storage (*)(const c10::Storage&, const c10::Device&);

DECLARE_DISPATCH(usm_share_fn, usm_share_stub);

} // namespace at::native

