#pragma once
#include <utility>

template <typename... Args>
std::array<void*,sizeof...(Args)> conv_pack( Args&& ...args ) {

  return { ((void*)&std::forward<Args>(args))... };

}
