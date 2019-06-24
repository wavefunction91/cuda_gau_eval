#pragma once
#include <chrono>
#include <map>
#include <string>
#include <iostream>
#include <iomanip>

class Timer {

  std::map< std::string, std::chrono::duration<double> >
    timings_;

public:

  Timer() = default;
  ~Timer() noexcept = default;
  Timer(const Timer&) = default;
  Timer( Timer&& ) noexcept = default;


  template <typename Func>
  void time( const std::string &name, const Func& f) {

    auto f_st = std::chrono::high_resolution_clock::now();
    f();
    auto f_en = std::chrono::high_resolution_clock::now();

    if( timings_.find( name ) != timings_.end() )
      timings_[name] += f_en - f_st;
    else
      timings_[name] = f_en - f_st;

  }

  const decltype(timings_)& timings() const { return timings_; }

};

inline std::ostream& operator<<( std::ostream& out, const Timer& timer ) {

  const auto& timings = timer.timings();
  // Get longest string
  auto max_str_it = std::max_element( timings.begin(), timings.end(), [](const auto& a, const auto& b) { return a.first.size() < b.first.size(); } );

  size_t max_len = max_str_it->first.size();

  out << "Timings:" << std::endl;
  for( const auto& t : timings ) {

    out << "  " << std::left << std::setw(max_len) << t.first << ":  " << t.second.count() << std::endl;

  }

  return out;
}
