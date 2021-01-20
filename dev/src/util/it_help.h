#pragma once

#include "itensor/all_basic.h"
#include "itensor/util/print_macro.h"
using namespace itensor;

double get_max(ITensor&);
double get_max_abs(ITensor&);
double get_sum(ITensor&);

ITensor trivialIT();
void show_shape(const ITensor&);
void show_shape(const IndexSet&);


std::string myprint_ts(const TagSet& ts);
std::string myprint_idx(const Index& idx);
std::string myprint_is(const IndexSet& is);
std::string myprint(const ITensor& A);
std::vector<double> mypeek(const ITensor& A);