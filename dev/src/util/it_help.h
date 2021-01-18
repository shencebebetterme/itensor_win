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


std::string myprint(TagSet ts);
std::string myprint(Index idx);
std::string myprint(IndexSet is);
std::string myprint(ITensor A);