#pragma once

#include "../pch.h"

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

vector_no_init<Real>* peek(const ITensor& A);
vector_no_init<Cplx>* peek_cx(const ITensor& A);