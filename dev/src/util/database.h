#pragma once

#include "../inc.h"


namespace database {

// 2d classical Ising model
// 4 indices: up, right, down, left, 
ITensor ising2d(double beta);

ITensor ising2d_gauge(double beta);

// ground state of fibonacci string net 
// anti-Y-shaped
ITensor fibo_avtx();

// ground state of fibonacci string net 
// Y-shaped
ITensor fibo_vtx();

// fibo building block, 4-leg tensor
// direct contraction of Vtx and aVtx.
// remaining indices have tags "ul", "ur", "ll", "lr"
ITensor fibo_vav();

// fibo building block after svd, 4-leg tensor
// horizontal index dim=5, vertical index dim=3
ITensor fibo_svd_bd();


// obtain the contraction of 4 copies of tensor A
// A must have indices with tags "u", "r", "l", "d"
ITensor x4(const ITensor& A);

}